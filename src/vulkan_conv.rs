use anyhow::{Context, Result, anyhow, bail};
use half::f16;
use naga::{
    back::spv::{Options as SpvOptions, PipelineOptions as SpvPipelineOptions, write_vec},
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};
use vulkano::{
    Version, VulkanLibrary,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDevice,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    library::DynamicLibraryLoader,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::{ComputePipeline, ComputePipelineCreateInfo},
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
    sync::{self as vk_sync, GpuFuture},
};

use crate::{
    fused_weights::{Conv2dSpec, FusedConv2dWeights},
    tensor_ops::TensorShape,
    yolox_blocks::{
        BaseConvBlock, BottleneckBlock, ConvBlock, CspBottleneckBlock, CspDarknetDemo,
        CspStageBlock, Dark5Block, DarknetFeatures, DecodedPredictions, DwsConvBlock,
        FocusStemBlock, HeadFeatures, PafpnFeatures, SppBottleneckBlock, YoloxDecodeDemo,
        YoloxHeadDemo, YoloxHeadScaleDemo, YoloxPafpnDemo,
    },
};

const IMAGE_LOCAL_SIZE_X: u32 = 8;
const IMAGE_LOCAL_SIZE_Y: u32 = 8;
const FP16_CONV3X3_LOCAL_SIZE_X: u32 = 8;
const FP16_CONV3X3_LOCAL_SIZE_Y: u32 = 8;
const FP16_CONV3X3_OUTPUT_CHANNELS_PER_INVOCATION: u32 = 2;
const LINEAR_LOCAL_SIZE_X: u32 = 64;
const MATRIX_LOCAL_SIZE_X: u32 = 16;
const MATRIX_LOCAL_SIZE_Y: u32 = 16;

const CONV2D_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> weight_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> bias_buffer: BufferF32;
@group(0) @binding(3) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(4) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc0 = gid.z * 2u;
    let oc1 = oc0 + 1u;

    let in_channels = params_buffer.data[0];
    let out_channels = params_buffer.data[1];
    let input_h = params_buffer.data[2];
    let input_w = params_buffer.data[3];
    let kernel_h = params_buffer.data[4];
    let kernel_w = params_buffer.data[5];
    let stride_h = params_buffer.data[6];
    let stride_w = params_buffer.data[7];
    let pad_h = params_buffer.data[8];
    let pad_w = params_buffer.data[9];
    let groups = params_buffer.data[10];
    let in_channels_per_group = params_buffer.data[11];
    let output_h = params_buffer.data[12];
    let output_w = params_buffer.data[13];

    if (ox >= output_w || oy >= output_h || oc >= out_channels) { return; }

    var acc = bias_buffer.data[oc];
    let out_channels_per_group = out_channels / groups;
    let group = oc / out_channels_per_group;
    let ic_start = group * in_channels_per_group;
    let ic_end = ic_start + in_channels_per_group;

    for (var ic: u32 = ic_start; ic < ic_end; ic = ic + 1u) {
        let group_ic = ic - ic_start;
        for (var ky: u32 = 0u; ky < kernel_h; ky = ky + 1u) {
            let src_y = i32(oy * stride_h + ky) - i32(pad_h);
            if (src_y < 0 || src_y >= i32(input_h)) { continue; }
            for (var kx: u32 = 0u; kx < kernel_w; kx = kx + 1u) {
                let src_x = i32(ox * stride_w + kx) - i32(pad_w);
                if (src_x < 0 || src_x >= i32(input_w)) { continue; }
                let input_index = ((ic * input_h + u32(src_y)) * input_w) + u32(src_x);
                let weight_index = (((oc * in_channels_per_group + group_ic) * kernel_h + ky) * kernel_w) + kx;
                acc = acc + input_buffer.data[input_index] * weight_buffer.data[weight_index];
            }
        }
    }

    let output_index = ((oc * output_h + oy) * output_w) + ox;
    output_buffer.data[output_index] = acc;
}
"#;

const CONV2D_SILU_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> weight_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> bias_buffer: BufferF32;
@group(0) @binding(3) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(4) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;

    let out_channels = params_buffer.data[1];
    let input_h = params_buffer.data[2];
    let input_w = params_buffer.data[3];
    let kernel_h = params_buffer.data[4];
    let kernel_w = params_buffer.data[5];
    let stride_h = params_buffer.data[6];
    let stride_w = params_buffer.data[7];
    let pad_h = params_buffer.data[8];
    let pad_w = params_buffer.data[9];
    let groups = params_buffer.data[10];
    let in_channels_per_group = params_buffer.data[11];
    let output_h = params_buffer.data[12];
    let output_w = params_buffer.data[13];

    if (ox >= output_w || oy >= output_h || oc >= out_channels) { return; }

    var acc = bias_buffer.data[oc];
    let out_channels_per_group = out_channels / groups;
    let group = oc / out_channels_per_group;
    let ic_start = group * in_channels_per_group;
    let ic_end = ic_start + in_channels_per_group;

    for (var ic: u32 = ic_start; ic < ic_end; ic = ic + 1u) {
        let group_ic = ic - ic_start;
        for (var ky: u32 = 0u; ky < kernel_h; ky = ky + 1u) {
            let src_y = i32(oy * stride_h + ky) - i32(pad_h);
            if (src_y < 0 || src_y >= i32(input_h)) { continue; }
            for (var kx: u32 = 0u; kx < kernel_w; kx = kx + 1u) {
                let src_x = i32(ox * stride_w + kx) - i32(pad_w);
                if (src_x < 0 || src_x >= i32(input_w)) { continue; }
                let input_index = ((ic * input_h + u32(src_y)) * input_w) + u32(src_x);
                let weight_index = (((oc * in_channels_per_group + group_ic) * kernel_h + ky) * kernel_w) + kx;
                acc = acc + input_buffer.data[input_index] * weight_buffer.data[weight_index];
            }
        }
    }

    let output_index = ((oc * output_h + oy) * output_w) + ox;
    output_buffer.data[output_index] = acc / (1.0 + exp(-acc));
}
"#;

const CONV2D_FP16_WEIGHTS_WGSL: &str = r#"
enable f16;

struct BufferF32 { data: array<f32>, }
struct BufferF16 { data: array<f16>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> weight_buffer: BufferF16;
@group(0) @binding(2) var<storage, read> bias_buffer: BufferF32;
@group(0) @binding(3) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(4) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;

    let in_channels = params_buffer.data[0];
    let out_channels = params_buffer.data[1];
    let input_h = params_buffer.data[2];
    let input_w = params_buffer.data[3];
    let kernel_h = params_buffer.data[4];
    let kernel_w = params_buffer.data[5];
    let stride_h = params_buffer.data[6];
    let stride_w = params_buffer.data[7];
    let pad_h = params_buffer.data[8];
    let pad_w = params_buffer.data[9];
    let groups = params_buffer.data[10];
    let in_channels_per_group = params_buffer.data[11];
    let output_h = params_buffer.data[12];
    let output_w = params_buffer.data[13];

    if (ox >= output_w || oy >= output_h || oc >= out_channels) { return; }

    var acc = f16(bias_buffer.data[oc]);
    let out_channels_per_group = out_channels / groups;
    let group = oc / out_channels_per_group;
    let ic_start = group * in_channels_per_group;
    let ic_end = ic_start + in_channels_per_group;

    for (var ic: u32 = ic_start; ic < ic_end; ic = ic + 1u) {
        let group_ic = ic - ic_start;
        for (var ky: u32 = 0u; ky < kernel_h; ky = ky + 1u) {
            let src_y = i32(oy * stride_h + ky) - i32(pad_h);
            if (src_y < 0 || src_y >= i32(input_h)) { continue; }
            for (var kx: u32 = 0u; kx < kernel_w; kx = kx + 1u) {
                let src_x = i32(ox * stride_w + kx) - i32(pad_w);
                if (src_x < 0 || src_x >= i32(input_w)) { continue; }
                let input_index = ((ic * input_h + u32(src_y)) * input_w) + u32(src_x);
                let weight_index = (((oc * in_channels_per_group + group_ic) * kernel_h + ky) * kernel_w) + kx;
                acc = acc + f16(input_buffer.data[input_index]) * weight_buffer.data[weight_index];
            }
        }
    }

    let output_index = ((oc * output_h + oy) * output_w) + ox;
    output_buffer.data[output_index] = f32(acc);
}
"#;

const CONV2D_SILU_FP16_WEIGHTS_WGSL: &str = r#"
enable f16;

struct BufferF32 { data: array<f32>, }
struct BufferF16 { data: array<f16>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> weight_buffer: BufferF16;
@group(0) @binding(2) var<storage, read> bias_buffer: BufferF32;
@group(0) @binding(3) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(4) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;

    let out_channels = params_buffer.data[1];
    let input_h = params_buffer.data[2];
    let input_w = params_buffer.data[3];
    let kernel_h = params_buffer.data[4];
    let kernel_w = params_buffer.data[5];
    let stride_h = params_buffer.data[6];
    let stride_w = params_buffer.data[7];
    let pad_h = params_buffer.data[8];
    let pad_w = params_buffer.data[9];
    let groups = params_buffer.data[10];
    let in_channels_per_group = params_buffer.data[11];
    let output_h = params_buffer.data[12];
    let output_w = params_buffer.data[13];

    if (ox >= output_w || oy >= output_h || oc >= out_channels) { return; }

    var acc = f16(bias_buffer.data[oc]);
    let out_channels_per_group = out_channels / groups;
    let group = oc / out_channels_per_group;
    let ic_start = group * in_channels_per_group;
    let ic_end = ic_start + in_channels_per_group;

    for (var ic: u32 = ic_start; ic < ic_end; ic = ic + 1u) {
        let group_ic = ic - ic_start;
        for (var ky: u32 = 0u; ky < kernel_h; ky = ky + 1u) {
            let src_y = i32(oy * stride_h + ky) - i32(pad_h);
            if (src_y < 0 || src_y >= i32(input_h)) { continue; }
            for (var kx: u32 = 0u; kx < kernel_w; kx = kx + 1u) {
                let src_x = i32(ox * stride_w + kx) - i32(pad_w);
                if (src_x < 0 || src_x >= i32(input_w)) { continue; }
                let input_index = ((ic * input_h + u32(src_y)) * input_w) + u32(src_x);
                let weight_index = (((oc * in_channels_per_group + group_ic) * kernel_h + ky) * kernel_w) + kx;
                acc = acc + f16(input_buffer.data[input_index]) * weight_buffer.data[weight_index];
            }
        }
    }

    let acc32 = f32(acc);
    let output_index = ((oc * output_h + oy) * output_w) + ox;
    output_buffer.data[output_index] = acc32 / (1.0 + exp(-acc32));
}
"#;

const CONV2D_1X1_FP16_WEIGHTS_WGSL: &str = r#"
enable f16;

struct BufferF32 { data: array<f32>, }
struct BufferF16 { data: array<f16>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> weight_buffer: BufferF16;
@group(0) @binding(2) var<storage, read> bias_buffer: BufferF32;
@group(0) @binding(3) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(4) var<storage, read> params_buffer: BufferU32;

fn dot4_f16(a: vec4<f16>, b: vec4<f16>) -> f16 {
    let prod = a * b;
    return prod.x + prod.y + prod.z + prod.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;

    let out_channels = params_buffer.data[1];
    let input_h = params_buffer.data[2];
    let input_w = params_buffer.data[3];
    let stride_h = params_buffer.data[6];
    let stride_w = params_buffer.data[7];
    let groups = params_buffer.data[10];
    let in_channels_per_group = params_buffer.data[11];
    let output_h = params_buffer.data[12];
    let output_w = params_buffer.data[13];

    if (ox >= output_w || oy >= output_h || oc >= out_channels) { return; }

    var acc = f16(bias_buffer.data[oc]);
    let out_channels_per_group = out_channels / groups;
    let group = oc / out_channels_per_group;
    let ic_start = group * in_channels_per_group;
    let ic_end = ic_start + in_channels_per_group;
    let src_y = oy * stride_h;
    let src_x = ox * stride_w;

    var ic = ic_start;
    loop {
        if (ic + 3u >= ic_end) {
            break;
        }

        let group_ic = ic - ic_start;
        let input_index0 = ((ic * input_h + src_y) * input_w) + src_x;
        let input_index1 = (((ic + 1u) * input_h + src_y) * input_w) + src_x;
        let input_index2 = (((ic + 2u) * input_h + src_y) * input_w) + src_x;
        let input_index3 = (((ic + 3u) * input_h + src_y) * input_w) + src_x;
        let weight_index = (oc * in_channels_per_group) + group_ic;

        let input_vec = vec4<f16>(
            f16(input_buffer.data[input_index0]),
            f16(input_buffer.data[input_index1]),
            f16(input_buffer.data[input_index2]),
            f16(input_buffer.data[input_index3]),
        );
        let weight_vec = vec4<f16>(
            weight_buffer.data[weight_index],
            weight_buffer.data[weight_index + 1u],
            weight_buffer.data[weight_index + 2u],
            weight_buffer.data[weight_index + 3u],
        );
        acc = acc + dot4_f16(input_vec, weight_vec);
        ic = ic + 4u;
    }

    for (; ic < ic_end; ic = ic + 1u) {
        let group_ic = ic - ic_start;
        let input_index = ((ic * input_h + src_y) * input_w) + src_x;
        let weight_index = (oc * in_channels_per_group) + group_ic;
        acc = acc + f16(input_buffer.data[input_index]) * weight_buffer.data[weight_index];
    }

    let output_index = ((oc * output_h + oy) * output_w) + ox;
    output_buffer.data[output_index] = f32(acc);
}
"#;

const CONV2D_1X1_SILU_FP16_WEIGHTS_WGSL: &str = r#"
enable f16;

struct BufferF32 { data: array<f32>, }
struct BufferF16 { data: array<f16>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> weight_buffer: BufferF16;
@group(0) @binding(2) var<storage, read> bias_buffer: BufferF32;
@group(0) @binding(3) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(4) var<storage, read> params_buffer: BufferU32;

fn dot4_f16(a: vec4<f16>, b: vec4<f16>) -> f16 {
    let prod = a * b;
    return prod.x + prod.y + prod.z + prod.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;

    let out_channels = params_buffer.data[1];
    let input_h = params_buffer.data[2];
    let input_w = params_buffer.data[3];
    let stride_h = params_buffer.data[6];
    let stride_w = params_buffer.data[7];
    let groups = params_buffer.data[10];
    let in_channels_per_group = params_buffer.data[11];
    let output_h = params_buffer.data[12];
    let output_w = params_buffer.data[13];

    if (ox >= output_w || oy >= output_h || oc >= out_channels) { return; }

    var acc = f16(bias_buffer.data[oc]);
    let out_channels_per_group = out_channels / groups;
    let group = oc / out_channels_per_group;
    let ic_start = group * in_channels_per_group;
    let ic_end = ic_start + in_channels_per_group;
    let src_y = oy * stride_h;
    let src_x = ox * stride_w;

    var ic = ic_start;
    loop {
        if (ic + 3u >= ic_end) {
            break;
        }

        let group_ic = ic - ic_start;
        let input_index0 = ((ic * input_h + src_y) * input_w) + src_x;
        let input_index1 = (((ic + 1u) * input_h + src_y) * input_w) + src_x;
        let input_index2 = (((ic + 2u) * input_h + src_y) * input_w) + src_x;
        let input_index3 = (((ic + 3u) * input_h + src_y) * input_w) + src_x;
        let weight_index = (oc * in_channels_per_group) + group_ic;

        let input_vec = vec4<f16>(
            f16(input_buffer.data[input_index0]),
            f16(input_buffer.data[input_index1]),
            f16(input_buffer.data[input_index2]),
            f16(input_buffer.data[input_index3]),
        );
        let weight_vec = vec4<f16>(
            weight_buffer.data[weight_index],
            weight_buffer.data[weight_index + 1u],
            weight_buffer.data[weight_index + 2u],
            weight_buffer.data[weight_index + 3u],
        );
        acc = acc + dot4_f16(input_vec, weight_vec);
        ic = ic + 4u;
    }

    for (; ic < ic_end; ic = ic + 1u) {
        let group_ic = ic - ic_start;
        let input_index = ((ic * input_h + src_y) * input_w) + src_x;
        let weight_index = (oc * in_channels_per_group) + group_ic;
        acc = acc + f16(input_buffer.data[input_index]) * weight_buffer.data[weight_index];
    }

    let acc32 = f32(acc);
    let output_index = ((oc * output_h + oy) * output_w) + ox;
    output_buffer.data[output_index] = acc32 / (1.0 + exp(-acc32));
}
"#;

const CONV2D_1X1_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> weight_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> bias_buffer: BufferF32;
@group(0) @binding(3) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(4) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;

    let out_channels = params_buffer.data[1];
    let input_h = params_buffer.data[2];
    let input_w = params_buffer.data[3];
    let stride_h = params_buffer.data[6];
    let stride_w = params_buffer.data[7];
    let groups = params_buffer.data[10];
    let in_channels_per_group = params_buffer.data[11];
    let output_h = params_buffer.data[12];
    let output_w = params_buffer.data[13];

    if (ox >= output_w || oy >= output_h || oc >= out_channels) { return; }

    var acc = bias_buffer.data[oc];
    let out_channels_per_group = out_channels / groups;
    let group = oc / out_channels_per_group;
    let ic_start = group * in_channels_per_group;
    let ic_end = ic_start + in_channels_per_group;
    let src_y = oy * stride_h;
    let src_x = ox * stride_w;

    for (var ic: u32 = ic_start; ic < ic_end; ic = ic + 1u) {
        let group_ic = ic - ic_start;
        let input_index = ((ic * input_h + src_y) * input_w) + src_x;
        let weight_index = (oc * in_channels_per_group) + group_ic;
        acc = acc + input_buffer.data[input_index] * weight_buffer.data[weight_index];
    }

    let output_index = ((oc * output_h + oy) * output_w) + ox;
    output_buffer.data[output_index] = acc;
}
"#;

const CONV2D_1X1_SILU_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> weight_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> bias_buffer: BufferF32;
@group(0) @binding(3) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(4) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;

    let in_channels = params_buffer.data[0];
    let out_channels = params_buffer.data[1];
    let input_h = params_buffer.data[2];
    let input_w = params_buffer.data[3];
    let stride_h = params_buffer.data[6];
    let stride_w = params_buffer.data[7];
    let groups = params_buffer.data[10];
    let in_channels_per_group = params_buffer.data[11];
    let output_h = params_buffer.data[12];
    let output_w = params_buffer.data[13];

    if (ox >= output_w || oy >= output_h || oc >= out_channels) { return; }

    var acc = bias_buffer.data[oc];
    let out_channels_per_group = out_channels / groups;
    let group = oc / out_channels_per_group;
    let ic_start = group * in_channels_per_group;
    let ic_end = ic_start + in_channels_per_group;
    let src_y = oy * stride_h;
    let src_x = ox * stride_w;

    for (var ic: u32 = ic_start; ic < ic_end; ic = ic + 1u) {
        let group_ic = ic - ic_start;
        let input_index = ((ic * input_h + src_y) * input_w) + src_x;
        let weight_index = (oc * in_channels_per_group) + group_ic;
        acc = acc + input_buffer.data[input_index] * weight_buffer.data[weight_index];
    }

    let output_index = ((oc * output_h + oy) * output_w) + ox;
    output_buffer.data[output_index] = acc / (1.0 + exp(-acc));
}
"#;

const CONV2D_3X3_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> weight_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> bias_buffer: BufferF32;
@group(0) @binding(3) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(4) var<storage, read> params_buffer: BufferU32;

fn sample_or_zero(
    channel: u32,
    y: i32,
    x: i32,
    input_h: u32,
    input_w: u32,
) -> f32 {
    if (y < 0 || y >= i32(input_h) || x < 0 || x >= i32(input_w)) {
        return 0.0;
    }
    let index = ((channel * input_h + u32(y)) * input_w) + u32(x);
    return input_buffer.data[index];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;

    let in_channels = params_buffer.data[0];
    let out_channels = params_buffer.data[1];
    let input_h = params_buffer.data[2];
    let input_w = params_buffer.data[3];
    let stride_h = params_buffer.data[6];
    let stride_w = params_buffer.data[7];
    let output_h = params_buffer.data[12];
    let output_w = params_buffer.data[13];

    if (ox >= output_w || oy >= output_h || oc >= out_channels) { return; }

    var acc = bias_buffer.data[oc];
    let base_y = i32(oy * stride_h) - 1;
    let base_x = i32(ox * stride_w) - 1;

    for (var ic: u32 = 0u; ic < in_channels; ic = ic + 1u) {
        let wbase = (oc * in_channels + ic) * 9u;
        acc = acc + sample_or_zero(ic, base_y, base_x, input_h, input_w) * weight_buffer.data[wbase];
        acc = acc + sample_or_zero(ic, base_y, base_x + 1, input_h, input_w) * weight_buffer.data[wbase + 1u];
        acc = acc + sample_or_zero(ic, base_y, base_x + 2, input_h, input_w) * weight_buffer.data[wbase + 2u];
        acc = acc + sample_or_zero(ic, base_y + 1, base_x, input_h, input_w) * weight_buffer.data[wbase + 3u];
        acc = acc + sample_or_zero(ic, base_y + 1, base_x + 1, input_h, input_w) * weight_buffer.data[wbase + 4u];
        acc = acc + sample_or_zero(ic, base_y + 1, base_x + 2, input_h, input_w) * weight_buffer.data[wbase + 5u];
        acc = acc + sample_or_zero(ic, base_y + 2, base_x, input_h, input_w) * weight_buffer.data[wbase + 6u];
        acc = acc + sample_or_zero(ic, base_y + 2, base_x + 1, input_h, input_w) * weight_buffer.data[wbase + 7u];
        acc = acc + sample_or_zero(ic, base_y + 2, base_x + 2, input_h, input_w) * weight_buffer.data[wbase + 8u];
    }

    let output_index = ((oc * output_h + oy) * output_w) + ox;
    output_buffer.data[output_index] = acc;
}
"#;

const CONV2D_3X3_SILU_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> weight_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> bias_buffer: BufferF32;
@group(0) @binding(3) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(4) var<storage, read> params_buffer: BufferU32;

fn sample_or_zero(
    channel: u32,
    y: i32,
    x: i32,
    input_h: u32,
    input_w: u32,
) -> f32 {
    if (y < 0 || y >= i32(input_h) || x < 0 || x >= i32(input_w)) {
        return 0.0;
    }
    let index = ((channel * input_h + u32(y)) * input_w) + u32(x);
    return input_buffer.data[index];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;

    let in_channels = params_buffer.data[0];
    let out_channels = params_buffer.data[1];
    let input_h = params_buffer.data[2];
    let input_w = params_buffer.data[3];
    let stride_h = params_buffer.data[6];
    let stride_w = params_buffer.data[7];
    let output_h = params_buffer.data[12];
    let output_w = params_buffer.data[13];

    if (ox >= output_w || oy >= output_h || oc >= out_channels) { return; }

    var acc = bias_buffer.data[oc];
    let base_y = i32(oy * stride_h) - 1;
    let base_x = i32(ox * stride_w) - 1;

    for (var ic: u32 = 0u; ic < in_channels; ic = ic + 1u) {
        let wbase = (oc * in_channels + ic) * 9u;
        acc = acc + sample_or_zero(ic, base_y, base_x, input_h, input_w) * weight_buffer.data[wbase];
        acc = acc + sample_or_zero(ic, base_y, base_x + 1, input_h, input_w) * weight_buffer.data[wbase + 1u];
        acc = acc + sample_or_zero(ic, base_y, base_x + 2, input_h, input_w) * weight_buffer.data[wbase + 2u];
        acc = acc + sample_or_zero(ic, base_y + 1, base_x, input_h, input_w) * weight_buffer.data[wbase + 3u];
        acc = acc + sample_or_zero(ic, base_y + 1, base_x + 1, input_h, input_w) * weight_buffer.data[wbase + 4u];
        acc = acc + sample_or_zero(ic, base_y + 1, base_x + 2, input_h, input_w) * weight_buffer.data[wbase + 5u];
        acc = acc + sample_or_zero(ic, base_y + 2, base_x, input_h, input_w) * weight_buffer.data[wbase + 6u];
        acc = acc + sample_or_zero(ic, base_y + 2, base_x + 1, input_h, input_w) * weight_buffer.data[wbase + 7u];
        acc = acc + sample_or_zero(ic, base_y + 2, base_x + 2, input_h, input_w) * weight_buffer.data[wbase + 8u];
    }

    let output_index = ((oc * output_h + oy) * output_w) + ox;
    output_buffer.data[output_index] = acc / (1.0 + exp(-acc));
}
"#;

const CONV2D_3X3_FP16_WEIGHTS_WGSL: &str = r#"
enable f16;

struct BufferF32 { data: array<f32>, }
struct BufferF16 { data: array<f16>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> weight_buffer: BufferF16;
@group(0) @binding(2) var<storage, read> bias_buffer: BufferF32;
@group(0) @binding(3) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(4) var<storage, read> params_buffer: BufferU32;

fn dot4_f16(a: vec4<f16>, b: vec4<f16>) -> f16 {
    let prod = a * b;
    return prod.x + prod.y + prod.z + prod.w;
}

fn sample_or_zero(
    channel: u32,
    y: i32,
    x: i32,
    input_h: u32,
    input_w: u32,
) -> f32 {
    if (y < 0 || y >= i32(input_h) || x < 0 || x >= i32(input_w)) {
        return 0.0;
    }
    let index = ((channel * input_h + u32(y)) * input_w) + u32(x);
    return input_buffer.data[index];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;

    let in_channels = params_buffer.data[0];
    let out_channels = params_buffer.data[1];
    let input_h = params_buffer.data[2];
    let input_w = params_buffer.data[3];
    let stride_h = params_buffer.data[6];
    let stride_w = params_buffer.data[7];
    let output_h = params_buffer.data[12];
    let output_w = params_buffer.data[13];

    if (ox >= output_w || oy >= output_h || oc0 >= out_channels) { return; }

    var acc0 = f16(bias_buffer.data[oc0]);
    var acc1 = f16(0.0);
    let has_oc1 = oc1 < out_channels;
    if (has_oc1) {
        acc1 = f16(bias_buffer.data[oc1]);
    }
    let base_y = i32(oy * stride_h) - 1;
    let base_x = i32(ox * stride_w) - 1;

    for (var ic: u32 = 0u; ic < in_channels; ic = ic + 1u) {
        let wbase0 = (oc0 * in_channels + ic) * 9u;
        let wbase1 = (oc1 * in_channels + ic) * 9u;
        let input_vec0 = vec4<f16>(
            f16(sample_or_zero(ic, base_y, base_x, input_h, input_w)),
            f16(sample_or_zero(ic, base_y, base_x + 1, input_h, input_w)),
            f16(sample_or_zero(ic, base_y, base_x + 2, input_h, input_w)),
            f16(sample_or_zero(ic, base_y + 1, base_x, input_h, input_w)),
        );
        let weight0_vec0 = vec4<f16>(
            weight_buffer.data[wbase0],
            weight_buffer.data[wbase0 + 1u],
            weight_buffer.data[wbase0 + 2u],
            weight_buffer.data[wbase0 + 3u],
        );
        let input_vec1 = vec4<f16>(
            f16(sample_or_zero(ic, base_y + 1, base_x + 1, input_h, input_w)),
            f16(sample_or_zero(ic, base_y + 1, base_x + 2, input_h, input_w)),
            f16(sample_or_zero(ic, base_y + 2, base_x, input_h, input_w)),
            f16(sample_or_zero(ic, base_y + 2, base_x + 1, input_h, input_w)),
        );
        let weight0_vec1 = vec4<f16>(
            weight_buffer.data[wbase0 + 4u],
            weight_buffer.data[wbase0 + 5u],
            weight_buffer.data[wbase0 + 6u],
            weight_buffer.data[wbase0 + 7u],
        );
        let input_scalar =
            f16(sample_or_zero(ic, base_y + 2, base_x + 2, input_h, input_w));
        acc0 = acc0 + dot4_f16(input_vec0, weight0_vec0);
        acc0 = acc0 + dot4_f16(input_vec1, weight0_vec1);
        acc0 = acc0
            + input_scalar * weight_buffer.data[wbase0 + 8u];
        if (has_oc1) {
            let weight1_vec0 = vec4<f16>(
                weight_buffer.data[wbase1],
                weight_buffer.data[wbase1 + 1u],
                weight_buffer.data[wbase1 + 2u],
                weight_buffer.data[wbase1 + 3u],
            );
            let weight1_vec1 = vec4<f16>(
                weight_buffer.data[wbase1 + 4u],
                weight_buffer.data[wbase1 + 5u],
                weight_buffer.data[wbase1 + 6u],
                weight_buffer.data[wbase1 + 7u],
            );
            acc1 = acc1 + dot4_f16(input_vec0, weight1_vec0);
            acc1 = acc1 + dot4_f16(input_vec1, weight1_vec1);
            acc1 = acc1
                + input_scalar * weight_buffer.data[wbase1 + 8u];
        }
    }

    let output_index0 = ((oc0 * output_h + oy) * output_w) + ox;
    output_buffer.data[output_index0] = f32(acc0);
    if (has_oc1) {
        let output_index1 = ((oc1 * output_h + oy) * output_w) + ox;
        output_buffer.data[output_index1] = f32(acc1);
    }
}
"#;

const CONV2D_3X3_SILU_FP16_WEIGHTS_WGSL: &str = r#"
enable f16;

struct BufferF32 { data: array<f32>, }
struct BufferF16 { data: array<f16>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> weight_buffer: BufferF16;
@group(0) @binding(2) var<storage, read> bias_buffer: BufferF32;
@group(0) @binding(3) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(4) var<storage, read> params_buffer: BufferU32;

fn dot4_f16(a: vec4<f16>, b: vec4<f16>) -> f16 {
    let prod = a * b;
    return prod.x + prod.y + prod.z + prod.w;
}

fn sample_or_zero(
    channel: u32,
    y: i32,
    x: i32,
    input_h: u32,
    input_w: u32,
) -> f32 {
    if (y < 0 || y >= i32(input_h) || x < 0 || x >= i32(input_w)) {
        return 0.0;
    }
    let index = ((channel * input_h + u32(y)) * input_w) + u32(x);
    return input_buffer.data[index];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc0 = gid.z * 2u;
    let oc1 = oc0 + 1u;

    let in_channels = params_buffer.data[0];
    let out_channels = params_buffer.data[1];
    let input_h = params_buffer.data[2];
    let input_w = params_buffer.data[3];
    let stride_h = params_buffer.data[6];
    let stride_w = params_buffer.data[7];
    let output_h = params_buffer.data[12];
    let output_w = params_buffer.data[13];

    if (ox >= output_w || oy >= output_h || oc0 >= out_channels) { return; }

    var acc0 = f16(bias_buffer.data[oc0]);
    var acc1 = f16(0.0);
    let has_oc1 = oc1 < out_channels;
    if (has_oc1) {
        acc1 = f16(bias_buffer.data[oc1]);
    }
    let base_y = i32(oy * stride_h) - 1;
    let base_x = i32(ox * stride_w) - 1;

    for (var ic: u32 = 0u; ic < in_channels; ic = ic + 1u) {
        let wbase0 = (oc0 * in_channels + ic) * 9u;
        let wbase1 = (oc1 * in_channels + ic) * 9u;
        let input_vec0 = vec4<f16>(
            f16(sample_or_zero(ic, base_y, base_x, input_h, input_w)),
            f16(sample_or_zero(ic, base_y, base_x + 1, input_h, input_w)),
            f16(sample_or_zero(ic, base_y, base_x + 2, input_h, input_w)),
            f16(sample_or_zero(ic, base_y + 1, base_x, input_h, input_w)),
        );
        let weight0_vec0 = vec4<f16>(
            weight_buffer.data[wbase0],
            weight_buffer.data[wbase0 + 1u],
            weight_buffer.data[wbase0 + 2u],
            weight_buffer.data[wbase0 + 3u],
        );
        let input_vec1 = vec4<f16>(
            f16(sample_or_zero(ic, base_y + 1, base_x + 1, input_h, input_w)),
            f16(sample_or_zero(ic, base_y + 1, base_x + 2, input_h, input_w)),
            f16(sample_or_zero(ic, base_y + 2, base_x, input_h, input_w)),
            f16(sample_or_zero(ic, base_y + 2, base_x + 1, input_h, input_w)),
        );
        let weight0_vec1 = vec4<f16>(
            weight_buffer.data[wbase0 + 4u],
            weight_buffer.data[wbase0 + 5u],
            weight_buffer.data[wbase0 + 6u],
            weight_buffer.data[wbase0 + 7u],
        );
        let input_scalar =
            f16(sample_or_zero(ic, base_y + 2, base_x + 2, input_h, input_w));
        acc0 = acc0 + dot4_f16(input_vec0, weight0_vec0);
        acc0 = acc0 + dot4_f16(input_vec1, weight0_vec1);
        acc0 = acc0 + input_scalar * weight_buffer.data[wbase0 + 8u];
        if (has_oc1) {
            let weight1_vec0 = vec4<f16>(
                weight_buffer.data[wbase1],
                weight_buffer.data[wbase1 + 1u],
                weight_buffer.data[wbase1 + 2u],
                weight_buffer.data[wbase1 + 3u],
            );
            let weight1_vec1 = vec4<f16>(
                weight_buffer.data[wbase1 + 4u],
                weight_buffer.data[wbase1 + 5u],
                weight_buffer.data[wbase1 + 6u],
                weight_buffer.data[wbase1 + 7u],
            );
            acc1 = acc1 + dot4_f16(input_vec0, weight1_vec0);
            acc1 = acc1 + dot4_f16(input_vec1, weight1_vec1);
            acc1 = acc1 + input_scalar * weight_buffer.data[wbase1 + 8u];
        }
    }

    let acc0_32 = f32(acc0);
    let output_index0 = ((oc0 * output_h + oy) * output_w) + ox;
    output_buffer.data[output_index0] = acc0_32 / (1.0 + exp(-acc0_32));
    if (has_oc1) {
        let acc1_32 = f32(acc1);
        let output_index1 = ((oc1 * output_h + oy) * output_w) + ox;
        output_buffer.data[output_index1] = acc1_32 / (1.0 + exp(-acc1_32));
    }
}
"#;

const DEPTHWISE_CONV2D_3X3_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> weight_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> bias_buffer: BufferF32;
@group(0) @binding(3) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(4) var<storage, read> params_buffer: BufferU32;

fn sample_or_zero(
    channel: u32,
    y: i32,
    x: i32,
    input_h: u32,
    input_w: u32,
) -> f32 {
    if (y < 0 || y >= i32(input_h) || x < 0 || x >= i32(input_w)) {
        return 0.0;
    }
    let index = ((channel * input_h + u32(y)) * input_w) + u32(x);
    return input_buffer.data[index];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;

    let input_h = params_buffer.data[2];
    let input_w = params_buffer.data[3];
    let stride_h = params_buffer.data[6];
    let stride_w = params_buffer.data[7];
    let output_h = params_buffer.data[12];
    let output_w = params_buffer.data[13];
    let out_channels = params_buffer.data[1];

    if (ox >= output_w || oy >= output_h || oc >= out_channels) { return; }

    var acc = bias_buffer.data[oc];
    let base_y = i32(oy * stride_h) - 1;
    let base_x = i32(ox * stride_w) - 1;
    let wbase = oc * 9u;

    acc = acc + sample_or_zero(oc, base_y, base_x, input_h, input_w) * weight_buffer.data[wbase];
    acc = acc + sample_or_zero(oc, base_y, base_x + 1, input_h, input_w) * weight_buffer.data[wbase + 1u];
    acc = acc + sample_or_zero(oc, base_y, base_x + 2, input_h, input_w) * weight_buffer.data[wbase + 2u];
    acc = acc + sample_or_zero(oc, base_y + 1, base_x, input_h, input_w) * weight_buffer.data[wbase + 3u];
    acc = acc + sample_or_zero(oc, base_y + 1, base_x + 1, input_h, input_w) * weight_buffer.data[wbase + 4u];
    acc = acc + sample_or_zero(oc, base_y + 1, base_x + 2, input_h, input_w) * weight_buffer.data[wbase + 5u];
    acc = acc + sample_or_zero(oc, base_y + 2, base_x, input_h, input_w) * weight_buffer.data[wbase + 6u];
    acc = acc + sample_or_zero(oc, base_y + 2, base_x + 1, input_h, input_w) * weight_buffer.data[wbase + 7u];
    acc = acc + sample_or_zero(oc, base_y + 2, base_x + 2, input_h, input_w) * weight_buffer.data[wbase + 8u];

    let output_index = ((oc * output_h + oy) * output_w) + ox;
    output_buffer.data[output_index] = acc;
}
"#;

const DEPTHWISE_CONV2D_3X3_SILU_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> weight_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> bias_buffer: BufferF32;
@group(0) @binding(3) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(4) var<storage, read> params_buffer: BufferU32;

fn sample_or_zero(
    channel: u32,
    y: i32,
    x: i32,
    input_h: u32,
    input_w: u32,
) -> f32 {
    if (y < 0 || y >= i32(input_h) || x < 0 || x >= i32(input_w)) {
        return 0.0;
    }
    let index = ((channel * input_h + u32(y)) * input_w) + u32(x);
    return input_buffer.data[index];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;

    let input_h = params_buffer.data[2];
    let input_w = params_buffer.data[3];
    let stride_h = params_buffer.data[6];
    let stride_w = params_buffer.data[7];
    let output_h = params_buffer.data[12];
    let output_w = params_buffer.data[13];
    let out_channels = params_buffer.data[1];

    if (ox >= output_w || oy >= output_h || oc >= out_channels) { return; }

    var acc = bias_buffer.data[oc];
    let base_y = i32(oy * stride_h) - 1;
    let base_x = i32(ox * stride_w) - 1;
    let wbase = oc * 9u;

    acc = acc + sample_or_zero(oc, base_y, base_x, input_h, input_w) * weight_buffer.data[wbase];
    acc = acc + sample_or_zero(oc, base_y, base_x + 1, input_h, input_w) * weight_buffer.data[wbase + 1u];
    acc = acc + sample_or_zero(oc, base_y, base_x + 2, input_h, input_w) * weight_buffer.data[wbase + 2u];
    acc = acc + sample_or_zero(oc, base_y + 1, base_x, input_h, input_w) * weight_buffer.data[wbase + 3u];
    acc = acc + sample_or_zero(oc, base_y + 1, base_x + 1, input_h, input_w) * weight_buffer.data[wbase + 4u];
    acc = acc + sample_or_zero(oc, base_y + 1, base_x + 2, input_h, input_w) * weight_buffer.data[wbase + 5u];
    acc = acc + sample_or_zero(oc, base_y + 2, base_x, input_h, input_w) * weight_buffer.data[wbase + 6u];
    acc = acc + sample_or_zero(oc, base_y + 2, base_x + 1, input_h, input_w) * weight_buffer.data[wbase + 7u];
    acc = acc + sample_or_zero(oc, base_y + 2, base_x + 2, input_h, input_w) * weight_buffer.data[wbase + 8u];

    let output_index = ((oc * output_h + oy) * output_w) + ox;
    output_buffer.data[output_index] = acc / (1.0 + exp(-acc));
}
"#;

const SILU_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let len = params_buffer.data[0];
    if (index >= len) { return; }

    let x = input_buffer.data[index];
    output_buffer.data[index] = x / (1.0 + exp(-x));
}
"#;

const UPSAMPLE_NEAREST_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let channel = gid.z;

    let channels = params_buffer.data[0];
    let input_h = params_buffer.data[1];
    let input_w = params_buffer.data[2];
    let scale = params_buffer.data[3];
    let output_h = params_buffer.data[4];
    let output_w = params_buffer.data[5];

    if (ox >= output_w || oy >= output_h || channel >= channels) { return; }

    let iy = oy / scale;
    let ix = ox / scale;

    let src = ((channel * input_h + iy) * input_w) + ix;
    let dst = ((channel * output_h + oy) * output_w) + ox;
    output_buffer.data[dst] = input_buffer.data[src];
}
"#;

const CONCAT_CHANNELS_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> lhs_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> rhs_buffer: BufferF32;
@group(0) @binding(2) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(3) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let channel = gid.z;

    let lhs_channels = params_buffer.data[0];
    let rhs_channels = params_buffer.data[1];
    let height = params_buffer.data[2];
    let width = params_buffer.data[3];
    let out_channels = lhs_channels + rhs_channels;

    if (ox >= width || oy >= height || channel >= out_channels) { return; }

    let plane_index = oy * width + ox;
    let dst = (channel * height * width) + plane_index;

    if (channel < lhs_channels) {
        let src = (channel * height * width) + plane_index;
        output_buffer.data[dst] = lhs_buffer.data[src];
    } else {
        let rhs_channel = channel - lhs_channels;
        let src = (rhs_channel * height * width) + plane_index;
        output_buffer.data[dst] = rhs_buffer.data[src];
    }
}
"#;

const CONCAT3_CHANNELS_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> a_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> b_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> c_buffer: BufferF32;
@group(0) @binding(3) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(4) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let channel = gid.z;

    let a_channels = params_buffer.data[0];
    let b_channels = params_buffer.data[1];
    let c_channels = params_buffer.data[2];
    let height = params_buffer.data[3];
    let width = params_buffer.data[4];
    let ab_channels = a_channels + b_channels;
    let out_channels = ab_channels + c_channels;

    if (ox >= width || oy >= height || channel >= out_channels) { return; }

    let plane_index = oy * width + ox;
    let dst = (channel * height * width) + plane_index;

    if (channel < a_channels) {
        let src = (channel * height * width) + plane_index;
        output_buffer.data[dst] = a_buffer.data[src];
    } else if (channel < ab_channels) {
        let b_channel = channel - a_channels;
        let src = (b_channel * height * width) + plane_index;
        output_buffer.data[dst] = b_buffer.data[src];
    } else {
        let c_channel = channel - ab_channels;
        let src = (c_channel * height * width) + plane_index;
        output_buffer.data[dst] = c_buffer.data[src];
    }
}
"#;

const SIGMOID_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let len = params_buffer.data[0];
    if (index >= len) { return; }

    let x = input_buffer.data[index];
    output_buffer.data[index] = 1.0 / (1.0 + exp(-x));
}
"#;

const ADD_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> lhs_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> rhs_buffer: BufferF32;
@group(0) @binding(2) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(3) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let len = params_buffer.data[0];
    if (index >= len) { return; }
    output_buffer.data[index] = lhs_buffer.data[index] + rhs_buffer.data[index];
}
"#;

const DECODE_HEAD_SCALE_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> params_buffer: BufferU32;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn clamp_exp_input(x: f32) -> f32 {
    return clamp(x, -16.0, 16.0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;

    let channels = params_buffer.data[0];
    let input_h = params_buffer.data[1];
    let input_w = params_buffer.data[2];
    let stride = params_buffer.data[3];
    let num_classes = params_buffer.data[4];
    let rows = params_buffer.data[5];
    let cols = params_buffer.data[6];
    let row_offset = params_buffer.data[7];
    let output_rows = params_buffer.data[8];

    if (row >= rows || col >= cols) { return; }
    if (channels != cols) { return; }
    if (row_offset + row >= output_rows) { return; }

    let y = row / input_w;
    let x = row % input_w;

    var value: f32;
    if (col == 0u) {
        let idx = (y * input_w) + x;
        value = (input_buffer.data[idx] + f32(x)) * f32(stride);
    } else if (col == 1u) {
        let idx = (input_h * input_w) + (y * input_w) + x;
        value = (input_buffer.data[idx] + f32(y)) * f32(stride);
    } else if (col == 2u) {
        let idx = (2u * input_h * input_w) + (y * input_w) + x;
        value = exp(clamp_exp_input(input_buffer.data[idx])) * f32(stride);
    } else if (col == 3u) {
        let idx = (3u * input_h * input_w) + (y * input_w) + x;
        value = exp(clamp_exp_input(input_buffer.data[idx])) * f32(stride);
    } else if (col == 4u) {
        let idx = (4u * input_h * input_w) + (y * input_w) + x;
        value = sigmoid(input_buffer.data[idx]);
    } else {
        let cls = col - 5u;
        if (cls >= num_classes) { return; }
        let idx = ((5u + cls) * input_h * input_w) + (y * input_w) + x;
        value = sigmoid(input_buffer.data[idx]);
    }

    let out_idx = (row_offset + row) * cols + col;
    output_buffer.data[out_idx] = value;
}
"#;

#[allow(dead_code)]
const CONCAT_ROWS_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> lhs_buffer: BufferF32;
@group(0) @binding(1) var<storage, read> rhs_buffer: BufferF32;
@group(0) @binding(2) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(3) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;

    let lhs_rows = params_buffer.data[0];
    let rhs_rows = params_buffer.data[1];
    let cols = params_buffer.data[2];
    let total_rows = lhs_rows + rhs_rows;

    if (row >= total_rows || col >= cols) { return; }

    let out_idx = row * cols + col;
    if (row < lhs_rows) {
        let src_idx = row * cols + col;
        output_buffer.data[out_idx] = lhs_buffer.data[src_idx];
    } else {
        let rhs_row = row - lhs_rows;
        let src_idx = rhs_row * cols + col;
        output_buffer.data[out_idx] = rhs_buffer.data[src_idx];
    }
}
"#;

const MAXPOOL2D_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let channel = gid.z;

    let channels = params_buffer.data[0];
    let input_h = params_buffer.data[1];
    let input_w = params_buffer.data[2];
    let kernel = params_buffer.data[3];
    let stride = params_buffer.data[4];
    let padding = params_buffer.data[5];
    let output_h = params_buffer.data[6];
    let output_w = params_buffer.data[7];

    if (ox >= output_w || oy >= output_h || channel >= channels) { return; }

    var max_value = -3.4028235e38;
    for (var ky: u32 = 0u; ky < kernel; ky = ky + 1u) {
        let src_y = i32(oy * stride + ky) - i32(padding);
        if (src_y < 0 || src_y >= i32(input_h)) { continue; }
        for (var kx: u32 = 0u; kx < kernel; kx = kx + 1u) {
            let src_x = i32(ox * stride + kx) - i32(padding);
            if (src_x < 0 || src_x >= i32(input_w)) { continue; }
            let src = ((channel * input_h + u32(src_y)) * input_w) + u32(src_x);
            max_value = max(max_value, input_buffer.data[src]);
        }
    }

    let dst = ((channel * output_h + oy) * output_w) + ox;
    output_buffer.data[dst] = max_value;
}
"#;

const SPP_CONCAT_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let out_channel = gid.z;

    let channels = params_buffer.data[0];
    let input_h = params_buffer.data[1];
    let input_w = params_buffer.data[2];
    let kernel0 = params_buffer.data[3];
    let kernel1 = params_buffer.data[4];
    let kernel2 = params_buffer.data[5];
    let out_channels = channels * 4u;

    if (ox >= input_w || oy >= input_h || out_channel >= out_channels) { return; }

    let src_channel = out_channel % channels;
    let plane_index = oy * input_w + ox;
    let dst = (out_channel * input_h * input_w) + plane_index;

    if (out_channel < channels) {
        let src = (src_channel * input_h * input_w) + plane_index;
        output_buffer.data[dst] = input_buffer.data[src];
        return;
    }

    var kernel = kernel0;
    if (out_channel >= channels * 3u) {
        kernel = kernel2;
    } else if (out_channel >= channels * 2u) {
        kernel = kernel1;
    }

    let padding = kernel / 2u;
    var max_value = -3.4028235e38;
    for (var ky: u32 = 0u; ky < kernel; ky = ky + 1u) {
        let src_y = i32(oy + ky) - i32(padding);
        if (src_y < 0 || src_y >= i32(input_h)) { continue; }
        for (var kx: u32 = 0u; kx < kernel; kx = kx + 1u) {
            let src_x = i32(ox + kx) - i32(padding);
            if (src_x < 0 || src_x >= i32(input_w)) { continue; }
            let src = ((src_channel * input_h + u32(src_y)) * input_w) + u32(src_x);
            max_value = max(max_value, input_buffer.data[src]);
        }
    }

    output_buffer.data[dst] = max_value;
}
"#;

const FOCUS_WGSL: &str = r#"
struct BufferF32 { data: array<f32>, }
struct BufferU32 { data: array<u32>, }

@group(0) @binding(0) var<storage, read> input_buffer: BufferF32;
@group(0) @binding(1) var<storage, read_write> output_buffer: BufferF32;
@group(0) @binding(2) var<storage, read> params_buffer: BufferU32;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let out_channel = gid.z;

    let input_channels = params_buffer.data[0];
    let input_h = params_buffer.data[1];
    let input_w = params_buffer.data[2];
    let output_h = params_buffer.data[3];
    let output_w = params_buffer.data[4];
    let output_channels = input_channels * 4u;

    if (ox >= output_w || oy >= output_h || out_channel >= output_channels) { return; }

    let patch_index = out_channel / input_channels;
    let in_channel = out_channel % input_channels;
    var src_y = oy * 2u;
    var src_x = ox * 2u;

    if (patch_index == 1u) {
        src_y = src_y + 1u;
    } else if (patch_index == 2u) {
        src_x = src_x + 1u;
    } else if (patch_index == 3u) {
        src_y = src_y + 1u;
        src_x = src_x + 1u;
    }

    let src = ((in_channel * input_h + src_y) * input_w) + src_x;
    let dst = ((out_channel * output_h + oy) * output_w) + ox;
    output_buffer.data[dst] = input_buffer.data[src];
}
"#;

pub fn run_conv2d_demo(
    input: &[f32],
    input_h: usize,
    input_w: usize,
    weights: &FusedConv2dWeights,
    device_index: usize,
) -> Result<Vec<f32>> {
    let runtime = VulkanRuntime::new(device_index)?;
    let input_shape = TensorShape::new(weights.spec.in_channels, input_h, input_w);
    let input = runtime.upload_tensor(input, input_shape)?;
    let output = runtime.run_conv2d_tensor(&input, weights)?;
    runtime.read_tensor(&output)
}

pub fn run_demo_block(
    input: &[f32],
    input_shape: TensorShape,
    weights: &FusedConv2dWeights,
    skip: &[f32],
    skip_shape: TensorShape,
    upsample_scale: usize,
    device_index: usize,
) -> Result<(TensorShape, Vec<f32>)> {
    let runtime = VulkanRuntime::new(device_index)?;
    let input = runtime.upload_tensor(input, input_shape)?;
    let skip = runtime.upload_tensor(skip, skip_shape)?;

    let conv = runtime.run_conv2d_tensor(&input, weights)?;
    let silu = runtime.run_silu_tensor(&conv)?;
    let upsampled = runtime.run_upsample_nearest_tensor(&silu, upsample_scale)?;
    let concat = runtime.run_concat_channels_tensor(&upsampled, &skip)?;
    let output = runtime.read_tensor(&concat)?;

    Ok((concat.shape, output))
}

pub fn run_demo_stem(
    input: &[f32],
    input_shape: TensorShape,
    weights: &FusedConv2dWeights,
    device_index: usize,
) -> Result<(TensorShape, Vec<f32>)> {
    let runtime = VulkanRuntime::new(device_index)?;
    let input = runtime.upload_tensor(input, input_shape)?;

    let focus = runtime.run_focus_tensor(&input)?;
    let conv = runtime.run_conv2d_tensor(&focus, weights)?;
    let silu = runtime.run_silu_tensor(&conv)?;
    let pooled = runtime.run_maxpool2d_tensor(&silu, 3, 1, 1)?;
    let added = runtime.run_add_tensors(&pooled, &silu)?;
    let sigmoid = runtime.run_sigmoid_tensor(&added)?;
    let output = runtime.read_tensor(&sigmoid)?;

    Ok((sigmoid.shape, output))
}

pub fn run_demo_bottleneck(
    input: &[f32],
    input_shape: TensorShape,
    block: &BottleneckBlock,
    device_index: usize,
) -> Result<(TensorShape, Vec<f32>)> {
    let runtime = VulkanRuntime::new(device_index)?;
    let input = runtime.upload_tensor(input, input_shape)?;
    let identity = input.clone();

    let conv1 = runtime.run_baseconv_block(&input, &block.conv1)?;
    let mut conv2 = match &block.conv2 {
        ConvBlock::Base(base) => runtime.run_baseconv_block(&conv1, base)?,
        ConvBlock::Dws(dws) => runtime.run_dwsconv_block(&conv1, dws)?,
    };

    if block.shortcut {
        conv2 = runtime.run_add_tensors(&conv2, &identity)?;
    }

    let output = runtime.read_tensor(&conv2)?;
    Ok((conv2.shape, output))
}

pub fn run_demo_csp(
    input: &[f32],
    input_shape: TensorShape,
    block: &CspStageBlock,
    device_index: usize,
) -> Result<(TensorShape, Vec<f32>)> {
    let runtime = VulkanRuntime::new(device_index)?;
    let input = runtime.upload_tensor(input, input_shape)?;
    let output = runtime.run_csp_stage_block(&input, block)?;
    let output_data = runtime.read_tensor(&output)?;

    Ok((output.shape, output_data))
}

pub fn run_demo_dark5(
    input: &[f32],
    input_shape: TensorShape,
    block: &Dark5Block,
    device_index: usize,
) -> Result<(TensorShape, Vec<f32>)> {
    let runtime = VulkanRuntime::new(device_index)?;
    let input = runtime.upload_tensor(input, input_shape)?;
    let output = runtime.run_dark5_block(&input, block)?;
    let output_data = runtime.read_tensor(&output)?;

    Ok((output.shape, output_data))
}

pub fn run_demo_backbone(
    input: &[f32],
    input_shape: TensorShape,
    backbone: &CspDarknetDemo,
    device_index: usize,
) -> Result<DarknetFeatures> {
    let runtime = VulkanRuntime::new(device_index)?;
    let input = runtime.upload_tensor(input, input_shape)?;
    let outputs = runtime.run_backbone(&input, backbone)?;

    Ok(DarknetFeatures {
        f1_shape: outputs.f1.shape,
        f1: runtime.read_tensor(&outputs.f1)?,
        f2_shape: outputs.f2.shape,
        f2: runtime.read_tensor(&outputs.f2)?,
        f3_shape: outputs.f3.shape,
        f3: runtime.read_tensor(&outputs.f3)?,
    })
}

pub fn run_demo_pafpn(
    input: &[f32],
    input_shape: TensorShape,
    backbone: &CspDarknetDemo,
    pafpn: &YoloxPafpnDemo,
    device_index: usize,
) -> Result<PafpnFeatures> {
    let runtime = VulkanRuntime::new(device_index)?;
    let input = runtime.upload_tensor(input, input_shape)?;
    let backbone_outputs = runtime.run_backbone(&input, backbone)?;
    let outputs = runtime.run_pafpn(&backbone_outputs, pafpn)?;

    Ok(PafpnFeatures {
        p3_shape: outputs.p3.shape,
        p3: runtime.read_tensor(&outputs.p3)?,
        p4_shape: outputs.p4.shape,
        p4: runtime.read_tensor(&outputs.p4)?,
        p5_shape: outputs.p5.shape,
        p5: runtime.read_tensor(&outputs.p5)?,
    })
}

pub fn run_demo_head(
    input: &[f32],
    input_shape: TensorShape,
    backbone: &CspDarknetDemo,
    pafpn: &YoloxPafpnDemo,
    head: &YoloxHeadDemo,
    device_index: usize,
) -> Result<HeadFeatures> {
    let runtime = VulkanRuntime::new(device_index)?;
    let input = runtime.upload_tensor(input, input_shape)?;
    let backbone_outputs = runtime.run_backbone(&input, backbone)?;
    let pafpn_outputs = runtime.run_pafpn(&backbone_outputs, pafpn)?;
    let outputs = runtime.run_head(&pafpn_outputs, head)?;

    Ok(HeadFeatures {
        s8_shape: outputs.s8.shape,
        s8: runtime.read_tensor(&outputs.s8)?,
        s16_shape: outputs.s16.shape,
        s16: runtime.read_tensor(&outputs.s16)?,
        s32_shape: outputs.s32.shape,
        s32: runtime.read_tensor(&outputs.s32)?,
    })
}

pub fn run_demo_decode(
    input: &[f32],
    input_shape: TensorShape,
    backbone: &CspDarknetDemo,
    pafpn: &YoloxPafpnDemo,
    head: &YoloxHeadDemo,
    decode: &YoloxDecodeDemo,
    device_index: usize,
) -> Result<DecodedPredictions> {
    GpuDecodeSession::new(device_index)?.run_decode(
        input,
        input_shape,
        backbone,
        pafpn,
        head,
        decode,
    )
}

pub fn run_demo_decode_resident(
    input: &[f32],
    input_shape: TensorShape,
    backbone: &CspDarknetDemo,
    pafpn: &YoloxPafpnDemo,
    head: &YoloxHeadDemo,
    decode: &YoloxDecodeDemo,
    fp16: bool,
    device_index: usize,
) -> Result<DecodedPredictions> {
    GpuResidentDecodeSession::new(backbone, pafpn, head, fp16, device_index)?.run_decode(
        input,
        input_shape,
        decode,
    )
}

#[derive(Clone)]
struct GpuTensor {
    shape: TensorShape,
    buffer: Subbuffer<[f32]>,
}

struct GpuFusedConv2dWeights {
    spec: Conv2dSpec,
    weights_f32: Option<Subbuffer<[f32]>>,
    weights_f16: Option<Subbuffer<[u16]>>,
    bias: Subbuffer<[f32]>,
    use_fp16: bool,
}

impl GpuFusedConv2dWeights {
    fn weights_f32(&self) -> Result<Subbuffer<[f32]>> {
        self.weights_f32
            .clone()
            .ok_or_else(|| anyhow!("pesos f32 não disponíveis para este kernel"))
    }

    fn weights_f16(&self) -> Result<Subbuffer<[u16]>> {
        self.weights_f16
            .clone()
            .ok_or_else(|| anyhow!("pesos fp16 não disponíveis para este kernel"))
    }
}

struct GpuDarknetFeatures {
    f1: GpuTensor,
    f2: GpuTensor,
    f3: GpuTensor,
}

struct GpuPafpnFeatures {
    p3: GpuTensor,
    p4: GpuTensor,
    p5: GpuTensor,
}

struct GpuHeadFeatures {
    s8: GpuTensor,
    s16: GpuTensor,
    s32: GpuTensor,
}

struct GpuMatrix {
    rows: usize,
    cols: usize,
    buffer: Subbuffer<[f32]>,
}

impl GpuDecodeSession {
    pub fn new(device_index: usize) -> Result<Self> {
        Ok(Self {
            runtime: VulkanRuntime::new(device_index)?,
        })
    }

    pub fn run_decode(
        &self,
        input: &[f32],
        input_shape: TensorShape,
        backbone: &CspDarknetDemo,
        pafpn: &YoloxPafpnDemo,
        head: &YoloxHeadDemo,
        decode: &YoloxDecodeDemo,
    ) -> Result<DecodedPredictions> {
        let (decoded, _) =
            self.run_decode_profiled(input, input_shape, backbone, pafpn, head, decode)?;
        Ok(decoded)
    }

    pub fn run_decode_profiled(
        &self,
        input: &[f32],
        input_shape: TensorShape,
        backbone: &CspDarknetDemo,
        pafpn: &YoloxPafpnDemo,
        head: &YoloxHeadDemo,
        decode: &YoloxDecodeDemo,
    ) -> Result<(DecodedPredictions, GpuDecodeStageDurations)> {
        let started = std::time::Instant::now();
        let input = self.runtime.upload_tensor(input, input_shape)?;
        let upload = started.elapsed();

        let started = std::time::Instant::now();
        let backbone_outputs = self.runtime.run_backbone(&input, backbone)?;
        let backbone_time = started.elapsed();
        self.runtime.recycle_tensor(input)?;

        let started = std::time::Instant::now();
        let pafpn_outputs = self.runtime.run_pafpn(&backbone_outputs, pafpn)?;
        let pafpn_time = started.elapsed();
        self.runtime.recycle_tensor(backbone_outputs.f1)?;
        self.runtime.recycle_tensor(backbone_outputs.f2)?;
        self.runtime.recycle_tensor(backbone_outputs.f3)?;

        let started = std::time::Instant::now();
        let head_outputs = self.runtime.run_head(&pafpn_outputs, head)?;
        let head_time = started.elapsed();
        self.runtime.recycle_tensor(pafpn_outputs.p3)?;
        self.runtime.recycle_tensor(pafpn_outputs.p4)?;
        self.runtime.recycle_tensor(pafpn_outputs.p5)?;

        let started = std::time::Instant::now();
        let decoded = self.runtime.run_decode(&head_outputs, decode)?;
        let decode_time = started.elapsed();
        self.runtime.recycle_tensor(head_outputs.s8)?;
        self.runtime.recycle_tensor(head_outputs.s16)?;
        self.runtime.recycle_tensor(head_outputs.s32)?;

        let rows = decoded.rows;
        let cols = decoded.cols;
        let started = std::time::Instant::now();
        let data = self.runtime.read_matrix(&decoded)?;
        let readback = started.elapsed();
        self.runtime.recycle_matrix(decoded)?;

        Ok((
            DecodedPredictions { rows, cols, data },
            GpuDecodeStageDurations {
                upload,
                backbone: backbone_time,
                pafpn: pafpn_time,
                head: head_time,
                decode: decode_time,
                readback,
            },
        ))
    }

    pub fn run_decode_profiled_sync(
        &self,
        input: &[f32],
        input_shape: TensorShape,
        backbone: &CspDarknetDemo,
        pafpn: &YoloxPafpnDemo,
        head: &YoloxHeadDemo,
        decode: &YoloxDecodeDemo,
    ) -> Result<(DecodedPredictions, GpuDecodeStageDurations)> {
        let started = std::time::Instant::now();
        let input = self.runtime.upload_tensor(input, input_shape)?;
        self.runtime.finish_pending_work()?;
        let upload = started.elapsed();

        let started = std::time::Instant::now();
        let backbone_outputs = self.runtime.run_backbone(&input, backbone)?;
        self.runtime.finish_pending_work()?;
        let backbone_time = started.elapsed();
        self.runtime.recycle_tensor(input)?;

        let started = std::time::Instant::now();
        let pafpn_outputs = self.runtime.run_pafpn(&backbone_outputs, pafpn)?;
        self.runtime.finish_pending_work()?;
        let pafpn_time = started.elapsed();
        self.runtime.recycle_tensor(backbone_outputs.f1)?;
        self.runtime.recycle_tensor(backbone_outputs.f2)?;
        self.runtime.recycle_tensor(backbone_outputs.f3)?;

        let started = std::time::Instant::now();
        let head_outputs = self.runtime.run_head(&pafpn_outputs, head)?;
        self.runtime.finish_pending_work()?;
        let head_time = started.elapsed();
        self.runtime.recycle_tensor(pafpn_outputs.p3)?;
        self.runtime.recycle_tensor(pafpn_outputs.p4)?;
        self.runtime.recycle_tensor(pafpn_outputs.p5)?;

        let started = std::time::Instant::now();
        let decoded = self.runtime.run_decode(&head_outputs, decode)?;
        self.runtime.finish_pending_work()?;
        let decode_time = started.elapsed();
        self.runtime.recycle_tensor(head_outputs.s8)?;
        self.runtime.recycle_tensor(head_outputs.s16)?;
        self.runtime.recycle_tensor(head_outputs.s32)?;

        let rows = decoded.rows;
        let cols = decoded.cols;
        let started = std::time::Instant::now();
        let data = self.runtime.read_matrix(&decoded)?;
        let readback = started.elapsed();
        self.runtime.recycle_matrix(decoded)?;

        Ok((
            DecodedPredictions { rows, cols, data },
            GpuDecodeStageDurations {
                upload,
                backbone: backbone_time,
                pafpn: pafpn_time,
                head: head_time,
                decode: decode_time,
                readback,
            },
        ))
    }
}

impl GpuResidentDecodeSession {
    pub fn new(
        backbone: &CspDarknetDemo,
        pafpn: &YoloxPafpnDemo,
        head: &YoloxHeadDemo,
        fp16: bool,
        device_index: usize,
    ) -> Result<Self> {
        let runtime = VulkanRuntime::new_with_options(device_index, fp16)?;
        Ok(Self {
            backbone: runtime.prepare_backbone(backbone)?,
            pafpn: runtime.prepare_pafpn(pafpn)?,
            head: runtime.prepare_head(head)?,
            runtime,
        })
    }

    pub fn run_decode(
        &self,
        input: &[f32],
        input_shape: TensorShape,
        decode: &YoloxDecodeDemo,
    ) -> Result<DecodedPredictions> {
        let (decoded, _) = self.run_decode_profiled(input, input_shape, decode)?;
        Ok(decoded)
    }

    pub fn run_decode_profiled(
        &self,
        input: &[f32],
        input_shape: TensorShape,
        decode: &YoloxDecodeDemo,
    ) -> Result<(DecodedPredictions, GpuDecodeStageDurations)> {
        let started = std::time::Instant::now();
        let input = self.runtime.upload_tensor(input, input_shape)?;
        let upload = started.elapsed();

        let started = std::time::Instant::now();
        let backbone_outputs = self.runtime.run_backbone_prepared(&input, &self.backbone)?;
        let backbone_time = started.elapsed();
        self.runtime.recycle_tensor(input)?;

        let started = std::time::Instant::now();
        let pafpn_outputs = self
            .runtime
            .run_pafpn_prepared(&backbone_outputs, &self.pafpn)?;
        let pafpn_time = started.elapsed();
        self.runtime.recycle_tensor(backbone_outputs.f1)?;
        self.runtime.recycle_tensor(backbone_outputs.f2)?;
        self.runtime.recycle_tensor(backbone_outputs.f3)?;

        let started = std::time::Instant::now();
        let head_outputs = self.runtime.run_head_prepared(&pafpn_outputs, &self.head)?;
        let head_time = started.elapsed();
        self.runtime.recycle_tensor(pafpn_outputs.p3)?;
        self.runtime.recycle_tensor(pafpn_outputs.p4)?;
        self.runtime.recycle_tensor(pafpn_outputs.p5)?;

        let started = std::time::Instant::now();
        let decoded = self.runtime.run_decode(&head_outputs, decode)?;
        let decode_time = started.elapsed();
        self.runtime.recycle_tensor(head_outputs.s8)?;
        self.runtime.recycle_tensor(head_outputs.s16)?;
        self.runtime.recycle_tensor(head_outputs.s32)?;

        let rows = decoded.rows;
        let cols = decoded.cols;
        let started = std::time::Instant::now();
        let data = self.runtime.read_matrix(&decoded)?;
        let readback = started.elapsed();
        self.runtime.recycle_matrix(decoded)?;

        Ok((
            DecodedPredictions { rows, cols, data },
            GpuDecodeStageDurations {
                upload,
                backbone: backbone_time,
                pafpn: pafpn_time,
                head: head_time,
                decode: decode_time,
                readback,
            },
        ))
    }

    pub fn run_decode_profiled_sync(
        &self,
        input: &[f32],
        input_shape: TensorShape,
        decode: &YoloxDecodeDemo,
    ) -> Result<(DecodedPredictions, GpuDecodeStageDurations)> {
        let started = std::time::Instant::now();
        let input = self.runtime.upload_tensor(input, input_shape)?;
        self.runtime.finish_pending_work()?;
        let upload = started.elapsed();

        let started = std::time::Instant::now();
        let backbone_outputs = self.runtime.run_backbone_prepared(&input, &self.backbone)?;
        self.runtime.finish_pending_work()?;
        let backbone_time = started.elapsed();
        self.runtime.recycle_tensor(input)?;

        let started = std::time::Instant::now();
        let pafpn_outputs = self
            .runtime
            .run_pafpn_prepared(&backbone_outputs, &self.pafpn)?;
        self.runtime.finish_pending_work()?;
        let pafpn_time = started.elapsed();
        self.runtime.recycle_tensor(backbone_outputs.f1)?;
        self.runtime.recycle_tensor(backbone_outputs.f2)?;
        self.runtime.recycle_tensor(backbone_outputs.f3)?;

        let started = std::time::Instant::now();
        let head_outputs = self.runtime.run_head_prepared(&pafpn_outputs, &self.head)?;
        self.runtime.finish_pending_work()?;
        let head_time = started.elapsed();
        self.runtime.recycle_tensor(pafpn_outputs.p3)?;
        self.runtime.recycle_tensor(pafpn_outputs.p4)?;
        self.runtime.recycle_tensor(pafpn_outputs.p5)?;

        let started = std::time::Instant::now();
        let decoded = self.runtime.run_decode(&head_outputs, decode)?;
        self.runtime.finish_pending_work()?;
        let decode_time = started.elapsed();
        self.runtime.recycle_tensor(head_outputs.s8)?;
        self.runtime.recycle_tensor(head_outputs.s16)?;
        self.runtime.recycle_tensor(head_outputs.s32)?;

        let rows = decoded.rows;
        let cols = decoded.cols;
        let started = std::time::Instant::now();
        let data = self.runtime.read_matrix(&decoded)?;
        let readback = started.elapsed();
        self.runtime.recycle_matrix(decoded)?;

        Ok((
            DecodedPredictions { rows, cols, data },
            GpuDecodeStageDurations {
                upload,
                backbone: backbone_time,
                pafpn: pafpn_time,
                head: head_time,
                decode: decode_time,
                readback,
            },
        ))
    }
}

pub fn query_vulkan_device_info(device_index: usize) -> Result<VulkanDeviceInfo> {
    let library = load_vulkan_library()?;
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            max_api_version: Some(Version::V1_0),
            enabled_extensions: InstanceExtensions {
                khr_get_physical_device_properties2: true,
                ..InstanceExtensions::empty()
            },
            ..Default::default()
        },
    )
    .context("falha ao criar instância Vulkan para coletar metadados")?;

    let physical_devices = instance
        .enumerate_physical_devices()
        .context("falha ao enumerar GPUs Vulkan para coletar metadados")?
        .collect::<Vec<_>>();
    let physical_device = physical_devices
        .get(device_index)
        .cloned()
        .ok_or_else(|| anyhow!("device_index {} inválido", device_index))?;
    let properties = physical_device.properties();

    Ok(VulkanDeviceInfo {
        device_name: properties.device_name.clone(),
        device_type: format!("{:?}", properties.device_type),
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        api_version: properties.api_version.to_string(),
        driver_version: properties.driver_version,
        driver_name: properties.driver_name.clone(),
        driver_info: properties.driver_info.clone(),
        driver_id: properties.driver_id.map(|item| format!("{item:?}")),
    })
}

pub fn query_vulkan_fp16_support(device_index: usize) -> Result<VulkanFp16SupportInfo> {
    let library = load_vulkan_library()?;
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            max_api_version: Some(Version::V1_0),
            enabled_extensions: InstanceExtensions {
                khr_get_physical_device_properties2: true,
                ..InstanceExtensions::empty()
            },
            ..Default::default()
        },
    )
    .context("falha ao criar instância Vulkan para coletar suporte fp16")?;

    let physical_devices = instance
        .enumerate_physical_devices()
        .context("falha ao enumerar GPUs Vulkan para coletar suporte fp16")?
        .collect::<Vec<_>>();
    let physical_device = physical_devices
        .get(device_index)
        .cloned()
        .ok_or_else(|| anyhow!("device_index {} inválido", device_index))?;
    let properties = physical_device.properties();
    let features = physical_device.supported_features();
    let extensions = physical_device.supported_extensions();

    Ok(VulkanFp16SupportInfo {
        device_name: properties.device_name.clone(),
        api_version: properties.api_version.to_string(),
        driver_name: properties.driver_name.clone(),
        driver_info: properties.driver_info.clone(),
        extension_khr_shader_float16_int8: extensions.khr_shader_float16_int8,
        shader_float16: features.shader_float16,
        storage_buffer16_bit_access: features.storage_buffer16_bit_access,
        uniform_and_storage_buffer16_bit_access: features.uniform_and_storage_buffer16_bit_access,
        storage_input_output16: features.storage_input_output16,
        workgroup_memory_explicit_layout16_bit_access: features
            .workgroup_memory_explicit_layout16_bit_access,
        shader_denorm_flush_to_zero_float16: properties.shader_denorm_flush_to_zero_float16,
        shader_denorm_preserve_float16: properties.shader_denorm_preserve_float16,
        shader_rounding_mode_rte_float16: properties.shader_rounding_mode_rte_float16,
        shader_rounding_mode_rtz_float16: properties.shader_rounding_mode_rtz_float16,
    })
}

struct PreparedBaseConvBlock {
    conv: GpuFusedConv2dWeights,
}

struct PreparedDwsConvBlock {
    depthwise: GpuFusedConv2dWeights,
    pointwise: GpuFusedConv2dWeights,
}

enum PreparedConvBlock {
    Base(PreparedBaseConvBlock),
    Dws(PreparedDwsConvBlock),
}

struct PreparedBottleneckBlock {
    conv1: PreparedBaseConvBlock,
    conv2: PreparedConvBlock,
    shortcut: bool,
}

struct PreparedCspBottleneckBlock {
    conv1: PreparedBaseConvBlock,
    conv2: PreparedBaseConvBlock,
    conv3: PreparedBaseConvBlock,
    blocks: Vec<PreparedBottleneckBlock>,
}

struct PreparedCspStageBlock {
    conv: PreparedConvBlock,
    c3: PreparedCspBottleneckBlock,
}

struct PreparedSppBottleneckBlock {
    conv1: PreparedBaseConvBlock,
    conv2: PreparedBaseConvBlock,
    pooling: [usize; 3],
}

struct PreparedDark5Block {
    conv: PreparedConvBlock,
    spp: PreparedSppBottleneckBlock,
    c3: PreparedCspBottleneckBlock,
}

struct PreparedFocusStemBlock {
    conv: PreparedBaseConvBlock,
}

struct PreparedBackbone {
    stem: PreparedFocusStemBlock,
    dark2: PreparedCspStageBlock,
    dark3: PreparedCspStageBlock,
    dark4: PreparedCspStageBlock,
    dark5: PreparedDark5Block,
}

struct PreparedPafpn {
    lateral_conv0: PreparedBaseConvBlock,
    c3_p4: PreparedCspBottleneckBlock,
    reduce_conv1: PreparedBaseConvBlock,
    c3_p3: PreparedCspBottleneckBlock,
    bu_conv2: PreparedConvBlock,
    c3_n3: PreparedCspBottleneckBlock,
    bu_conv1: PreparedConvBlock,
    c3_n4: PreparedCspBottleneckBlock,
}

struct PreparedHeadScale {
    stem: PreparedBaseConvBlock,
    cls_conv1: PreparedConvBlock,
    cls_conv2: PreparedConvBlock,
    reg_conv1: PreparedConvBlock,
    reg_conv2: PreparedConvBlock,
    cls_pred: GpuFusedConv2dWeights,
    reg_pred: GpuFusedConv2dWeights,
    obj_pred: GpuFusedConv2dWeights,
}

struct PreparedHead {
    head_s8: PreparedHeadScale,
    head_s16: PreparedHeadScale,
    head_s32: PreparedHeadScale,
}

pub struct GpuDecodeSession {
    runtime: VulkanRuntime,
}

pub struct GpuResidentDecodeSession {
    runtime: VulkanRuntime,
    backbone: PreparedBackbone,
    pafpn: PreparedPafpn,
    head: PreparedHead,
}

#[derive(Debug, Clone)]
pub struct VulkanDeviceInfo {
    pub device_name: String,
    pub device_type: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: String,
    pub driver_version: u32,
    pub driver_name: Option<String>,
    pub driver_info: Option<String>,
    pub driver_id: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct GpuDecodeStageDurations {
    pub upload: Duration,
    pub backbone: Duration,
    pub pafpn: Duration,
    pub head: Duration,
    pub decode: Duration,
    pub readback: Duration,
}

#[derive(Debug, Clone)]
pub struct VulkanFp16SupportInfo {
    pub device_name: String,
    pub api_version: String,
    pub driver_name: Option<String>,
    pub driver_info: Option<String>,
    pub extension_khr_shader_float16_int8: bool,
    pub shader_float16: bool,
    pub storage_buffer16_bit_access: bool,
    pub uniform_and_storage_buffer16_bit_access: bool,
    pub storage_input_output16: bool,
    pub workgroup_memory_explicit_layout16_bit_access: bool,
    pub shader_denorm_flush_to_zero_float16: Option<bool>,
    pub shader_denorm_preserve_float16: Option<bool>,
    pub shader_rounding_mode_rte_float16: Option<bool>,
    pub shader_rounding_mode_rtz_float16: Option<bool>,
}

struct VulkanRuntime {
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pipeline_cache: Mutex<HashMap<String, Arc<ComputePipeline>>>,
    submission_future: Mutex<Option<Box<dyn GpuFuture>>>,
    temp_buffer_pool: Mutex<HashMap<usize, Vec<Subbuffer<[f32]>>>>,
    enable_fp16: bool,
}

impl VulkanRuntime {
    fn new(device_index: usize) -> Result<Self> {
        Self::new_with_options(device_index, false)
    }

    fn new_with_options(device_index: usize, enable_fp16: bool) -> Result<Self> {
        let library = load_vulkan_library()?;
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                max_api_version: Some(Version::V1_0),
                enabled_extensions: InstanceExtensions {
                    khr_get_physical_device_properties2: true,
                    ..InstanceExtensions::empty()
                },
                ..Default::default()
            },
        )
        .context("falha ao criar instância Vulkan")?;

        let physical_devices = instance
            .enumerate_physical_devices()
            .context("falha ao enumerar GPUs Vulkan")?
            .collect::<Vec<_>>();
        let physical_device = physical_devices
            .get(device_index)
            .cloned()
            .ok_or_else(|| anyhow!("device_index {} inválido", device_index))?;
        let queue_family_index = select_queue_family(&physical_device)
            .ok_or_else(|| anyhow!("nenhuma queue family com suporte a compute foi encontrada"))?;

        let supported_extensions = physical_device.supported_extensions();
        let supported_features = physical_device.supported_features();
        let device_extensions = DeviceExtensions {
            khr_portability_subset: supported_extensions.khr_portability_subset,
            khr_storage_buffer_storage_class: supported_extensions.khr_storage_buffer_storage_class,
            khr_maintenance3: supported_extensions.khr_maintenance3,
            khr_16bit_storage: enable_fp16 && supported_extensions.khr_16bit_storage,
            khr_shader_float16_int8: enable_fp16 && supported_extensions.khr_shader_float16_int8,
            ..DeviceExtensions::empty()
        };
        let enabled_features = if enable_fp16 {
            if !supported_features.shader_float16
                || !supported_features.storage_buffer16_bit_access
                || !supported_features.uniform_and_storage_buffer16_bit_access
            {
                bail!("dispositivo Vulkan sem suporte suficiente para --fp16 experimental");
            }

            vulkano::device::DeviceFeatures {
                shader_float16: true,
                storage_buffer16_bit_access: true,
                uniform_and_storage_buffer16_bit_access: true,
                storage_input_output16: supported_features.storage_input_output16,
                ..Default::default()
            }
        } else {
            Default::default()
        };

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .context("falha ao criar o device lógico Vulkan")?;
        let queue = queues.next().context("falha ao obter queue de compute")?;

        Ok(Self {
            memory_allocator: Arc::new(StandardMemoryAllocator::new_default(device.clone())),
            command_buffer_allocator: Arc::new(StandardCommandBufferAllocator::new(
                device.clone(),
                StandardCommandBufferAllocatorCreateInfo::default(),
            )),
            descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(
                device.clone(),
                Default::default(),
            )),
            device: device.clone(),
            queue,
            pipeline_cache: Mutex::new(HashMap::new()),
            submission_future: Mutex::new(Some(vk_sync::now(device.clone()).boxed())),
            temp_buffer_pool: Mutex::new(HashMap::new()),
            enable_fp16,
        })
    }

    fn upload_tensor(&self, data: &[f32], shape: TensorShape) -> Result<GpuTensor> {
        if data.len() != shape.len() {
            bail!(
                "tensor incompatível com shape {}: esperado {} elementos, recebido {}",
                shape.display_nchw(),
                shape.len(),
                data.len()
            );
        }

        Ok(GpuTensor {
            shape,
            buffer: self.upload_device_buffer(data, BufferUsage::STORAGE_BUFFER)?,
        })
    }

    fn read_tensor(&self, tensor: &GpuTensor) -> Result<Vec<f32>> {
        let readback =
            self.create_host_readback_buffer::<f32>(tensor.shape.len(), BufferUsage::TRANSFER_DST)?;
        self.copy_buffer_with_pending_wait(tensor.buffer.clone(), readback.clone())?;
        let output = readback
            .read()
            .context("falha ao mapear o readback buffer")?;
        Ok(output.to_vec())
    }

    fn read_matrix(&self, matrix: &GpuMatrix) -> Result<Vec<f32>> {
        let readback = self.create_host_readback_buffer::<f32>(
            matrix.rows * matrix.cols,
            BufferUsage::TRANSFER_DST,
        )?;
        self.copy_buffer_with_pending_wait(matrix.buffer.clone(), readback.clone())?;
        let output = readback
            .read()
            .context("falha ao mapear o readback buffer da matriz")?;
        Ok(output.to_vec())
    }

    fn prepare_conv2d_weights(
        &self,
        weights: &FusedConv2dWeights,
    ) -> Result<GpuFusedConv2dWeights> {
        let weights_f16 = if self.enable_fp16 {
            Some(
                self.upload_device_buffer(
                    &weights
                        .weights
                        .iter()
                        .map(|item| f16::from_f32(*item).to_bits())
                        .collect::<Vec<_>>(),
                    BufferUsage::STORAGE_BUFFER,
                )?,
            )
        } else {
            None
        };
        Ok(GpuFusedConv2dWeights {
            spec: weights.spec,
            weights_f32: if self.enable_fp16 {
                None
            } else {
                Some(self.upload_device_buffer(&weights.weights, BufferUsage::STORAGE_BUFFER)?)
            },
            weights_f16,
            bias: self.upload_device_buffer(&weights.bias, BufferUsage::STORAGE_BUFFER)?,
            use_fp16: self.enable_fp16,
        })
    }

    fn prepare_baseconv_block(&self, block: &BaseConvBlock) -> Result<PreparedBaseConvBlock> {
        Ok(PreparedBaseConvBlock {
            conv: self.prepare_conv2d_weights(&block.conv)?,
        })
    }

    fn prepare_conv_block(&self, block: &ConvBlock) -> Result<PreparedConvBlock> {
        match block {
            ConvBlock::Base(base) => {
                Ok(PreparedConvBlock::Base(self.prepare_baseconv_block(base)?))
            }
            ConvBlock::Dws(dws) => Ok(PreparedConvBlock::Dws(PreparedDwsConvBlock {
                depthwise: self.prepare_conv2d_weights(&dws.depthwise)?,
                pointwise: self.prepare_conv2d_weights(&dws.pointwise)?,
            })),
        }
    }

    fn prepare_bottleneck_block(&self, block: &BottleneckBlock) -> Result<PreparedBottleneckBlock> {
        Ok(PreparedBottleneckBlock {
            conv1: self.prepare_baseconv_block(&block.conv1)?,
            conv2: self.prepare_conv_block(&block.conv2)?,
            shortcut: block.shortcut,
        })
    }

    fn prepare_csp_bottleneck_block(
        &self,
        block: &CspBottleneckBlock,
    ) -> Result<PreparedCspBottleneckBlock> {
        Ok(PreparedCspBottleneckBlock {
            conv1: self.prepare_baseconv_block(&block.conv1)?,
            conv2: self.prepare_baseconv_block(&block.conv2)?,
            conv3: self.prepare_baseconv_block(&block.conv3)?,
            blocks: block
                .blocks
                .iter()
                .map(|block| self.prepare_bottleneck_block(block))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn prepare_csp_stage_block(&self, block: &CspStageBlock) -> Result<PreparedCspStageBlock> {
        Ok(PreparedCspStageBlock {
            conv: self.prepare_conv_block(&block.conv)?,
            c3: self.prepare_csp_bottleneck_block(&block.c3)?,
        })
    }

    fn prepare_spp_bottleneck_block(
        &self,
        block: &SppBottleneckBlock,
    ) -> Result<PreparedSppBottleneckBlock> {
        Ok(PreparedSppBottleneckBlock {
            conv1: self.prepare_baseconv_block(&block.conv1)?,
            conv2: self.prepare_baseconv_block(&block.conv2)?,
            pooling: block.pooling,
        })
    }

    fn prepare_dark5_block(&self, block: &Dark5Block) -> Result<PreparedDark5Block> {
        Ok(PreparedDark5Block {
            conv: self.prepare_conv_block(&block.conv)?,
            spp: self.prepare_spp_bottleneck_block(&block.spp)?,
            c3: self.prepare_csp_bottleneck_block(&block.c3)?,
        })
    }

    fn prepare_backbone(&self, backbone: &CspDarknetDemo) -> Result<PreparedBackbone> {
        Ok(PreparedBackbone {
            stem: PreparedFocusStemBlock {
                conv: self.prepare_baseconv_block(&backbone.stem.conv)?,
            },
            dark2: self.prepare_csp_stage_block(&backbone.dark2)?,
            dark3: self.prepare_csp_stage_block(&backbone.dark3)?,
            dark4: self.prepare_csp_stage_block(&backbone.dark4)?,
            dark5: self.prepare_dark5_block(&backbone.dark5)?,
        })
    }

    fn prepare_pafpn(&self, pafpn: &YoloxPafpnDemo) -> Result<PreparedPafpn> {
        Ok(PreparedPafpn {
            lateral_conv0: self.prepare_baseconv_block(&pafpn.lateral_conv0)?,
            c3_p4: self.prepare_csp_bottleneck_block(&pafpn.c3_p4)?,
            reduce_conv1: self.prepare_baseconv_block(&pafpn.reduce_conv1)?,
            c3_p3: self.prepare_csp_bottleneck_block(&pafpn.c3_p3)?,
            bu_conv2: self.prepare_conv_block(&pafpn.bu_conv2)?,
            c3_n3: self.prepare_csp_bottleneck_block(&pafpn.c3_n3)?,
            bu_conv1: self.prepare_conv_block(&pafpn.bu_conv1)?,
            c3_n4: self.prepare_csp_bottleneck_block(&pafpn.c3_n4)?,
        })
    }

    fn prepare_head_scale(&self, head: &YoloxHeadScaleDemo) -> Result<PreparedHeadScale> {
        Ok(PreparedHeadScale {
            stem: self.prepare_baseconv_block(&head.stem)?,
            cls_conv1: self.prepare_conv_block(&head.cls_conv1)?,
            cls_conv2: self.prepare_conv_block(&head.cls_conv2)?,
            reg_conv1: self.prepare_conv_block(&head.reg_conv1)?,
            reg_conv2: self.prepare_conv_block(&head.reg_conv2)?,
            cls_pred: self.prepare_conv2d_weights(&head.cls_pred)?,
            reg_pred: self.prepare_conv2d_weights(&head.reg_pred)?,
            obj_pred: self.prepare_conv2d_weights(&head.obj_pred)?,
        })
    }

    fn prepare_head(&self, head: &YoloxHeadDemo) -> Result<PreparedHead> {
        Ok(PreparedHead {
            head_s8: self.prepare_head_scale(&head.head_s8)?,
            head_s16: self.prepare_head_scale(&head.head_s16)?,
            head_s32: self.prepare_head_scale(&head.head_s32)?,
        })
    }

    fn run_conv2d_tensor(
        &self,
        input: &GpuTensor,
        weights: &FusedConv2dWeights,
    ) -> Result<GpuTensor> {
        let prepared = self.prepare_conv2d_weights(weights)?;
        self.run_conv2d_prepared(input, &prepared)
    }

    fn run_conv2d_silu_tensor(
        &self,
        input: &GpuTensor,
        weights: &FusedConv2dWeights,
    ) -> Result<GpuTensor> {
        let prepared = self.prepare_conv2d_weights(weights)?;
        self.run_conv2d_silu_prepared(input, &prepared)
    }

    fn supports_1x1_fast_path(weights: &GpuFusedConv2dWeights) -> bool {
        weights.spec.kernel_h == 1
            && weights.spec.kernel_w == 1
            && weights.spec.pad_h == 0
            && weights.spec.pad_w == 0
    }

    fn supports_3x3_fast_path(weights: &GpuFusedConv2dWeights) -> bool {
        weights.spec.kernel_h == 3
            && weights.spec.kernel_w == 3
            && weights.spec.pad_h == 1
            && weights.spec.pad_w == 1
            && weights.spec.groups == 1
    }

    fn supports_depthwise_3x3_fast_path(weights: &GpuFusedConv2dWeights) -> bool {
        let _ = weights;
        false
    }

    fn run_conv2d_silu_prepared(
        &self,
        input: &GpuTensor,
        weights: &GpuFusedConv2dWeights,
    ) -> Result<GpuTensor> {
        if weights.use_fp16 {
            if Self::supports_1x1_fast_path(weights) {
                return self.run_conv2d_silu_1x1_prepared_fp16(input, weights);
            }
            if Self::supports_3x3_fast_path(weights) {
                return self.run_conv2d_silu_3x3_prepared_fp16(input, weights);
            }
            return self.run_conv2d_silu_prepared_fp16(input, weights);
        }
        if Self::supports_depthwise_3x3_fast_path(weights) {
            return self.run_depthwise_conv2d_silu_3x3_prepared(input, weights);
        }
        if Self::supports_1x1_fast_path(weights) {
            return self.run_conv2d_silu_1x1_prepared(input, weights);
        }
        if Self::supports_3x3_fast_path(weights) {
            return self.run_conv2d_silu_3x3_prepared(input, weights);
        }
        if input.shape.channels != weights.spec.in_channels {
            bail!(
                "conv2d incompatÃ­vel: input C={} pesos C={}",
                input.shape.channels,
                weights.spec.in_channels
            );
        }

        let (output_h, output_w) = weights
            .spec
            .output_hw(input.shape.height, input.shape.width)?;
        let output_shape = TensorShape::new(weights.spec.out_channels, output_h, output_w);
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            weights.spec.in_channels as u32,
            weights.spec.out_channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            weights.spec.kernel_h as u32,
            weights.spec.kernel_w as u32,
            weights.spec.stride_h as u32,
            weights.spec.stride_w as u32,
            weights.spec.pad_h as u32,
            weights.spec.pad_w as u32,
            weights.spec.groups as u32,
            weights.spec.in_channels_per_group() as u32,
            output_h as u32,
            output_w as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(CONV2D_SILU_WGSL, "conv2d-silu")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, weights.weights_f32()?),
                WriteDescriptorSet::buffer(2, weights.bias.clone()),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
                WriteDescriptorSet::buffer(4, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_w as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_h as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "conv2d-silu",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_conv2d_silu_prepared_fp16(
        &self,
        input: &GpuTensor,
        weights: &GpuFusedConv2dWeights,
    ) -> Result<GpuTensor> {
        if input.shape.channels != weights.spec.in_channels {
            bail!(
                "conv2d incompatível: input C={} pesos C={}",
                input.shape.channels,
                weights.spec.in_channels
            );
        }

        let (output_h, output_w) = weights
            .spec
            .output_hw(input.shape.height, input.shape.width)?;
        let output_shape = TensorShape::new(weights.spec.out_channels, output_h, output_w);
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            weights.spec.in_channels as u32,
            weights.spec.out_channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            weights.spec.kernel_h as u32,
            weights.spec.kernel_w as u32,
            weights.spec.stride_h as u32,
            weights.spec.stride_w as u32,
            weights.spec.pad_h as u32,
            weights.spec.pad_w as u32,
            weights.spec.groups as u32,
            weights.spec.in_channels_per_group() as u32,
            output_h as u32,
            output_w as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline =
            self.create_pipeline(CONV2D_SILU_FP16_WEIGHTS_WGSL, "conv2d-silu-fp16-weights")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, weights.weights_f16()?),
                WriteDescriptorSet::buffer(2, weights.bias.clone()),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
                WriteDescriptorSet::buffer(4, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_w as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_h as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "conv2d-silu-fp16-weights",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_conv2d_silu_1x1_prepared(
        &self,
        input: &GpuTensor,
        weights: &GpuFusedConv2dWeights,
    ) -> Result<GpuTensor> {
        if input.shape.channels != weights.spec.in_channels {
            bail!(
                "conv2d incompatÃ­vel: input C={} pesos C={}",
                input.shape.channels,
                weights.spec.in_channels
            );
        }

        let (output_h, output_w) = weights
            .spec
            .output_hw(input.shape.height, input.shape.width)?;
        let output_shape = TensorShape::new(weights.spec.out_channels, output_h, output_w);
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            weights.spec.in_channels as u32,
            weights.spec.out_channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            weights.spec.kernel_h as u32,
            weights.spec.kernel_w as u32,
            weights.spec.stride_h as u32,
            weights.spec.stride_w as u32,
            weights.spec.pad_h as u32,
            weights.spec.pad_w as u32,
            weights.spec.groups as u32,
            weights.spec.in_channels_per_group() as u32,
            output_h as u32,
            output_w as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(CONV2D_1X1_SILU_WGSL, "conv2d-1x1-silu")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, weights.weights_f32()?),
                WriteDescriptorSet::buffer(2, weights.bias.clone()),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
                WriteDescriptorSet::buffer(4, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_w as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_h as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "conv2d-1x1-silu",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_conv2d_silu_1x1_prepared_fp16(
        &self,
        input: &GpuTensor,
        weights: &GpuFusedConv2dWeights,
    ) -> Result<GpuTensor> {
        if input.shape.channels != weights.spec.in_channels {
            bail!(
                "conv2d incompatível: input C={} pesos C={}",
                input.shape.channels,
                weights.spec.in_channels
            );
        }

        let (output_h, output_w) = weights
            .spec
            .output_hw(input.shape.height, input.shape.width)?;
        let output_shape = TensorShape::new(weights.spec.out_channels, output_h, output_w);
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            weights.spec.in_channels as u32,
            weights.spec.out_channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            weights.spec.kernel_h as u32,
            weights.spec.kernel_w as u32,
            weights.spec.stride_h as u32,
            weights.spec.stride_w as u32,
            weights.spec.pad_h as u32,
            weights.spec.pad_w as u32,
            weights.spec.groups as u32,
            weights.spec.in_channels_per_group() as u32,
            output_h as u32,
            output_w as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline =
            self.create_pipeline(CONV2D_1X1_SILU_FP16_WEIGHTS_WGSL, "conv2d-1x1-silu-fp16")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, weights.weights_f16()?),
                WriteDescriptorSet::buffer(2, weights.bias.clone()),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
                WriteDescriptorSet::buffer(4, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_w as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_h as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "conv2d-1x1-silu-fp16",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_conv2d_silu_3x3_prepared(
        &self,
        input: &GpuTensor,
        weights: &GpuFusedConv2dWeights,
    ) -> Result<GpuTensor> {
        if input.shape.channels != weights.spec.in_channels {
            bail!(
                "conv2d incompatível: input C={} pesos C={}",
                input.shape.channels,
                weights.spec.in_channels
            );
        }

        let (output_h, output_w) = weights
            .spec
            .output_hw(input.shape.height, input.shape.width)?;
        let output_shape = TensorShape::new(weights.spec.out_channels, output_h, output_w);
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            weights.spec.in_channels as u32,
            weights.spec.out_channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            weights.spec.kernel_h as u32,
            weights.spec.kernel_w as u32,
            weights.spec.stride_h as u32,
            weights.spec.stride_w as u32,
            weights.spec.pad_h as u32,
            weights.spec.pad_w as u32,
            weights.spec.groups as u32,
            weights.spec.in_channels_per_group() as u32,
            output_h as u32,
            output_w as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(CONV2D_3X3_SILU_WGSL, "conv2d-3x3-silu")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, weights.weights_f32()?),
                WriteDescriptorSet::buffer(2, weights.bias.clone()),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
                WriteDescriptorSet::buffer(4, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_w as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_h as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "conv2d-3x3-silu",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_conv2d_silu_3x3_prepared_fp16(
        &self,
        input: &GpuTensor,
        weights: &GpuFusedConv2dWeights,
    ) -> Result<GpuTensor> {
        if input.shape.channels != weights.spec.in_channels {
            bail!(
                "conv2d incompatível: input C={} pesos C={}",
                input.shape.channels,
                weights.spec.in_channels
            );
        }

        let (output_h, output_w) = weights
            .spec
            .output_hw(input.shape.height, input.shape.width)?;
        let output_shape = TensorShape::new(weights.spec.out_channels, output_h, output_w);
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            weights.spec.in_channels as u32,
            weights.spec.out_channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            weights.spec.kernel_h as u32,
            weights.spec.kernel_w as u32,
            weights.spec.stride_h as u32,
            weights.spec.stride_w as u32,
            weights.spec.pad_h as u32,
            weights.spec.pad_w as u32,
            weights.spec.groups as u32,
            weights.spec.in_channels_per_group() as u32,
            output_h as u32,
            output_w as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline =
            self.create_pipeline(CONV2D_3X3_SILU_FP16_WEIGHTS_WGSL, "conv2d-3x3-silu-fp16")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, weights.weights_f16()?),
                WriteDescriptorSet::buffer(2, weights.bias.clone()),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
                WriteDescriptorSet::buffer(4, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_w as u32).div_ceil(FP16_CONV3X3_LOCAL_SIZE_X),
                (output_h as u32).div_ceil(FP16_CONV3X3_LOCAL_SIZE_Y),
                (output_shape.channels as u32)
                    .div_ceil(FP16_CONV3X3_OUTPUT_CHANNELS_PER_INVOCATION),
            ],
            "conv2d-3x3-silu-fp16",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_depthwise_conv2d_silu_3x3_prepared(
        &self,
        input: &GpuTensor,
        weights: &GpuFusedConv2dWeights,
    ) -> Result<GpuTensor> {
        if input.shape.channels != weights.spec.in_channels {
            bail!(
                "conv2d incompatível: input C={} pesos C={}",
                input.shape.channels,
                weights.spec.in_channels
            );
        }

        let (output_h, output_w) = weights
            .spec
            .output_hw(input.shape.height, input.shape.width)?;
        let output_shape = TensorShape::new(weights.spec.out_channels, output_h, output_w);
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            weights.spec.in_channels as u32,
            weights.spec.out_channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            weights.spec.kernel_h as u32,
            weights.spec.kernel_w as u32,
            weights.spec.stride_h as u32,
            weights.spec.stride_w as u32,
            weights.spec.pad_h as u32,
            weights.spec.pad_w as u32,
            weights.spec.groups as u32,
            weights.spec.in_channels_per_group() as u32,
            output_h as u32,
            output_w as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline =
            self.create_pipeline(DEPTHWISE_CONV2D_3X3_SILU_WGSL, "depthwise-conv2d-3x3-silu")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, weights.weights_f32()?),
                WriteDescriptorSet::buffer(2, weights.bias.clone()),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
                WriteDescriptorSet::buffer(4, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_w as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_h as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "depthwise-conv2d-3x3-silu",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_conv2d_prepared(
        &self,
        input: &GpuTensor,
        weights: &GpuFusedConv2dWeights,
    ) -> Result<GpuTensor> {
        if weights.use_fp16 {
            if Self::supports_1x1_fast_path(weights) {
                return self.run_conv2d_1x1_prepared_fp16(input, weights);
            }
            if Self::supports_3x3_fast_path(weights) {
                return self.run_conv2d_3x3_prepared_fp16(input, weights);
            }
            return self.run_conv2d_prepared_fp16(input, weights);
        }
        if Self::supports_depthwise_3x3_fast_path(weights) {
            return self.run_depthwise_conv2d_3x3_prepared(input, weights);
        }
        if Self::supports_1x1_fast_path(weights) {
            return self.run_conv2d_1x1_prepared(input, weights);
        }
        if Self::supports_3x3_fast_path(weights) {
            return self.run_conv2d_3x3_prepared(input, weights);
        }
        if input.shape.channels != weights.spec.in_channels {
            bail!(
                "conv2d incompatível: input C={} pesos C={}",
                input.shape.channels,
                weights.spec.in_channels
            );
        }

        let (output_h, output_w) = weights
            .spec
            .output_hw(input.shape.height, input.shape.width)?;
        let output_shape = TensorShape::new(weights.spec.out_channels, output_h, output_w);
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            weights.spec.in_channels as u32,
            weights.spec.out_channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            weights.spec.kernel_h as u32,
            weights.spec.kernel_w as u32,
            weights.spec.stride_h as u32,
            weights.spec.stride_w as u32,
            weights.spec.pad_h as u32,
            weights.spec.pad_w as u32,
            weights.spec.groups as u32,
            weights.spec.in_channels_per_group() as u32,
            output_h as u32,
            output_w as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(CONV2D_WGSL, "conv2d")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, weights.weights_f32()?),
                WriteDescriptorSet::buffer(2, weights.bias.clone()),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
                WriteDescriptorSet::buffer(4, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_w as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_h as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "conv2d",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_conv2d_prepared_fp16(
        &self,
        input: &GpuTensor,
        weights: &GpuFusedConv2dWeights,
    ) -> Result<GpuTensor> {
        if input.shape.channels != weights.spec.in_channels {
            bail!(
                "conv2d incompatível: input C={} pesos C={}",
                input.shape.channels,
                weights.spec.in_channels
            );
        }

        let (output_h, output_w) = weights
            .spec
            .output_hw(input.shape.height, input.shape.width)?;
        let output_shape = TensorShape::new(weights.spec.out_channels, output_h, output_w);
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            weights.spec.in_channels as u32,
            weights.spec.out_channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            weights.spec.kernel_h as u32,
            weights.spec.kernel_w as u32,
            weights.spec.stride_h as u32,
            weights.spec.stride_w as u32,
            weights.spec.pad_h as u32,
            weights.spec.pad_w as u32,
            weights.spec.groups as u32,
            weights.spec.in_channels_per_group() as u32,
            output_h as u32,
            output_w as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(CONV2D_FP16_WEIGHTS_WGSL, "conv2d-fp16-weights")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, weights.weights_f16()?),
                WriteDescriptorSet::buffer(2, weights.bias.clone()),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
                WriteDescriptorSet::buffer(4, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_w as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_h as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "conv2d-fp16-weights",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_conv2d_1x1_prepared(
        &self,
        input: &GpuTensor,
        weights: &GpuFusedConv2dWeights,
    ) -> Result<GpuTensor> {
        if input.shape.channels != weights.spec.in_channels {
            bail!(
                "conv2d incompatÃ­vel: input C={} pesos C={}",
                input.shape.channels,
                weights.spec.in_channels
            );
        }

        let (output_h, output_w) = weights
            .spec
            .output_hw(input.shape.height, input.shape.width)?;
        let output_shape = TensorShape::new(weights.spec.out_channels, output_h, output_w);
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            weights.spec.in_channels as u32,
            weights.spec.out_channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            weights.spec.kernel_h as u32,
            weights.spec.kernel_w as u32,
            weights.spec.stride_h as u32,
            weights.spec.stride_w as u32,
            weights.spec.pad_h as u32,
            weights.spec.pad_w as u32,
            weights.spec.groups as u32,
            weights.spec.in_channels_per_group() as u32,
            output_h as u32,
            output_w as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(CONV2D_1X1_WGSL, "conv2d-1x1")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, weights.weights_f32()?),
                WriteDescriptorSet::buffer(2, weights.bias.clone()),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
                WriteDescriptorSet::buffer(4, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_w as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_h as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "conv2d-1x1",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_conv2d_1x1_prepared_fp16(
        &self,
        input: &GpuTensor,
        weights: &GpuFusedConv2dWeights,
    ) -> Result<GpuTensor> {
        if input.shape.channels != weights.spec.in_channels {
            bail!(
                "conv2d incompatível: input C={} pesos C={}",
                input.shape.channels,
                weights.spec.in_channels
            );
        }

        let (output_h, output_w) = weights
            .spec
            .output_hw(input.shape.height, input.shape.width)?;
        let output_shape = TensorShape::new(weights.spec.out_channels, output_h, output_w);
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            weights.spec.in_channels as u32,
            weights.spec.out_channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            weights.spec.kernel_h as u32,
            weights.spec.kernel_w as u32,
            weights.spec.stride_h as u32,
            weights.spec.stride_w as u32,
            weights.spec.pad_h as u32,
            weights.spec.pad_w as u32,
            weights.spec.groups as u32,
            weights.spec.in_channels_per_group() as u32,
            output_h as u32,
            output_w as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(CONV2D_1X1_FP16_WEIGHTS_WGSL, "conv2d-1x1-fp16")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, weights.weights_f16()?),
                WriteDescriptorSet::buffer(2, weights.bias.clone()),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
                WriteDescriptorSet::buffer(4, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_w as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_h as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "conv2d-1x1-fp16",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_conv2d_3x3_prepared(
        &self,
        input: &GpuTensor,
        weights: &GpuFusedConv2dWeights,
    ) -> Result<GpuTensor> {
        if input.shape.channels != weights.spec.in_channels {
            bail!(
                "conv2d incompatível: input C={} pesos C={}",
                input.shape.channels,
                weights.spec.in_channels
            );
        }

        let (output_h, output_w) = weights
            .spec
            .output_hw(input.shape.height, input.shape.width)?;
        let output_shape = TensorShape::new(weights.spec.out_channels, output_h, output_w);
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            weights.spec.in_channels as u32,
            weights.spec.out_channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            weights.spec.kernel_h as u32,
            weights.spec.kernel_w as u32,
            weights.spec.stride_h as u32,
            weights.spec.stride_w as u32,
            weights.spec.pad_h as u32,
            weights.spec.pad_w as u32,
            weights.spec.groups as u32,
            weights.spec.in_channels_per_group() as u32,
            output_h as u32,
            output_w as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(CONV2D_3X3_WGSL, "conv2d-3x3")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, weights.weights_f32()?),
                WriteDescriptorSet::buffer(2, weights.bias.clone()),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
                WriteDescriptorSet::buffer(4, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_w as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_h as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "conv2d-3x3",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_conv2d_3x3_prepared_fp16(
        &self,
        input: &GpuTensor,
        weights: &GpuFusedConv2dWeights,
    ) -> Result<GpuTensor> {
        if input.shape.channels != weights.spec.in_channels {
            bail!(
                "conv2d incompatível: input C={} pesos C={}",
                input.shape.channels,
                weights.spec.in_channels
            );
        }

        let (output_h, output_w) = weights
            .spec
            .output_hw(input.shape.height, input.shape.width)?;
        let output_shape = TensorShape::new(weights.spec.out_channels, output_h, output_w);
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            weights.spec.in_channels as u32,
            weights.spec.out_channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            weights.spec.kernel_h as u32,
            weights.spec.kernel_w as u32,
            weights.spec.stride_h as u32,
            weights.spec.stride_w as u32,
            weights.spec.pad_h as u32,
            weights.spec.pad_w as u32,
            weights.spec.groups as u32,
            weights.spec.in_channels_per_group() as u32,
            output_h as u32,
            output_w as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(CONV2D_3X3_FP16_WEIGHTS_WGSL, "conv2d-3x3-fp16")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, weights.weights_f16()?),
                WriteDescriptorSet::buffer(2, weights.bias.clone()),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
                WriteDescriptorSet::buffer(4, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_w as u32).div_ceil(FP16_CONV3X3_LOCAL_SIZE_X),
                (output_h as u32).div_ceil(FP16_CONV3X3_LOCAL_SIZE_Y),
                (output_shape.channels as u32)
                    .div_ceil(FP16_CONV3X3_OUTPUT_CHANNELS_PER_INVOCATION),
            ],
            "conv2d-3x3-fp16",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_depthwise_conv2d_3x3_prepared(
        &self,
        input: &GpuTensor,
        weights: &GpuFusedConv2dWeights,
    ) -> Result<GpuTensor> {
        if input.shape.channels != weights.spec.in_channels {
            bail!(
                "conv2d incompatível: input C={} pesos C={}",
                input.shape.channels,
                weights.spec.in_channels
            );
        }

        let (output_h, output_w) = weights
            .spec
            .output_hw(input.shape.height, input.shape.width)?;
        let output_shape = TensorShape::new(weights.spec.out_channels, output_h, output_w);
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            weights.spec.in_channels as u32,
            weights.spec.out_channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            weights.spec.kernel_h as u32,
            weights.spec.kernel_w as u32,
            weights.spec.stride_h as u32,
            weights.spec.stride_w as u32,
            weights.spec.pad_h as u32,
            weights.spec.pad_w as u32,
            weights.spec.groups as u32,
            weights.spec.in_channels_per_group() as u32,
            output_h as u32,
            output_w as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(DEPTHWISE_CONV2D_3X3_WGSL, "depthwise-conv2d-3x3")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, weights.weights_f32()?),
                WriteDescriptorSet::buffer(2, weights.bias.clone()),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
                WriteDescriptorSet::buffer(4, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_w as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_h as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "depthwise-conv2d-3x3",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_silu_tensor(&self, input: &GpuTensor) -> Result<GpuTensor> {
        let output_buffer =
            self.create_temp_buffer_f32(input.shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [input.shape.len() as u32];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(SILU_WGSL, "silu")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, output_buffer.clone()),
                WriteDescriptorSet::buffer(2, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (input.shape.len() as u32).div_ceil(LINEAR_LOCAL_SIZE_X),
                1,
                1,
            ],
            "silu",
        )?;

        Ok(GpuTensor {
            shape: input.shape,
            buffer: output_buffer,
        })
    }

    fn run_upsample_nearest_tensor(&self, input: &GpuTensor, scale: usize) -> Result<GpuTensor> {
        if scale == 0 {
            bail!("scale deve ser maior que zero");
        }

        let output_shape = TensorShape::new(
            input.shape.channels,
            input.shape.height * scale,
            input.shape.width * scale,
        );
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            input.shape.channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            scale as u32,
            output_shape.height as u32,
            output_shape.width as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(UPSAMPLE_NEAREST_WGSL, "upsample-nearest")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, output_buffer.clone()),
                WriteDescriptorSet::buffer(2, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_shape.width as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_shape.height as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "upsample-nearest",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_concat_channels_tensor(&self, lhs: &GpuTensor, rhs: &GpuTensor) -> Result<GpuTensor> {
        if lhs.shape.height != rhs.shape.height || lhs.shape.width != rhs.shape.width {
            bail!(
                "concat requer mesmo HxW: lhs={} rhs={}",
                lhs.shape.display_nchw(),
                rhs.shape.display_nchw()
            );
        }

        let output_shape = TensorShape::new(
            lhs.shape.channels + rhs.shape.channels,
            lhs.shape.height,
            lhs.shape.width,
        );
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            lhs.shape.channels as u32,
            rhs.shape.channels as u32,
            lhs.shape.height as u32,
            lhs.shape.width as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(CONCAT_CHANNELS_WGSL, "concat-channels")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, lhs.buffer.clone()),
                WriteDescriptorSet::buffer(1, rhs.buffer.clone()),
                WriteDescriptorSet::buffer(2, output_buffer.clone()),
                WriteDescriptorSet::buffer(3, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_shape.width as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_shape.height as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "concat-channels",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_sigmoid_tensor(&self, input: &GpuTensor) -> Result<GpuTensor> {
        let output_buffer =
            self.create_temp_buffer_f32(input.shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [input.shape.len() as u32];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(SIGMOID_WGSL, "sigmoid")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, output_buffer.clone()),
                WriteDescriptorSet::buffer(2, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (input.shape.len() as u32).div_ceil(LINEAR_LOCAL_SIZE_X),
                1,
                1,
            ],
            "sigmoid",
        )?;

        Ok(GpuTensor {
            shape: input.shape,
            buffer: output_buffer,
        })
    }

    fn run_add_tensors(&self, lhs: &GpuTensor, rhs: &GpuTensor) -> Result<GpuTensor> {
        if lhs.shape != rhs.shape {
            bail!(
                "add requer shapes idênticos: lhs={} rhs={}",
                lhs.shape.display_nchw(),
                rhs.shape.display_nchw()
            );
        }

        let output_buffer =
            self.create_temp_buffer_f32(lhs.shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [lhs.shape.len() as u32];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(ADD_WGSL, "add")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, lhs.buffer.clone()),
                WriteDescriptorSet::buffer(1, rhs.buffer.clone()),
                WriteDescriptorSet::buffer(2, output_buffer.clone()),
                WriteDescriptorSet::buffer(3, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [(lhs.shape.len() as u32).div_ceil(LINEAR_LOCAL_SIZE_X), 1, 1],
            "add",
        )?;

        Ok(GpuTensor {
            shape: lhs.shape,
            buffer: output_buffer,
        })
    }

    fn run_maxpool2d_tensor(
        &self,
        input: &GpuTensor,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<GpuTensor> {
        if kernel == 0 || stride == 0 {
            bail!("kernel e stride devem ser maiores que zero");
        }

        let padded_h = input.shape.height + padding * 2;
        let padded_w = input.shape.width + padding * 2;
        if padded_h < kernel || padded_w < kernel {
            bail!("kernel maior que a entrada efetiva após padding");
        }
        let output_shape = TensorShape::new(
            input.shape.channels,
            (padded_h - kernel) / stride + 1,
            (padded_w - kernel) / stride + 1,
        );
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            input.shape.channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            kernel as u32,
            stride as u32,
            padding as u32,
            output_shape.height as u32,
            output_shape.width as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(MAXPOOL2D_WGSL, "maxpool2d")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, output_buffer.clone()),
                WriteDescriptorSet::buffer(2, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_shape.width as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_shape.height as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "maxpool2d",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_concat3_channels_tensor(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
    ) -> Result<GpuTensor> {
        if a.shape.height != b.shape.height
            || a.shape.height != c.shape.height
            || a.shape.width != b.shape.width
            || a.shape.width != c.shape.width
        {
            bail!(
                "concat3 requer mesmo HxW: a={} b={} c={}",
                a.shape.display_nchw(),
                b.shape.display_nchw(),
                c.shape.display_nchw()
            );
        }

        let output_shape = TensorShape::new(
            a.shape.channels + b.shape.channels + c.shape.channels,
            a.shape.height,
            a.shape.width,
        );
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            a.shape.channels as u32,
            b.shape.channels as u32,
            c.shape.channels as u32,
            a.shape.height as u32,
            a.shape.width as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(CONCAT3_CHANNELS_WGSL, "concat3-channels")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, a.buffer.clone()),
                WriteDescriptorSet::buffer(1, b.buffer.clone()),
                WriteDescriptorSet::buffer(2, c.buffer.clone()),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
                WriteDescriptorSet::buffer(4, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_shape.width as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_shape.height as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "concat3-channels",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_spp_concat_tensor(&self, input: &GpuTensor, pooling: [usize; 3]) -> Result<GpuTensor> {
        if pooling.iter().any(|kernel| *kernel == 0 || kernel % 2 == 0) {
            bail!("SPP concat exige kernels impares e maiores que zero");
        }

        let output_shape = TensorShape::new(
            input.shape.channels * 4,
            input.shape.height,
            input.shape.width,
        );
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            input.shape.channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            pooling[0] as u32,
            pooling[1] as u32,
            pooling[2] as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(SPP_CONCAT_WGSL, "spp-concat")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, output_buffer.clone()),
                WriteDescriptorSet::buffer(2, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_shape.width as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_shape.height as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "spp-concat",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_focus_tensor(&self, input: &GpuTensor) -> Result<GpuTensor> {
        if input.shape.height % 2 != 0 || input.shape.width % 2 != 0 {
            bail!(
                "focus exige H e W pares, recebido {}",
                input.shape.display_nchw()
            );
        }

        let output_shape = TensorShape::new(
            input.shape.channels * 4,
            input.shape.height / 2,
            input.shape.width / 2,
        );
        let output_buffer =
            self.create_temp_buffer_f32(output_shape.len(), BufferUsage::STORAGE_BUFFER)?;
        let params = [
            input.shape.channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            output_shape.height as u32,
            output_shape.width as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(FOCUS_WGSL, "focus")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, output_buffer.clone()),
                WriteDescriptorSet::buffer(2, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (output_shape.width as u32).div_ceil(IMAGE_LOCAL_SIZE_X),
                (output_shape.height as u32).div_ceil(IMAGE_LOCAL_SIZE_Y),
                output_shape.channels as u32,
            ],
            "focus",
        )?;

        Ok(GpuTensor {
            shape: output_shape,
            buffer: output_buffer,
        })
    }

    fn run_baseconv_block(&self, input: &GpuTensor, block: &BaseConvBlock) -> Result<GpuTensor> {
        self.run_conv2d_silu_tensor(input, &block.conv)
    }

    fn run_baseconv_prepared(
        &self,
        input: &GpuTensor,
        block: &PreparedBaseConvBlock,
    ) -> Result<GpuTensor> {
        self.run_conv2d_silu_prepared(input, &block.conv)
    }

    fn run_conv_block(&self, input: &GpuTensor, block: &ConvBlock) -> Result<GpuTensor> {
        match block {
            ConvBlock::Base(base) => self.run_baseconv_block(input, base),
            ConvBlock::Dws(dws) => self.run_dwsconv_block(input, dws),
        }
    }

    fn run_conv_prepared(&self, input: &GpuTensor, block: &PreparedConvBlock) -> Result<GpuTensor> {
        match block {
            PreparedConvBlock::Base(base) => self.run_baseconv_prepared(input, base),
            PreparedConvBlock::Dws(dws) => self.run_dws_prepared(input, dws),
        }
    }

    fn run_focus_stem_block(&self, input: &GpuTensor, block: &FocusStemBlock) -> Result<GpuTensor> {
        let focused = self.run_focus_tensor(input)?;
        self.run_baseconv_block(&focused, &block.conv)
    }

    fn run_dwsconv_block(&self, input: &GpuTensor, block: &DwsConvBlock) -> Result<GpuTensor> {
        let depthwise = self.run_conv2d_silu_tensor(input, &block.depthwise)?;
        self.run_conv2d_silu_tensor(&depthwise, &block.pointwise)
    }

    fn run_dws_prepared(
        &self,
        input: &GpuTensor,
        block: &PreparedDwsConvBlock,
    ) -> Result<GpuTensor> {
        let depthwise = self.run_conv2d_silu_prepared(input, &block.depthwise)?;
        let pointwise = self.run_conv2d_silu_prepared(&depthwise, &block.pointwise)?;
        self.recycle_tensor(depthwise)?;
        Ok(pointwise)
    }

    fn run_csp_bottleneck_block(
        &self,
        input: &GpuTensor,
        block: &CspBottleneckBlock,
    ) -> Result<GpuTensor> {
        let (mut x1, x2) = (
            self.run_baseconv_block(input, &block.conv1)?,
            self.run_baseconv_block(input, &block.conv2)?,
        );

        for bottleneck in &block.blocks {
            let next = self.run_bottleneck_block(&x1, bottleneck)?;
            x1 = next;
        }

        let cat = self.run_concat_channels_tensor(&x1, &x2)?;
        self.run_baseconv_block(&cat, &block.conv3)
    }

    fn run_csp_bottleneck_prepared(
        &self,
        input: &GpuTensor,
        block: &PreparedCspBottleneckBlock,
    ) -> Result<GpuTensor> {
        let (mut x1, x2) = (
            self.run_baseconv_prepared(input, &block.conv1)?,
            self.run_baseconv_prepared(input, &block.conv2)?,
        );

        for bottleneck in &block.blocks {
            let prev_x1 = x1;
            x1 = self.run_bottleneck_prepared(&prev_x1, bottleneck)?;
            self.recycle_tensor(prev_x1)?;
        }

        let cat = self.run_concat_channels_tensor(&x1, &x2)?;
        self.recycle_tensor(x1)?;
        self.recycle_tensor(x2)?;
        let output = self.run_baseconv_prepared(&cat, &block.conv3)?;
        self.recycle_tensor(cat)?;
        Ok(output)
    }

    fn run_csp_stage_block(&self, input: &GpuTensor, block: &CspStageBlock) -> Result<GpuTensor> {
        let x = self.run_conv_block(input, &block.conv)?;
        self.run_csp_bottleneck_block(&x, &block.c3)
    }

    fn run_csp_stage_prepared(
        &self,
        input: &GpuTensor,
        block: &PreparedCspStageBlock,
    ) -> Result<GpuTensor> {
        let x = self.run_conv_prepared(input, &block.conv)?;
        let output = self.run_csp_bottleneck_prepared(&x, &block.c3)?;
        self.recycle_tensor(x)?;
        Ok(output)
    }

    fn run_spp_bottleneck_block(
        &self,
        input: &GpuTensor,
        block: &SppBottleneckBlock,
    ) -> Result<GpuTensor> {
        let base = self.run_baseconv_block(input, &block.conv1)?;
        let cat = self.run_spp_concat_tensor(&base, block.pooling)?;
        self.run_baseconv_block(&cat, &block.conv2)
    }

    fn run_spp_bottleneck_prepared(
        &self,
        input: &GpuTensor,
        block: &PreparedSppBottleneckBlock,
    ) -> Result<GpuTensor> {
        let base = self.run_baseconv_prepared(input, &block.conv1)?;
        let cat = self.run_spp_concat_tensor(&base, block.pooling)?;
        self.recycle_tensor(base)?;
        let output = self.run_baseconv_prepared(&cat, &block.conv2)?;
        self.recycle_tensor(cat)?;
        Ok(output)
    }

    fn run_dark5_block(&self, input: &GpuTensor, block: &Dark5Block) -> Result<GpuTensor> {
        let x = self.run_conv_block(input, &block.conv)?;
        let x = self.run_spp_bottleneck_block(&x, &block.spp)?;
        self.run_csp_bottleneck_block(&x, &block.c3)
    }

    fn run_dark5_prepared(
        &self,
        input: &GpuTensor,
        block: &PreparedDark5Block,
    ) -> Result<GpuTensor> {
        let x = self.run_conv_prepared(input, &block.conv)?;
        let spp = self.run_spp_bottleneck_prepared(&x, &block.spp)?;
        self.recycle_tensor(x)?;
        let output = self.run_csp_bottleneck_prepared(&spp, &block.c3)?;
        self.recycle_tensor(spp)?;
        Ok(output)
    }

    fn run_backbone(
        &self,
        input: &GpuTensor,
        backbone: &CspDarknetDemo,
    ) -> Result<GpuDarknetFeatures> {
        let stem = self.run_focus_stem_block(input, &backbone.stem)?;
        let dark2 = self.run_csp_stage_block(&stem, &backbone.dark2)?;
        let f1 = self.run_csp_stage_block(&dark2, &backbone.dark3)?;
        let f2 = self.run_csp_stage_block(&f1, &backbone.dark4)?;
        let f3 = self.run_dark5_block(&f2, &backbone.dark5)?;

        Ok(GpuDarknetFeatures { f1, f2, f3 })
    }

    fn run_backbone_prepared(
        &self,
        input: &GpuTensor,
        backbone: &PreparedBackbone,
    ) -> Result<GpuDarknetFeatures> {
        let focused = self.run_focus_tensor(input)?;
        let stem = self.run_baseconv_prepared(&focused, &backbone.stem.conv)?;
        self.recycle_tensor(focused)?;
        let dark2 = self.run_csp_stage_prepared(&stem, &backbone.dark2)?;
        self.recycle_tensor(stem)?;
        let f1 = self.run_csp_stage_prepared(&dark2, &backbone.dark3)?;
        self.recycle_tensor(dark2)?;
        let f2 = self.run_csp_stage_prepared(&f1, &backbone.dark4)?;
        let f3 = self.run_dark5_prepared(&f2, &backbone.dark5)?;

        Ok(GpuDarknetFeatures { f1, f2, f3 })
    }

    fn run_pafpn(
        &self,
        features: &GpuDarknetFeatures,
        pafpn: &YoloxPafpnDemo,
    ) -> Result<GpuPafpnFeatures> {
        let fpn_out0 = self.run_baseconv_block(&features.f3, &pafpn.lateral_conv0)?;
        let f_out0_up = self.run_upsample_nearest_tensor(&fpn_out0, 2)?;
        let cat_p4 = self.run_concat_channels_tensor(&f_out0_up, &features.f2)?;
        let f_out0 = self.run_csp_bottleneck_block(&cat_p4, &pafpn.c3_p4)?;

        let fpn_out1 = self.run_baseconv_block(&f_out0, &pafpn.reduce_conv1)?;
        let f_out1_up = self.run_upsample_nearest_tensor(&fpn_out1, 2)?;
        let cat_p3 = self.run_concat_channels_tensor(&f_out1_up, &features.f1)?;
        let p3 = self.run_csp_bottleneck_block(&cat_p3, &pafpn.c3_p3)?;

        let p_out1_down = self.run_conv_block(&p3, &pafpn.bu_conv2)?;
        let cat_n3 = self.run_concat_channels_tensor(&p_out1_down, &fpn_out1)?;
        let p4 = self.run_csp_bottleneck_block(&cat_n3, &pafpn.c3_n3)?;

        let p_out0_down = self.run_conv_block(&p4, &pafpn.bu_conv1)?;
        let cat_n4 = self.run_concat_channels_tensor(&p_out0_down, &fpn_out0)?;
        let p5 = self.run_csp_bottleneck_block(&cat_n4, &pafpn.c3_n4)?;

        Ok(GpuPafpnFeatures { p3, p4, p5 })
    }

    fn run_pafpn_prepared(
        &self,
        features: &GpuDarknetFeatures,
        pafpn: &PreparedPafpn,
    ) -> Result<GpuPafpnFeatures> {
        let fpn_out0 = self.run_baseconv_prepared(&features.f3, &pafpn.lateral_conv0)?;
        let f_out0_up = self.run_upsample_nearest_tensor(&fpn_out0, 2)?;
        let cat_p4 = self.run_concat_channels_tensor(&f_out0_up, &features.f2)?;
        self.recycle_tensor(f_out0_up)?;
        let f_out0 = self.run_csp_bottleneck_prepared(&cat_p4, &pafpn.c3_p4)?;
        self.recycle_tensor(cat_p4)?;

        let fpn_out1 = self.run_baseconv_prepared(&f_out0, &pafpn.reduce_conv1)?;
        self.recycle_tensor(f_out0)?;
        let f_out1_up = self.run_upsample_nearest_tensor(&fpn_out1, 2)?;
        let cat_p3 = self.run_concat_channels_tensor(&f_out1_up, &features.f1)?;
        self.recycle_tensor(f_out1_up)?;
        let p3 = self.run_csp_bottleneck_prepared(&cat_p3, &pafpn.c3_p3)?;
        self.recycle_tensor(cat_p3)?;

        let p_out1_down = self.run_conv_prepared(&p3, &pafpn.bu_conv2)?;
        let cat_n3 = self.run_concat_channels_tensor(&p_out1_down, &fpn_out1)?;
        self.recycle_tensor(p_out1_down)?;
        self.recycle_tensor(fpn_out1)?;
        let p4 = self.run_csp_bottleneck_prepared(&cat_n3, &pafpn.c3_n3)?;
        self.recycle_tensor(cat_n3)?;

        let p_out0_down = self.run_conv_prepared(&p4, &pafpn.bu_conv1)?;
        let cat_n4 = self.run_concat_channels_tensor(&p_out0_down, &fpn_out0)?;
        self.recycle_tensor(p_out0_down)?;
        self.recycle_tensor(fpn_out0)?;
        let p5 = self.run_csp_bottleneck_prepared(&cat_n4, &pafpn.c3_n4)?;
        self.recycle_tensor(cat_n4)?;

        Ok(GpuPafpnFeatures { p3, p4, p5 })
    }

    fn run_head_scale_block(
        &self,
        input: &GpuTensor,
        head: &YoloxHeadScaleDemo,
    ) -> Result<GpuTensor> {
        let stem = self.run_baseconv_block(input, &head.stem)?;

        let cls = self.run_conv_block(&stem, &head.cls_conv1)?;
        let cls = self.run_conv_block(&cls, &head.cls_conv2)?;
        let cls_pred = self.run_conv2d_tensor(&cls, &head.cls_pred)?;

        let reg = self.run_conv_block(&stem, &head.reg_conv1)?;
        let reg = self.run_conv_block(&reg, &head.reg_conv2)?;
        let reg_pred = self.run_conv2d_tensor(&reg, &head.reg_pred)?;
        let obj_pred = self.run_conv2d_tensor(&reg, &head.obj_pred)?;

        self.run_concat3_channels_tensor(&reg_pred, &obj_pred, &cls_pred)
    }

    fn run_head_scale_prepared(
        &self,
        input: &GpuTensor,
        head: &PreparedHeadScale,
    ) -> Result<GpuTensor> {
        let stem = self.run_baseconv_prepared(input, &head.stem)?;

        let cls = self.run_conv_prepared(&stem, &head.cls_conv1)?;
        let cls2 = self.run_conv_prepared(&cls, &head.cls_conv2)?;
        self.recycle_tensor(cls)?;
        let cls_pred = self.run_conv2d_prepared(&cls2, &head.cls_pred)?;
        self.recycle_tensor(cls2)?;

        let reg = self.run_conv_prepared(&stem, &head.reg_conv1)?;
        self.recycle_tensor(stem)?;
        let reg2 = self.run_conv_prepared(&reg, &head.reg_conv2)?;
        self.recycle_tensor(reg)?;
        let reg_pred = self.run_conv2d_prepared(&reg2, &head.reg_pred)?;
        let obj_pred = self.run_conv2d_prepared(&reg2, &head.obj_pred)?;
        self.recycle_tensor(reg2)?;

        let output = self.run_concat3_channels_tensor(&reg_pred, &obj_pred, &cls_pred)?;
        self.recycle_tensor(reg_pred)?;
        self.recycle_tensor(obj_pred)?;
        self.recycle_tensor(cls_pred)?;
        Ok(output)
    }

    fn run_head(
        &self,
        features: &GpuPafpnFeatures,
        head: &YoloxHeadDemo,
    ) -> Result<GpuHeadFeatures> {
        let s8 = self.run_head_scale_block(&features.p3, &head.head_s8)?;
        let s16 = self.run_head_scale_block(&features.p4, &head.head_s16)?;
        let s32 = self.run_head_scale_block(&features.p5, &head.head_s32)?;

        Ok(GpuHeadFeatures { s8, s16, s32 })
    }

    fn run_head_prepared(
        &self,
        features: &GpuPafpnFeatures,
        head: &PreparedHead,
    ) -> Result<GpuHeadFeatures> {
        let s8 = self.run_head_scale_prepared(&features.p3, &head.head_s8)?;
        let s16 = self.run_head_scale_prepared(&features.p4, &head.head_s16)?;
        let s32 = self.run_head_scale_prepared(&features.p5, &head.head_s32)?;

        Ok(GpuHeadFeatures { s8, s16, s32 })
    }

    #[allow(dead_code)]
    fn run_decode_head_scale(
        &self,
        input: &GpuTensor,
        stride: usize,
        num_classes: usize,
    ) -> Result<GpuMatrix> {
        let cols = num_classes + 5;
        if input.shape.channels != cols {
            bail!(
                "decode incompatível: esperado C={} recebido C={}",
                cols,
                input.shape.channels
            );
        }

        let rows = input.shape.height * input.shape.width;
        let output_buffer =
            self.create_temp_buffer_f32(rows * cols, BufferUsage::STORAGE_BUFFER)?;
        let params = [
            input.shape.channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            stride as u32,
            num_classes as u32,
            rows as u32,
            cols as u32,
            0,
            rows as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(DECODE_HEAD_SCALE_WGSL, "decode-head-scale")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, output_buffer.clone()),
                WriteDescriptorSet::buffer(2, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (cols as u32).div_ceil(MATRIX_LOCAL_SIZE_X),
                (rows as u32).div_ceil(MATRIX_LOCAL_SIZE_Y),
                1,
            ],
            "decode-head-scale",
        )?;

        Ok(GpuMatrix {
            rows,
            cols,
            buffer: output_buffer,
        })
    }

    fn run_decode_head_scale_into(
        &self,
        input: &GpuTensor,
        stride: usize,
        num_classes: usize,
        output_buffer: &Subbuffer<[f32]>,
        output_rows: usize,
        row_offset: usize,
    ) -> Result<()> {
        let cols = num_classes + 5;
        if input.shape.channels != cols {
            bail!(
                "decode incompatÃ­vel: esperado C={} recebido C={}",
                cols,
                input.shape.channels
            );
        }

        let rows = input.shape.height * input.shape.width;
        let params = [
            input.shape.channels as u32,
            input.shape.height as u32,
            input.shape.width as u32,
            stride as u32,
            num_classes as u32,
            rows as u32,
            cols as u32,
            row_offset as u32,
            output_rows as u32,
        ];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(DECODE_HEAD_SCALE_WGSL, "decode-head-scale")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, input.buffer.clone()),
                WriteDescriptorSet::buffer(1, output_buffer.clone()),
                WriteDescriptorSet::buffer(2, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (cols as u32).div_ceil(MATRIX_LOCAL_SIZE_X),
                (rows as u32).div_ceil(MATRIX_LOCAL_SIZE_Y),
                1,
            ],
            "decode-head-scale",
        )?;

        Ok(())
    }

    #[allow(dead_code)]
    fn run_concat_rows(&self, lhs: &GpuMatrix, rhs: &GpuMatrix) -> Result<GpuMatrix> {
        if lhs.cols != rhs.cols {
            bail!(
                "concat rows incompatível: lhs.cols={} rhs.cols={}",
                lhs.cols,
                rhs.cols
            );
        }

        let rows = lhs.rows + rhs.rows;
        let cols = lhs.cols;
        let output_buffer =
            self.create_temp_buffer_f32(rows * cols, BufferUsage::STORAGE_BUFFER)?;
        let params = [lhs.rows as u32, rhs.rows as u32, cols as u32];
        let params_buffer =
            self.upload_host_storage_buffer(&params, BufferUsage::STORAGE_BUFFER)?;

        let pipeline = self.create_pipeline(CONCAT_ROWS_WGSL, "concat-rows")?;
        let descriptor_set = self.create_descriptor_set(
            &pipeline,
            [
                WriteDescriptorSet::buffer(0, lhs.buffer.clone()),
                WriteDescriptorSet::buffer(1, rhs.buffer.clone()),
                WriteDescriptorSet::buffer(2, output_buffer.clone()),
                WriteDescriptorSet::buffer(3, params_buffer),
            ],
        )?;

        self.execute_compute(
            pipeline,
            descriptor_set,
            [
                (cols as u32).div_ceil(MATRIX_LOCAL_SIZE_X),
                (rows as u32).div_ceil(MATRIX_LOCAL_SIZE_Y),
                1,
            ],
            "concat-rows",
        )?;

        Ok(GpuMatrix {
            rows,
            cols,
            buffer: output_buffer,
        })
    }

    fn run_decode(&self, head: &GpuHeadFeatures, decode: &YoloxDecodeDemo) -> Result<GpuMatrix> {
        let cols = decode.num_classes + 5;
        let s8_rows = head.s8.shape.height * head.s8.shape.width;
        let s16_rows = head.s16.shape.height * head.s16.shape.width;
        let s32_rows = head.s32.shape.height * head.s32.shape.width;
        let rows = s8_rows + s16_rows + s32_rows;
        let output_buffer =
            self.create_temp_buffer_f32(rows * cols, BufferUsage::STORAGE_BUFFER)?;

        self.run_decode_head_scale_into(&head.s8, 8, decode.num_classes, &output_buffer, rows, 0)?;
        self.run_decode_head_scale_into(
            &head.s16,
            16,
            decode.num_classes,
            &output_buffer,
            rows,
            s8_rows,
        )?;
        self.run_decode_head_scale_into(
            &head.s32,
            32,
            decode.num_classes,
            &output_buffer,
            rows,
            s8_rows + s16_rows,
        )?;

        Ok(GpuMatrix {
            rows,
            cols,
            buffer: output_buffer,
        })
    }

    fn run_bottleneck_block(
        &self,
        input: &GpuTensor,
        block: &BottleneckBlock,
    ) -> Result<GpuTensor> {
        let identity = input.clone();
        let conv1 = self.run_baseconv_block(input, &block.conv1)?;
        let mut conv2 = match &block.conv2 {
            ConvBlock::Base(base) => self.run_baseconv_block(&conv1, base)?,
            ConvBlock::Dws(dws) => self.run_dwsconv_block(&conv1, dws)?,
        };

        if block.shortcut {
            conv2 = self.run_add_tensors(&conv2, &identity)?;
        }

        Ok(conv2)
    }

    fn run_bottleneck_prepared(
        &self,
        input: &GpuTensor,
        block: &PreparedBottleneckBlock,
    ) -> Result<GpuTensor> {
        let identity = input.clone();
        let conv1 = self.run_baseconv_prepared(input, &block.conv1)?;
        let mut conv2 = self.run_conv_prepared(&conv1, &block.conv2)?;
        self.recycle_tensor(conv1)?;

        if block.shortcut {
            let added = self.run_add_tensors(&conv2, &identity)?;
            self.recycle_tensor(conv2)?;
            conv2 = added;
        }

        Ok(conv2)
    }

    fn create_pipeline(&self, source: &str, label: &str) -> Result<Arc<ComputePipeline>> {
        if let Some(cached) = self
            .pipeline_cache
            .lock()
            .map_err(|_| anyhow!("pipeline cache lock poisoned"))?
            .get(label)
            .cloned()
        {
            return Ok(cached);
        }

        let shader_module = compile_shader_module(self.device.clone(), source)?;
        let entry_point = shader_module
            .entry_point("main")
            .context("entry point `main` não encontrada no shader")?;
        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&[stage.clone()])
                .into_pipeline_layout_create_info(self.device.clone())
                .with_context(|| format!("falha ao derivar layout do pipeline `{label}`"))?,
        )
        .with_context(|| format!("falha ao criar pipeline layout `{label}`"))?;

        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .with_context(|| format!("falha ao criar pipeline `{label}`"))?;

        if let Ok(mut cache) = self.pipeline_cache.lock() {
            cache.insert(label.to_string(), pipeline.clone());
        }

        Ok(pipeline)
    }

    fn create_descriptor_set<const N: usize>(
        &self,
        pipeline: &Arc<ComputePipeline>,
        writes: [WriteDescriptorSet; N],
    ) -> Result<Arc<DescriptorSet>> {
        let set_layout = pipeline
            .layout()
            .set_layouts()
            .first()
            .cloned()
            .context("o pipeline não expôs descriptor set layout")?;

        DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout,
            writes,
            [],
        )
        .context("falha ao criar descriptor set")
    }

    fn take_submission_future(&self) -> Result<Box<dyn GpuFuture>> {
        let mut guard = self
            .submission_future
            .lock()
            .map_err(|_| anyhow!("submission future lock poisoned"))?;
        let mut future = guard
            .take()
            .unwrap_or_else(|| vk_sync::now(self.device.clone()).boxed());
        future.cleanup_finished();
        Ok(future)
    }

    fn store_submission_future(&self, future: Box<dyn GpuFuture>) -> Result<()> {
        let mut guard = self
            .submission_future
            .lock()
            .map_err(|_| anyhow!("submission future lock poisoned"))?;
        *guard = Some(future);
        Ok(())
    }

    fn reset_submission_future(&self) -> Result<()> {
        self.store_submission_future(vk_sync::now(self.device.clone()).boxed())
    }

    fn finish_pending_work(&self) -> Result<()> {
        let future = self.take_submission_future()?;
        if future.queue().is_none() {
            self.reset_submission_future()?;
            return Ok(());
        }

        let mut future = future
            .then_signal_fence_and_flush()
            .context("falha ao flushar submissões pendentes")?;
        let result = future
            .wait(None)
            .context("falha ao aguardar submissões pendentes");
        future.cleanup_finished();
        self.reset_submission_future()?;
        result
    }

    fn execute_compute(
        &self,
        pipeline: Arc<ComputePipeline>,
        descriptor_set: Arc<DescriptorSet>,
        dispatch: [u32; 3],
        label: &str,
    ) -> Result<()> {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .with_context(|| format!("falha ao criar command buffer builder `{label}`"))?;
        builder
            .bind_pipeline_compute(pipeline.clone())
            .with_context(|| format!("falha ao bindar pipeline `{label}`"))?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .with_context(|| format!("falha ao bindar descriptor set `{label}`"))?;

        unsafe {
            builder
                .dispatch(dispatch)
                .with_context(|| format!("falha ao gravar dispatch `{label}`"))?;
        }

        let command_buffer = builder
            .build()
            .with_context(|| format!("falha ao finalizar command buffer `{label}`"))?;
        let previous = self.take_submission_future()?;
        let future = previous
            .then_execute(self.queue.clone(), command_buffer)
            .with_context(|| format!("falha ao executar `{label}`"))?;
        self.store_submission_future(future.boxed())
    }

    fn create_device_buffer<
        T: vulkano::buffer::BufferContents + Default + Copy + Send + Sync + 'static,
    >(
        &self,
        len: usize,
        usage: BufferUsage,
    ) -> Result<Subbuffer<[T]>> {
        Buffer::new_slice::<T>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: usage | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            len as u64,
        )
        .context("falha ao criar buffer device-local")
    }

    fn create_temp_buffer_f32(&self, len: usize, usage: BufferUsage) -> Result<Subbuffer<[f32]>> {
        let mut pool = self
            .temp_buffer_pool
            .lock()
            .map_err(|_| anyhow!("temp buffer pool lock poisoned"))?;
        if let Some(buffers) = pool.get_mut(&len) {
            if let Some(buffer) = buffers.pop() {
                return Ok(buffer);
            }
        }
        drop(pool);
        self.create_device_buffer::<f32>(len, usage)
    }

    fn recycle_buffer_f32(&self, buffer: Subbuffer<[f32]>) -> Result<()> {
        const MAX_POOL_PER_LEN: usize = 8;

        let len = buffer.len() as usize;
        let mut pool = self
            .temp_buffer_pool
            .lock()
            .map_err(|_| anyhow!("temp buffer pool lock poisoned"))?;
        let entry = pool.entry(len).or_default();
        if entry.len() < MAX_POOL_PER_LEN {
            entry.push(buffer);
        }
        Ok(())
    }

    fn recycle_tensor(&self, tensor: GpuTensor) -> Result<()> {
        self.recycle_buffer_f32(tensor.buffer)
    }

    fn recycle_matrix(&self, matrix: GpuMatrix) -> Result<()> {
        self.recycle_buffer_f32(matrix.buffer)
    }

    fn create_host_readback_buffer<
        T: vulkano::buffer::BufferContents + Default + Copy + Send + Sync + 'static,
    >(
        &self,
        len: usize,
        usage: BufferUsage,
    ) -> Result<Subbuffer<[T]>> {
        let buffer_info = BufferCreateInfo {
            usage,
            ..Default::default()
        };
        let filters = [
            MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            MemoryTypeFilter::HOST_RANDOM_ACCESS,
            MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        ];
        let mut last_error = None;

        for memory_type_filter in filters {
            match Buffer::new_slice::<T>(
                self.memory_allocator.clone(),
                buffer_info.clone(),
                AllocationCreateInfo {
                    memory_type_filter,
                    ..Default::default()
                },
                len as u64,
            ) {
                Ok(buffer) => return Ok(buffer),
                Err(error) => last_error = Some(error),
            }
        }

        Err(last_error.unwrap()).context("falha ao criar buffer de readback")
    }

    fn upload_host_storage_buffer<
        T: vulkano::buffer::BufferContents + Default + Copy + Send + Sync + 'static,
    >(
        &self,
        data: &[T],
        usage: BufferUsage,
    ) -> Result<Subbuffer<[T]>> {
        Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.iter().copied(),
        )
        .context("falha ao criar buffer host-visible")
    }

    fn upload_device_buffer<
        T: vulkano::buffer::BufferContents + Default + Copy + Send + Sync + 'static,
    >(
        &self,
        data: &[T],
        usage: BufferUsage,
    ) -> Result<Subbuffer<[T]>> {
        let staging_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.iter().copied(),
        )
        .context("falha ao criar staging buffer de upload")?;

        let device_buffer = self.create_device_buffer::<T>(data.len(), usage)?;
        self.copy_buffer_with_pending_wait(staging_buffer, device_buffer.clone())?;
        Ok(device_buffer)
    }

    fn copy_buffer_with_pending_wait<T: vulkano::buffer::BufferContents + ?Sized>(
        &self,
        src: Subbuffer<T>,
        dst: Subbuffer<T>,
    ) -> Result<()> {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .context("falha ao criar command buffer de copia")?;
        builder
            .copy_buffer(CopyBufferInfo::buffers(src, dst))
            .context("falha ao gravar comando de copia")?;
        let command_buffer = builder
            .build()
            .context("falha ao finalizar command buffer de copia")?;
        let previous = self.take_submission_future()?;
        let mut future = previous
            .then_execute(self.queue.clone(), command_buffer)
            .context("falha ao executar copia")?
            .then_signal_fence_and_flush()
            .context("falha ao flushar copia")?;
        let result = future.wait(None).context("falha ao aguardar copia");
        future.cleanup_finished();
        self.reset_submission_future()?;
        result
    }

    #[allow(dead_code)]
    fn copy_buffer_and_wait<T: vulkano::buffer::BufferContents + ?Sized>(
        &self,
        src: Subbuffer<T>,
        dst: Subbuffer<T>,
    ) -> Result<()> {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .context("falha ao criar command buffer de cópia")?;
        builder
            .copy_buffer(CopyBufferInfo::buffers(src, dst))
            .context("falha ao gravar comando de cópia")?;
        let command_buffer = builder
            .build()
            .context("falha ao finalizar command buffer de cópia")?;
        command_buffer
            .execute(self.queue.clone())
            .context("falha ao executar cópia")?
            .then_signal_fence_and_flush()
            .context("falha ao flushar cópia")?
            .wait(None)
            .context("falha ao aguardar cópia")
    }
}

fn load_vulkan_library() -> Result<Arc<VulkanLibrary>> {
    VulkanLibrary::new()
        .or_else(|_| unsafe {
            VulkanLibrary::with_loader(DynamicLibraryLoader::new("/opt/homebrew/lib/libvulkan.dylib")?)
        })
        .or_else(|_| unsafe {
            VulkanLibrary::with_loader(DynamicLibraryLoader::new("/opt/homebrew/lib/libMoltenVK.dylib")?)
        })
        .context(
            "falha ao carregar o loader Vulkan; em macOS isso normalmente indica MoltenVK ou o Vulkan SDK ausente",
        )
}

fn select_queue_family(physical_device: &Arc<PhysicalDevice>) -> Option<u32> {
    physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .find_map(|(index, props)| {
            props
                .queue_flags
                .intersects(QueueFlags::COMPUTE)
                .then_some(index as u32)
        })
}

fn compile_shader_module(device: Arc<Device>, source: &str) -> Result<Arc<ShaderModule>> {
    let module = wgsl::parse_str(source).context("falha ao parsear WGSL")?;
    let module_info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .context("falha ao validar WGSL")?;
    let spirv = write_vec(
        &module,
        &module_info,
        &SpvOptions::default(),
        Some(&SpvPipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: "main".into(),
        }),
    )
    .context("falha ao converter WGSL para SPIR-V")?;

    unsafe {
        ShaderModule::new(device, ShaderModuleCreateInfo::new(&spirv))
            .context("falha ao criar ShaderModule")
    }
}
