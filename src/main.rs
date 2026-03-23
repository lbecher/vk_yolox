mod fused_weights;
mod model_bundle;
mod model_plan;
mod tensor_ops;
mod vulkan_conv;
mod yolox_blocks;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use fused_weights::{BatchNorm1d, Conv2dSpec, RawConv2dWeights, fuse_conv2d_bn};
use image::{DynamicImage, ImageBuffer, Rgb};
use model_bundle::{BundleRawWeightPatch, DemoModelBundle};
use model_plan::{ModelPlan, ModelVariant, build_model_plan, bytes_to_mib};
use serde::Serialize;
use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};
use tensor_ops::{
    TensorShape, add_nchw, compare_slices, concat_channels_nchw, focus_nchw, make_demo_tensor,
    maxpool2d_nchw, sigmoid_nchw, silu_nchw, upsample_nearest_nchw,
};
use vulkan_conv::{
    GpuDecodeSession, GpuResidentDecodeSession, run_conv2d_demo, run_demo_backbone,
    run_demo_block, run_demo_bottleneck, run_demo_csp, run_demo_dark5, run_demo_decode,
    run_demo_decode_resident, run_demo_head, run_demo_pafpn, run_demo_stem,
};
use yolox_blocks::{
    BottleneckBlock, CspDarknetDemo, CspStageBlock, Dark5Block, DecodedPredictions, Detection,
    YoloxDecodeDemo, YoloxHeadDemo, YoloxPafpnDemo, YoloxPostprocessDemo,
};

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Planejador e base de runtime para YOLOX em Vulkan 1.0"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    Inspect(InspectArgs),
    DemoConv(DemoConvArgs),
    DemoDepthwise(DemoDepthwiseArgs),
    DemoBlock(DemoBlockArgs),
    DemoStem(DemoStemArgs),
    DemoBottleneck(DemoBottleneckArgs),
    DemoCsp(DemoCspArgs),
    DemoDark5(DemoDark5Args),
    DemoBackbone(DemoBackboneArgs),
    DemoPafpn(DemoPafpnArgs),
    DemoHead(DemoHeadArgs),
    DemoDecode(DemoDecodeArgs),
    DemoDetect(DemoDetectArgs),
    DemoDetectCpu(DemoDetectArgs),
    DemoDetectResident(DemoDetectArgs),
    ExportDemoBundle(ExportDemoBundleArgs),
    InspectBundle(InspectBundleArgs),
    ExportBundleManifest(ExportBundleManifestArgs),
    ExportBurnMappingManifest(ExportBundleManifestArgs),
    ExportBundleWeights(ExportBundleWeightsArgs),
    BuildRawPatchFromBurnManifest(BuildRawPatchFromBurnManifestArgs),
    ApplyBundleWeights(ApplyBundleWeightsArgs),
    ApplyBundleRawWeights(ApplyBundleRawWeightsArgs),
    ExportBundleWeightDir(ExportBundleWeightDirArgs),
    ApplyBundleWeightDir(ApplyBundleWeightDirArgs),
    DemoDetectBundle(DemoDetectBundleArgs),
    InferBundle(InferBundleArgs),
    CompareInferBundle(CompareInferBundleArgs),
    BenchInferBundle(BenchInferBundleArgs),
}

#[derive(Debug, Args, Clone)]
struct InspectArgs {
    #[arg(long, value_enum, default_value_t = ModelVariant::Tiny)]
    model: ModelVariant,

    #[arg(long, default_value_t = 640)]
    input_size: usize,

    #[arg(long, default_value_t = 80)]
    num_classes: usize,

    #[arg(long)]
    json: bool,

    #[arg(long)]
    output: Option<PathBuf>,
}

impl Default for InspectArgs {
    fn default() -> Self {
        Self {
            model: ModelVariant::Tiny,
            input_size: 640,
            num_classes: 80,
            json: false,
            output: None,
        }
    }
}

#[derive(Debug, Args, Clone)]
struct DemoConvArgs {
    #[arg(long, default_value_t = 3)]
    in_channels: usize,

    #[arg(long, default_value_t = 8)]
    out_channels: usize,

    #[arg(long, default_value_t = 32)]
    input_h: usize,

    #[arg(long, default_value_t = 32)]
    input_w: usize,

    #[arg(long, default_value_t = 3)]
    kernel: usize,

    #[arg(long, default_value_t = 1)]
    stride: usize,

    #[arg(long, default_value_t = 1)]
    padding: usize,

    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[derive(Debug, Args, Clone)]
struct DemoDepthwiseArgs {
    #[arg(long, default_value_t = 8)]
    channels: usize,

    #[arg(long, default_value_t = 32)]
    input_h: usize,

    #[arg(long, default_value_t = 32)]
    input_w: usize,

    #[arg(long, default_value_t = 3)]
    kernel: usize,

    #[arg(long, default_value_t = 1)]
    stride: usize,

    #[arg(long, default_value_t = 1)]
    padding: usize,

    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[derive(Debug, Args, Clone)]
struct DemoBlockArgs {
    #[arg(long, default_value_t = 3)]
    in_channels: usize,

    #[arg(long, default_value_t = 8)]
    out_channels: usize,

    #[arg(long, default_value_t = 32)]
    input_h: usize,

    #[arg(long, default_value_t = 32)]
    input_w: usize,

    #[arg(long, default_value_t = 3)]
    kernel: usize,

    #[arg(long, default_value_t = 1)]
    stride: usize,

    #[arg(long, default_value_t = 1)]
    padding: usize,

    #[arg(long, default_value_t = 2)]
    upsample_scale: usize,

    #[arg(long, default_value_t = 6)]
    skip_channels: usize,

    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[derive(Debug, Args, Clone)]
struct DemoStemArgs {
    #[arg(long, default_value_t = 3)]
    in_channels: usize,

    #[arg(long, default_value_t = 16)]
    out_channels: usize,

    #[arg(long, default_value_t = 64)]
    input_h: usize,

    #[arg(long, default_value_t = 64)]
    input_w: usize,

    #[arg(long, default_value_t = 3)]
    kernel: usize,

    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[derive(Debug, Args, Clone)]
struct DemoBottleneckArgs {
    #[arg(long, default_value_t = 16)]
    channels: usize,

    #[arg(long, default_value_t = 32)]
    input_h: usize,

    #[arg(long, default_value_t = 32)]
    input_w: usize,

    #[arg(long)]
    no_shortcut: bool,

    #[arg(long)]
    depthwise: bool,

    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[derive(Debug, Args, Clone)]
struct DemoCspArgs {
    #[arg(long, default_value_t = 16)]
    in_channels: usize,

    #[arg(long, default_value_t = 32)]
    out_channels: usize,

    #[arg(long, default_value_t = 64)]
    input_h: usize,

    #[arg(long, default_value_t = 64)]
    input_w: usize,

    #[arg(long, default_value_t = 1)]
    depth: usize,

    #[arg(long)]
    depthwise: bool,

    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[derive(Debug, Args, Clone)]
struct DemoDark5Args {
    #[arg(long, default_value_t = 64)]
    in_channels: usize,

    #[arg(long, default_value_t = 128)]
    out_channels: usize,

    #[arg(long, default_value_t = 64)]
    input_h: usize,

    #[arg(long, default_value_t = 64)]
    input_w: usize,

    #[arg(long, default_value_t = 1)]
    depth: usize,

    #[arg(long)]
    depthwise: bool,

    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[derive(Debug, Args, Clone)]
struct DemoBackboneArgs {
    #[arg(long, default_value_t = 32)]
    base_channels: usize,

    #[arg(long, default_value_t = 1)]
    base_depth: usize,

    #[arg(long, default_value_t = 256)]
    input_h: usize,

    #[arg(long, default_value_t = 256)]
    input_w: usize,

    #[arg(long)]
    depthwise: bool,

    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[derive(Debug, Args, Clone)]
struct DemoPafpnArgs {
    #[arg(long, default_value_t = 32)]
    base_channels: usize,

    #[arg(long, default_value_t = 1)]
    base_depth: usize,

    #[arg(long, default_value_t = 256)]
    input_h: usize,

    #[arg(long, default_value_t = 256)]
    input_w: usize,

    #[arg(long)]
    depthwise: bool,

    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[derive(Debug, Args, Clone)]
struct DemoHeadArgs {
    #[arg(long, default_value_t = 32)]
    base_channels: usize,

    #[arg(long, default_value_t = 1)]
    base_depth: usize,

    #[arg(long, default_value_t = 80)]
    num_classes: usize,

    #[arg(long, default_value_t = 256)]
    input_h: usize,

    #[arg(long, default_value_t = 256)]
    input_w: usize,

    #[arg(long)]
    depthwise: bool,

    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[derive(Debug, Args, Clone)]
struct DemoDecodeArgs {
    #[arg(long, default_value_t = 32)]
    base_channels: usize,

    #[arg(long, default_value_t = 1)]
    base_depth: usize,

    #[arg(long, default_value_t = 80)]
    num_classes: usize,

    #[arg(long, default_value_t = 256)]
    input_h: usize,

    #[arg(long, default_value_t = 256)]
    input_w: usize,

    #[arg(long)]
    depthwise: bool,

    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[derive(Debug, Args, Clone)]
struct DemoDetectArgs {
    #[arg(long, default_value_t = 32)]
    base_channels: usize,

    #[arg(long, default_value_t = 1)]
    base_depth: usize,

    #[arg(long, default_value_t = 80)]
    num_classes: usize,

    #[arg(long, default_value_t = 256)]
    input_h: usize,

    #[arg(long, default_value_t = 256)]
    input_w: usize,

    #[arg(long, default_value_t = 0.25)]
    confidence_threshold: f32,

    #[arg(long, default_value_t = 0.65)]
    nms_threshold: f32,

    #[arg(long, default_value_t = 20)]
    max_detections: usize,

    #[arg(long)]
    depthwise: bool,

    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[derive(Debug, Args, Clone)]
struct ExportDemoBundleArgs {
    #[arg(long, default_value_t = 32)]
    base_channels: usize,

    #[arg(long, default_value_t = 1)]
    base_depth: usize,

    #[arg(long, default_value_t = 80)]
    num_classes: usize,

    #[arg(long)]
    depthwise: bool,

    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args, Clone)]
struct InspectBundleArgs {
    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    json: bool,

    #[arg(long)]
    layers: bool,
}

#[derive(Debug, Args, Clone)]
struct ExportBundleManifestArgs {
    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args, Clone)]
struct ExportBundleWeightsArgs {
    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args, Clone)]
struct ApplyBundleWeightsArgs {
    #[arg(long)]
    bundle: PathBuf,

    #[arg(long)]
    weights: PathBuf,

    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args, Clone)]
struct ApplyBundleRawWeightsArgs {
    #[arg(long)]
    bundle: PathBuf,

    #[arg(long)]
    raw_weights: PathBuf,

    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args, Clone)]
struct BuildRawPatchFromBurnManifestArgs {
    #[arg(long)]
    bundle: PathBuf,

    #[arg(long)]
    tensor_manifest: PathBuf,

    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args, Clone)]
struct ExportBundleWeightDirArgs {
    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    output_dir: PathBuf,
}

#[derive(Debug, Args, Clone)]
struct ApplyBundleWeightDirArgs {
    #[arg(long)]
    bundle: PathBuf,

    #[arg(long)]
    weights_dir: PathBuf,

    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args, Clone)]
struct DemoDetectBundleArgs {
    #[arg(long)]
    input: PathBuf,

    #[arg(long, default_value_t = 256)]
    input_h: usize,

    #[arg(long, default_value_t = 256)]
    input_w: usize,

    #[arg(long, default_value_t = 0.25)]
    confidence_threshold: f32,

    #[arg(long, default_value_t = 0.65)]
    nms_threshold: f32,

    #[arg(long, default_value_t = 20)]
    max_detections: usize,

    #[arg(long)]
    cpu_only: bool,

    #[arg(long)]
    resident_weights: bool,

    #[arg(long, default_value_t = 0)]
    device_index: usize,
}

#[derive(Debug, Args, Clone)]
struct InferBundleArgs {
    #[arg(long)]
    bundle: PathBuf,

    #[arg(long)]
    image: PathBuf,

    #[arg(long, default_value_t = 640)]
    input_h: usize,

    #[arg(long, default_value_t = 640)]
    input_w: usize,

    #[arg(long, default_value_t = 0.25)]
    confidence_threshold: f32,

    #[arg(long, default_value_t = 0.65)]
    nms_threshold: f32,

    #[arg(long, default_value_t = 20)]
    max_detections: usize,

    #[arg(long)]
    cpu_only: bool,

    #[arg(long)]
    resident_weights: bool,

    #[arg(long, default_value_t = 0)]
    device_index: usize,

    #[arg(long)]
    output_json: Option<PathBuf>,

    #[arg(long)]
    output_image: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct CompareInferBundleArgs {
    #[arg(long)]
    bundle: PathBuf,

    #[arg(long)]
    image: PathBuf,

    #[arg(long, default_value_t = 640)]
    input_h: usize,

    #[arg(long, default_value_t = 640)]
    input_w: usize,

    #[arg(long, default_value_t = 0.25)]
    confidence_threshold: f32,

    #[arg(long, default_value_t = 0.65)]
    nms_threshold: f32,

    #[arg(long, default_value_t = 20)]
    max_detections: usize,

    #[arg(long)]
    resident_weights: bool,

    #[arg(long, default_value_t = 0)]
    device_index: usize,

    #[arg(long)]
    output_json: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct BenchInferBundleArgs {
    #[arg(long)]
    bundle: PathBuf,

    #[arg(long)]
    image: PathBuf,

    #[arg(long, default_value_t = 640)]
    input_h: usize,

    #[arg(long, default_value_t = 640)]
    input_w: usize,

    #[arg(long, default_value_t = 0.25)]
    confidence_threshold: f32,

    #[arg(long, default_value_t = 0.65)]
    nms_threshold: f32,

    #[arg(long, default_value_t = 20)]
    max_detections: usize,

    #[arg(long)]
    resident_weights: bool,

    #[arg(long, default_value_t = 0)]
    device_index: usize,

    #[arg(long, default_value_t = 1)]
    warmup_iterations: usize,

    #[arg(long, default_value_t = 10)]
    iterations: usize,

    #[arg(long)]
    cpu_only: bool,

    #[arg(long)]
    gpu_only: bool,

    #[arg(long)]
    output_json: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct SerializableDetection {
    class_id: usize,
    score: f32,
    objectness: f32,
    class_confidence: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

#[derive(Debug, Serialize)]
struct InferenceReport {
    image: String,
    original_width: u32,
    original_height: u32,
    input_h: usize,
    input_w: usize,
    cpu_only: bool,
    resident_weights: bool,
    detections: Vec<SerializableDetection>,
}

#[derive(Debug, Serialize)]
struct DetectionComparison {
    index: usize,
    class_id_cpu: usize,
    class_id_gpu: usize,
    score_abs_diff: f32,
    x0_abs_diff: f32,
    y0_abs_diff: f32,
    x1_abs_diff: f32,
    y1_abs_diff: f32,
}

#[derive(Debug, Serialize)]
struct CompareInferenceReport {
    image: String,
    input_h: usize,
    input_w: usize,
    resident_weights: bool,
    decoded_elements: usize,
    decoded_max_abs_diff: f32,
    decoded_mean_abs_diff: f64,
    cpu_detection_count: usize,
    gpu_detection_count: usize,
    compared_detections: usize,
    detection_comparisons: Vec<DetectionComparison>,
}

#[derive(Debug, Serialize)]
struct BenchPathReport {
    label: String,
    iterations: usize,
    warmup_iterations: usize,
    resident_weights: bool,
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
    median_ms: f64,
    fps: f64,
    detection_count: usize,
}

#[derive(Debug, Serialize)]
struct BenchInferenceReport {
    image: String,
    input_h: usize,
    input_w: usize,
    bundle_load_ms: f64,
    input_prepare_ms: f64,
    cpu: Option<BenchPathReport>,
    gpu: Option<BenchPathReport>,
}

fn main() -> Result<()> {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(run_main)
        .context("falha ao criar thread principal com stack ampliada")?
        .join()
        .map_err(|_| anyhow::anyhow!("thread principal terminou com panic"))?
}

fn run_main() -> Result<()> {
    let cli = Cli::parse();

    match cli
        .command
        .unwrap_or(Command::Inspect(InspectArgs::default()))
    {
        Command::Inspect(args) => inspect(args),
        Command::DemoConv(args) => demo_conv(args),
        Command::DemoDepthwise(args) => demo_depthwise(args),
        Command::DemoBlock(args) => demo_block(args),
        Command::DemoStem(args) => demo_stem(args),
        Command::DemoBottleneck(args) => demo_bottleneck(args),
        Command::DemoCsp(args) => demo_csp(args),
        Command::DemoDark5(args) => demo_dark5(args),
        Command::DemoBackbone(args) => demo_backbone(args),
        Command::DemoPafpn(args) => demo_pafpn(args),
        Command::DemoHead(args) => demo_head(args),
        Command::DemoDecode(args) => demo_decode(args),
        Command::DemoDetect(args) => demo_detect(args),
        Command::DemoDetectCpu(args) => demo_detect_cpu(args),
        Command::DemoDetectResident(args) => demo_detect_resident(args),
        Command::ExportDemoBundle(args) => export_demo_bundle(args),
        Command::InspectBundle(args) => inspect_bundle(args),
        Command::ExportBundleManifest(args) => export_bundle_manifest(args),
        Command::ExportBurnMappingManifest(args) => export_burn_mapping_manifest(args),
        Command::ExportBundleWeights(args) => export_bundle_weights(args),
        Command::BuildRawPatchFromBurnManifest(args) => build_raw_patch_from_burn_manifest(args),
        Command::ApplyBundleWeights(args) => apply_bundle_weights(args),
        Command::ApplyBundleRawWeights(args) => apply_bundle_raw_weights(args),
        Command::ExportBundleWeightDir(args) => export_bundle_weight_dir(args),
        Command::ApplyBundleWeightDir(args) => apply_bundle_weight_dir(args),
        Command::DemoDetectBundle(args) => demo_detect_bundle(args),
        Command::InferBundle(args) => infer_bundle(args),
        Command::CompareInferBundle(args) => compare_infer_bundle(args),
        Command::BenchInferBundle(args) => bench_infer_bundle(args),
    }
}

fn inspect(args: InspectArgs) -> Result<()> {
    if args.input_size == 0 || !args.input_size.is_multiple_of(32) {
        bail!("input_size deve ser múltiplo de 32 para o grafo YOLOX");
    }

    let plan = build_model_plan(args.model, args.input_size, args.num_classes);
    let serialized = serde_json::to_string_pretty(&plan).context("falha ao serializar o plano")?;

    if let Some(path) = args.output {
        fs::write(&path, &serialized)
            .with_context(|| format!("falha ao escrever {}", path.display()))?;
        println!("plano salvo em {}", path.display());
    }

    if args.json {
        println!("{serialized}");
    } else {
        print_human_summary(&plan);
    }

    Ok(())
}

fn validate_detect_args(
    name: &str,
    input_h: usize,
    input_w: usize,
    num_classes: usize,
    max_detections: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> Result<()> {
    if num_classes == 0 || max_detections == 0 {
        bail!("num_classes e max_detections devem ser maiores que zero");
    }
    if input_h == 0 || input_w == 0 {
        bail!("input_h e input_w devem ser maiores que zero");
    }
    if !input_h.is_multiple_of(32) || !input_w.is_multiple_of(32) {
        bail!("{name} exige input_h e input_w múltiplos de 32");
    }
    if !(0.0..=1.0).contains(&confidence_threshold) || !(0.0..=1.0).contains(&nms_threshold) {
        bail!("confidence_threshold e nms_threshold devem estar em [0, 1]");
    }

    Ok(())
}

fn run_cpu_detect_demo(
    input: &[f32],
    input_shape: TensorShape,
    backbone: &CspDarknetDemo,
    pafpn: &YoloxPafpnDemo,
    head: &YoloxHeadDemo,
    decode: &YoloxDecodeDemo,
    postprocess: &YoloxPostprocessDemo,
) -> Result<(DecodedPredictions, Vec<Detection>)> {
    let backbone_cpu = backbone.forward_cpu(input, input_shape)?;
    let pafpn_cpu = pafpn.forward_cpu(&backbone_cpu)?;
    let head_cpu = head.forward_cpu(&pafpn_cpu)?;
    let decoded = decode.forward_cpu(&head_cpu)?;
    let detections = postprocess.forward_cpu(&decoded)?;
    Ok((decoded, detections))
}

fn print_detections(
    label: &str,
    input_shape: TensorShape,
    decoded: &DecodedPredictions,
    detections: &[Detection],
    confidence_threshold: f32,
    nms_threshold: f32,
) {
    println!(
        "{label}: input={} decoded=[{}x{}] detections={} conf_thres={:.2} nms_thres={:.2}",
        input_shape.display_nchw(),
        decoded.rows,
        decoded.cols,
        detections.len(),
        confidence_threshold,
        nms_threshold,
    );

    for (index, detection) in detections.iter().take(5).enumerate() {
        println!(
            "det {:02}: class={} score={:.5} obj={:.5} cls={:.5} box=[{:.2}, {:.2}, {:.2}, {:.2}]",
            index,
            detection.class_id,
            detection.score,
            detection.objectness,
            detection.class_confidence,
            detection.x0,
            detection.y0,
            detection.x1,
            detection.y1,
        );
    }
}

fn load_image_as_nchw_f32(
    path: &std::path::Path,
    input_h: usize,
    input_w: usize,
) -> Result<(DynamicImage, TensorShape, Vec<f32>)> {
    let image =
        image::open(path).with_context(|| format!("falha ao abrir imagem {}", path.display()))?;
    let resized = image.resize_exact(
        input_w as u32,
        input_h as u32,
        image::imageops::FilterType::Triangle,
    );
    let rgb = resized.to_rgb8().into_raw();
    let shape = TensorShape::new(3, input_h, input_w);
    let plane = input_h * input_w;
    let mut output = vec![0.0f32; shape.len()];

    for y in 0..input_h {
        for x in 0..input_w {
            let src = (y * input_w + x) * 3;
            output[y * input_w + x] = rgb[src] as f32;
            output[plane + y * input_w + x] = rgb[src + 1] as f32;
            output[plane * 2 + y * input_w + x] = rgb[src + 2] as f32;
        }
    }

    Ok((image, shape, output))
}

fn scale_detection_to_original(
    detection: &Detection,
    original_width: u32,
    original_height: u32,
    input_w: usize,
    input_h: usize,
) -> SerializableDetection {
    let scale_x = original_width as f32 / input_w as f32;
    let scale_y = original_height as f32 / input_h as f32;
    SerializableDetection {
        class_id: detection.class_id,
        score: detection.score,
        objectness: detection.objectness,
        class_confidence: detection.class_confidence,
        x0: (detection.x0 * scale_x).clamp(0.0, original_width as f32),
        y0: (detection.y0 * scale_y).clamp(0.0, original_height as f32),
        x1: (detection.x1 * scale_x).clamp(0.0, original_width as f32),
        y1: (detection.y1 * scale_y).clamp(0.0, original_height as f32),
    }
}

fn draw_detection_rectangles(
    image: DynamicImage,
    detections: &[SerializableDetection],
) -> DynamicImage {
    fn draw_rect(
        image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        color: [u8; 3],
    ) {
        if x0 >= image.width() || y0 >= image.height() || x1 >= image.width() || y1 >= image.height()
        {
            return;
        }
        for x in x0..=x1 {
            *image.get_pixel_mut(x, y0) = Rgb(color);
            *image.get_pixel_mut(x, y1) = Rgb(color);
        }
        for y in y0..=y1 {
            *image.get_pixel_mut(x0, y) = Rgb(color);
            *image.get_pixel_mut(x1, y) = Rgb(color);
        }
    }

    let mut output = image.to_rgb8();
    for detection in detections {
        let x0 = detection.x0.floor().max(0.0) as u32;
        let y0 = detection.y0.floor().max(0.0) as u32;
        let x1 = detection.x1.ceil().max(0.0) as u32;
        let y1 = detection.y1.ceil().max(0.0) as u32;
        if x1 > x0 && y1 > y0 {
            draw_rect(&mut output, x0, y0, x1, y1, [239, 62, 5]);
        }
    }
    DynamicImage::ImageRgb8(output)
}

fn run_bundle_inference(
    bundle: &DemoModelBundle,
    input: &[f32],
    input_shape: TensorShape,
    confidence_threshold: f32,
    nms_threshold: f32,
    max_detections: usize,
    cpu_only: bool,
    resident_weights: bool,
    device_index: usize,
) -> Result<(DecodedPredictions, Vec<Detection>)> {
    let (decode, postprocess) = make_bundle_postprocess(
        bundle,
        confidence_threshold,
        nms_threshold,
        max_detections,
    );

    if cpu_only {
        return run_cpu_detect_demo(
            input,
            input_shape,
            &bundle.backbone,
            &bundle.pafpn,
            &bundle.head,
            &decode,
            &postprocess,
        );
    }

    let decoded = if resident_weights {
        run_demo_decode_resident(
            input,
            input_shape,
            &bundle.backbone,
            &bundle.pafpn,
            &bundle.head,
            &decode,
            device_index,
        )?
    } else {
        run_demo_decode(
            input,
            input_shape,
            &bundle.backbone,
            &bundle.pafpn,
            &bundle.head,
            &decode,
            device_index,
        )?
    };
    let detections = postprocess.forward_cpu(&decoded)?;
    Ok((decoded, detections))
}

fn make_bundle_postprocess(
    bundle: &DemoModelBundle,
    confidence_threshold: f32,
    nms_threshold: f32,
    max_detections: usize,
) -> (YoloxDecodeDemo, YoloxPostprocessDemo) {
    let decode = YoloxDecodeDemo::new(bundle.meta.num_classes);
    let postprocess = YoloxPostprocessDemo::new(
        bundle.meta.num_classes,
        confidence_threshold,
        nms_threshold,
        max_detections,
    );
    (decode, postprocess)
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn summarize_bench_samples(
    label: &str,
    samples: &[Duration],
    warmup_iterations: usize,
    detection_count: usize,
    resident_weights: bool,
) -> Result<BenchPathReport> {
    if samples.is_empty() {
        bail!("benchmark `{label}` sem amostras");
    }

    let mut ms = samples.iter().map(|item| duration_ms(*item)).collect::<Vec<_>>();
    ms.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));

    let total_ms = ms.iter().sum::<f64>();
    let mean_ms = total_ms / ms.len() as f64;
    let median_ms = if ms.len().is_multiple_of(2) {
        let rhs = ms.len() / 2;
        let lhs = rhs - 1;
        (ms[lhs] + ms[rhs]) * 0.5
    } else {
        ms[ms.len() / 2]
    };

    Ok(BenchPathReport {
        label: label.to_string(),
        iterations: samples.len(),
        warmup_iterations,
        resident_weights,
        mean_ms,
        min_ms: *ms.first().unwrap_or(&0.0),
        max_ms: *ms.last().unwrap_or(&0.0),
        median_ms,
        fps: if mean_ms > 0.0 { 1000.0 / mean_ms } else { 0.0 },
        detection_count,
    })
}

fn bench_inference_path(
    label: &str,
    bundle: &DemoModelBundle,
    input: &[f32],
    input_shape: TensorShape,
    confidence_threshold: f32,
    nms_threshold: f32,
    max_detections: usize,
    cpu_only: bool,
    resident_weights: bool,
    device_index: usize,
    warmup_iterations: usize,
    iterations: usize,
) -> Result<BenchPathReport> {
    let mut last_detection_count = 0usize;
    let mut samples = Vec::with_capacity(iterations);
    let (decode, postprocess) = make_bundle_postprocess(
        bundle,
        confidence_threshold,
        nms_threshold,
        max_detections,
    );

    if cpu_only {
        for _ in 0..warmup_iterations {
            let (_, detections) = run_cpu_detect_demo(
                input,
                input_shape,
                &bundle.backbone,
                &bundle.pafpn,
                &bundle.head,
                &decode,
                &postprocess,
            )?;
            last_detection_count = detections.len();
        }

        for _ in 0..iterations {
            let started = Instant::now();
            let (_, detections) = run_cpu_detect_demo(
                input,
                input_shape,
                &bundle.backbone,
                &bundle.pafpn,
                &bundle.head,
                &decode,
                &postprocess,
            )?;
            samples.push(started.elapsed());
            last_detection_count = detections.len();
        }
    } else if resident_weights {
        let session = GpuResidentDecodeSession::new(
            &bundle.backbone,
            &bundle.pafpn,
            &bundle.head,
            device_index,
        )?;

        for _ in 0..warmup_iterations {
            let decoded = session.run_decode(input, input_shape, &decode)?;
            let detections = postprocess.forward_cpu(&decoded)?;
            last_detection_count = detections.len();
        }

        for _ in 0..iterations {
            let started = Instant::now();
            let decoded = session.run_decode(input, input_shape, &decode)?;
            let detections = postprocess.forward_cpu(&decoded)?;
            samples.push(started.elapsed());
            last_detection_count = detections.len();
        }
    } else {
        let session = GpuDecodeSession::new(device_index)?;

        for _ in 0..warmup_iterations {
            let decoded = session.run_decode(
                input,
                input_shape,
                &bundle.backbone,
                &bundle.pafpn,
                &bundle.head,
                &decode,
            )?;
            let detections = postprocess.forward_cpu(&decoded)?;
            last_detection_count = detections.len();
        }

        for _ in 0..iterations {
            let started = Instant::now();
            let decoded = session.run_decode(
                input,
                input_shape,
                &bundle.backbone,
                &bundle.pafpn,
                &bundle.head,
                &decode,
            )?;
            let detections = postprocess.forward_cpu(&decoded)?;
            samples.push(started.elapsed());
            last_detection_count = detections.len();
        }
    }

    summarize_bench_samples(
        label,
        &samples,
        warmup_iterations,
        last_detection_count,
        resident_weights,
    )
}

fn print_human_summary(plan: &ModelPlan) {
    println!(
        "modelo={} input={} num_classes={} depth={:.2} width={:.3} depthwise={}",
        plan.model.variant.as_str(),
        plan.model.input_shape.display(),
        plan.model.num_classes,
        plan.model.depth_multiplier,
        plan.model.width_multiplier,
        plan.model.depthwise
    );
    println!(
        "pesos(raw)={:.2} MiB pesos(fused)={:.2} MiB pico_ativacoes={:.2} MiB maior_tensor={} ({:.2} MiB)",
        bytes_to_mib(plan.parameters.raw_bytes),
        bytes_to_mib(plan.parameters.fused_bytes),
        bytes_to_mib(plan.memory.peak_live_bytes),
        plan.memory.max_single_tensor_name,
        bytes_to_mib(plan.memory.max_single_tensor_bytes),
    );
    println!(
        "pico em={} tensores_vivos={}",
        plan.memory.peak_at,
        plan.memory.live_tensors_at_peak.join(", "),
    );

    println!();
    println!("primitivas Vulkan 1.0 necessárias:");
    for primitive in &plan.primitives {
        println!(
            "- {} x{}: {}",
            primitive.primitive, primitive.count, primitive.notes
        );
    }

    println!();
    println!("plano de execução:");
    for node in &plan.execution {
        println!(
            "- {:>2} {} {} -> {} raw={:.2} MiB fused={:.2} MiB",
            node.id,
            node.name,
            node.output_shape.display(),
            node.op,
            bytes_to_mib(node.raw_parameter_bytes),
            bytes_to_mib(node.fused_parameter_bytes),
        );
    }

    println!();
    println!("notas:");
    for note in &plan.notes {
        println!("- {note}");
    }
}

fn demo_conv(args: DemoConvArgs) -> Result<()> {
    if args.in_channels == 0 || args.out_channels == 0 {
        bail!("in_channels e out_channels devem ser maiores que zero");
    }
    if args.input_h == 0 || args.input_w == 0 {
        bail!("input_h e input_w devem ser maiores que zero");
    }
    if args.kernel == 0 {
        bail!("kernel deve ser maior que zero");
    }
    if args.stride == 0 {
        bail!("stride deve ser maior que zero");
    }

    let spec = Conv2dSpec::new(
        args.in_channels,
        args.out_channels,
        args.kernel,
        args.kernel,
        args.stride,
        args.stride,
        args.padding,
        args.padding,
    );
    let output_hw = spec
        .output_hw(args.input_h, args.input_w)
        .context("configuração inválida para conv2d")?;

    let input = make_demo_tensor(
        TensorShape::new(args.in_channels, args.input_h, args.input_w),
        0,
    );
    let raw = RawConv2dWeights::demo(&spec);
    let bn = BatchNorm1d::demo(args.out_channels);
    let fused = fuse_conv2d_bn(&raw, &bn)?;
    let cpu_output = fused.convolve_nchw_f32(&input, args.input_h, args.input_w)?;
    let gpu_output = run_conv2d_demo(
        &input,
        args.input_h,
        args.input_w,
        &fused,
        args.device_index,
    )?;

    if cpu_output.len() != gpu_output.len() {
        bail!(
            "saídas CPU/GPU divergiram em tamanho: {} vs {}",
            cpu_output.len(),
            gpu_output.len()
        );
    }
    let diff = compare_slices(&cpu_output, &gpu_output)?;

    println!(
        "demo conv2d fused: input=[1x{}x{}x{}] output=[1x{}x{}x{}] kernel={} stride={} pad={}",
        args.in_channels,
        args.input_h,
        args.input_w,
        args.out_channels,
        output_hw.0,
        output_hw.1,
        args.kernel,
        args.stride,
        args.padding
    );
    println!(
        "pesos fused: {:.3} KiB",
        (fused.weights.len() + fused.bias.len()) as f64 * std::mem::size_of::<f32>() as f64
            / 1024.0
    );
    println!(
        "comparação CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu_output.len(),
        diff.max_abs_diff,
        diff.mean_abs_diff,
    );
    println!(
        "amostra saída[0..8]={:?}",
        &gpu_output[..gpu_output.len().min(8)]
    );

    Ok(())
}

fn demo_depthwise(args: DemoDepthwiseArgs) -> Result<()> {
    if args.channels == 0 {
        bail!("channels deve ser maior que zero");
    }
    if args.input_h == 0 || args.input_w == 0 {
        bail!("input_h e input_w devem ser maiores que zero");
    }
    if args.kernel == 0 || args.stride == 0 {
        bail!("kernel e stride devem ser maiores que zero");
    }

    let spec = Conv2dSpec::new_grouped(
        args.channels,
        args.channels,
        args.channels,
        args.kernel,
        args.kernel,
        args.stride,
        args.stride,
        args.padding,
        args.padding,
    );
    let output_hw = spec
        .output_hw(args.input_h, args.input_w)
        .context("configuração inválida para depthwise-conv2d")?;
    let input = make_demo_tensor(
        TensorShape::new(args.channels, args.input_h, args.input_w),
        7,
    );
    let raw = RawConv2dWeights::demo(&spec);
    let bn = BatchNorm1d::demo(args.channels);
    let fused = fuse_conv2d_bn(&raw, &bn)?;
    let cpu_output = fused.convolve_nchw_f32(&input, args.input_h, args.input_w)?;
    let gpu_output = run_conv2d_demo(
        &input,
        args.input_h,
        args.input_w,
        &fused,
        args.device_index,
    )?;

    if cpu_output.len() != gpu_output.len() {
        bail!(
            "saídas CPU/GPU divergiram em tamanho: {} vs {}",
            cpu_output.len(),
            gpu_output.len()
        );
    }
    let diff = compare_slices(&cpu_output, &gpu_output)?;

    println!(
        "demo depthwise-conv2d: input=[1x{}x{}x{}] output=[1x{}x{}x{}] kernel={} stride={} pad={}",
        args.channels,
        args.input_h,
        args.input_w,
        args.channels,
        output_hw.0,
        output_hw.1,
        args.kernel,
        args.stride,
        args.padding
    );
    println!(
        "pesos fused depthwise: {:.3} KiB",
        (fused.weights.len() + fused.bias.len()) as f64 * std::mem::size_of::<f32>() as f64
            / 1024.0
    );
    println!(
        "comparação CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu_output.len(),
        diff.max_abs_diff,
        diff.mean_abs_diff,
    );
    println!(
        "amostra saída[0..8]={:?}",
        &gpu_output[..gpu_output.len().min(8)]
    );

    Ok(())
}

fn demo_block(args: DemoBlockArgs) -> Result<()> {
    if args.in_channels == 0 || args.out_channels == 0 || args.skip_channels == 0 {
        bail!("in_channels, out_channels e skip_channels devem ser maiores que zero");
    }
    if args.input_h == 0 || args.input_w == 0 {
        bail!("input_h e input_w devem ser maiores que zero");
    }
    if args.kernel == 0 || args.stride == 0 || args.upsample_scale == 0 {
        bail!("kernel, stride e upsample_scale devem ser maiores que zero");
    }

    let spec = Conv2dSpec::new(
        args.in_channels,
        args.out_channels,
        args.kernel,
        args.kernel,
        args.stride,
        args.stride,
        args.padding,
        args.padding,
    );
    let input_shape = TensorShape::new(args.in_channels, args.input_h, args.input_w);
    let input = make_demo_tensor(input_shape, 0);

    let raw = RawConv2dWeights::demo(&spec);
    let bn = BatchNorm1d::demo(args.out_channels);
    let fused = fuse_conv2d_bn(&raw, &bn)?;

    let conv_cpu = fused.convolve_nchw_f32(&input, args.input_h, args.input_w)?;
    let conv_hw = spec
        .output_hw(args.input_h, args.input_w)
        .context("configuração inválida para conv2d no bloco")?;
    let conv_shape = TensorShape::new(args.out_channels, conv_hw.0, conv_hw.1);
    let silu_cpu = silu_nchw(&conv_cpu);
    let (upsampled_shape, upsampled_cpu) =
        upsample_nearest_nchw(&silu_cpu, conv_shape, args.upsample_scale)?;

    let skip_shape = TensorShape::new(
        args.skip_channels,
        upsampled_shape.height,
        upsampled_shape.width,
    );
    let skip = make_demo_tensor(skip_shape, 3);
    let (cpu_shape, cpu_output) =
        concat_channels_nchw(&upsampled_cpu, upsampled_shape, &skip, skip_shape)?;

    let (gpu_shape, gpu_output) = run_demo_block(
        &input,
        input_shape,
        &fused,
        &skip,
        skip_shape,
        args.upsample_scale,
        args.device_index,
    )?;

    if cpu_shape != gpu_shape {
        bail!(
            "shape final divergente: cpu={} gpu={}",
            cpu_shape.display_nchw(),
            gpu_shape.display_nchw()
        );
    }

    let diff = compare_slices(&cpu_output, &gpu_output)?;
    println!(
        "demo block: input={} conv={} upsample={} skip={} output={}",
        input_shape.display_nchw(),
        conv_shape.display_nchw(),
        upsampled_shape.display_nchw(),
        skip_shape.display_nchw(),
        cpu_shape.display_nchw(),
    );
    println!(
        "pipeline GPU: conv2d -> silu -> upsample-nearest(x{}) -> concat",
        args.upsample_scale
    );
    println!(
        "comparação CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu_output.len(),
        diff.max_abs_diff,
        diff.mean_abs_diff,
    );
    println!(
        "amostra saída[0..8]={:?}",
        &gpu_output[..gpu_output.len().min(8)]
    );

    Ok(())
}

fn demo_stem(args: DemoStemArgs) -> Result<()> {
    if args.in_channels == 0 || args.out_channels == 0 {
        bail!("in_channels e out_channels devem ser maiores que zero");
    }
    if args.input_h == 0 || args.input_w == 0 || args.input_h % 2 != 0 || args.input_w % 2 != 0 {
        bail!("demo-stem exige input_h e input_w pares e maiores que zero");
    }
    if args.kernel == 0 {
        bail!("kernel deve ser maior que zero");
    }

    let input_shape = TensorShape::new(args.in_channels, args.input_h, args.input_w);
    let input = make_demo_tensor(input_shape, 11);
    let (focus_shape, focused_cpu) = focus_nchw(&input, input_shape)?;

    let spec = Conv2dSpec::new(
        focus_shape.channels,
        args.out_channels,
        args.kernel,
        args.kernel,
        1,
        1,
        1,
        1,
    );
    let raw = RawConv2dWeights::demo(&spec);
    let bn = BatchNorm1d::demo(args.out_channels);
    let fused = fuse_conv2d_bn(&raw, &bn)?;

    let conv_cpu = fused.convolve_nchw_f32(&focused_cpu, focus_shape.height, focus_shape.width)?;
    let conv_shape = TensorShape::new(args.out_channels, focus_shape.height, focus_shape.width);
    let silu_cpu = silu_nchw(&conv_cpu);
    let (pooled_shape, pooled_cpu) = maxpool2d_nchw(&silu_cpu, conv_shape, 3, 1, 1)?;
    let added_cpu = add_nchw(&pooled_cpu, &silu_cpu)?;
    let output_cpu = sigmoid_nchw(&added_cpu);

    let (gpu_shape, gpu_output) = run_demo_stem(&input, input_shape, &fused, args.device_index)?;

    if gpu_shape != pooled_shape {
        bail!(
            "shape final divergente: cpu={} gpu={}",
            pooled_shape.display_nchw(),
            gpu_shape.display_nchw()
        );
    }

    let diff = compare_slices(&output_cpu, &gpu_output)?;
    println!(
        "demo stem: input={} focus={} conv={} output={}",
        input_shape.display_nchw(),
        focus_shape.display_nchw(),
        conv_shape.display_nchw(),
        pooled_shape.display_nchw(),
    );
    println!("pipeline GPU: focus -> conv2d -> silu -> maxpool2d -> add -> sigmoid");
    println!(
        "comparação CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        output_cpu.len(),
        diff.max_abs_diff,
        diff.mean_abs_diff,
    );
    println!(
        "amostra saída[0..8]={:?}",
        &gpu_output[..gpu_output.len().min(8)]
    );

    Ok(())
}

fn demo_bottleneck(args: DemoBottleneckArgs) -> Result<()> {
    if args.channels == 0 {
        bail!("channels deve ser maior que zero");
    }
    if args.input_h == 0 || args.input_w == 0 {
        bail!("input_h e input_w devem ser maiores que zero");
    }

    let input_shape = TensorShape::new(args.channels, args.input_h, args.input_w);
    let input = make_demo_tensor(input_shape, 19);
    let block = BottleneckBlock::demo(args.channels, !args.no_shortcut, args.depthwise)?;
    let (cpu_shape, cpu_output) = block.forward_cpu(&input, input_shape)?;
    let (gpu_shape, gpu_output) =
        run_demo_bottleneck(&input, input_shape, &block, args.device_index)?;

    if cpu_shape != gpu_shape {
        bail!(
            "shape final divergente: cpu={} gpu={}",
            cpu_shape.display_nchw(),
            gpu_shape.display_nchw()
        );
    }

    let diff = compare_slices(&cpu_output, &gpu_output)?;
    println!(
        "demo bottleneck: input={} output={} shortcut={} depthwise={}",
        input_shape.display_nchw(),
        cpu_shape.display_nchw(),
        !args.no_shortcut,
        args.depthwise
    );
    println!(
        "pipeline GPU: baseconv(1x1) -> {} -> {}",
        if args.depthwise {
            "dwsconv(3x3,1x1)"
        } else {
            "baseconv(3x3)"
        },
        if args.no_shortcut {
            "sem shortcut"
        } else {
            "shortcut add"
        }
    );
    println!(
        "comparação CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu_output.len(),
        diff.max_abs_diff,
        diff.mean_abs_diff,
    );
    println!(
        "amostra saída[0..8]={:?}",
        &gpu_output[..gpu_output.len().min(8)]
    );

    Ok(())
}

fn demo_csp(args: DemoCspArgs) -> Result<()> {
    if args.in_channels == 0 || args.out_channels == 0 || args.depth == 0 {
        bail!("in_channels, out_channels e depth devem ser maiores que zero");
    }
    if args.input_h == 0 || args.input_w == 0 {
        bail!("input_h e input_w devem ser maiores que zero");
    }

    let input_shape = TensorShape::new(args.in_channels, args.input_h, args.input_w);
    let input = make_demo_tensor(input_shape, 23);
    let block = CspStageBlock::demo(
        args.in_channels,
        args.out_channels,
        args.depth,
        args.depthwise,
    )?;
    let (cpu_shape, cpu_output) = block.forward_cpu(&input, input_shape)?;
    let (gpu_shape, gpu_output) = run_demo_csp(&input, input_shape, &block, args.device_index)?;

    if cpu_shape != gpu_shape {
        bail!(
            "shape final divergente: cpu={} gpu={}",
            cpu_shape.display_nchw(),
            gpu_shape.display_nchw()
        );
    }

    let diff = compare_slices(&cpu_output, &gpu_output)?;
    println!(
        "demo csp-stage: input={} output={} depth={} depthwise={}",
        input_shape.display_nchw(),
        cpu_shape.display_nchw(),
        args.depth,
        args.depthwise
    );
    println!("pipeline GPU: stride-conv -> csp-bottleneck");
    println!(
        "comparação CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu_output.len(),
        diff.max_abs_diff,
        diff.mean_abs_diff,
    );
    println!(
        "amostra saída[0..8]={:?}",
        &gpu_output[..gpu_output.len().min(8)]
    );

    Ok(())
}

fn demo_dark5(args: DemoDark5Args) -> Result<()> {
    if args.in_channels == 0 || args.out_channels == 0 || args.depth == 0 {
        bail!("in_channels, out_channels e depth devem ser maiores que zero");
    }
    if args.input_h == 0 || args.input_w == 0 {
        bail!("input_h e input_w devem ser maiores que zero");
    }

    let input_shape = TensorShape::new(args.in_channels, args.input_h, args.input_w);
    let input = make_demo_tensor(input_shape, 29);
    let block = Dark5Block::demo(
        args.in_channels,
        args.out_channels,
        args.depth,
        args.depthwise,
    )?;
    let (cpu_shape, cpu_output) = block.forward_cpu(&input, input_shape)?;
    let (gpu_shape, gpu_output) = run_demo_dark5(&input, input_shape, &block, args.device_index)?;

    if cpu_shape != gpu_shape {
        bail!(
            "shape final divergente: cpu={} gpu={}",
            cpu_shape.display_nchw(),
            gpu_shape.display_nchw()
        );
    }

    let diff = compare_slices(&cpu_output, &gpu_output)?;
    println!(
        "demo dark5: input={} output={} depth={} depthwise={}",
        input_shape.display_nchw(),
        cpu_shape.display_nchw(),
        args.depth,
        args.depthwise
    );
    println!("pipeline GPU: stride-conv -> spp-bottleneck -> csp-bottleneck");
    println!(
        "comparação CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu_output.len(),
        diff.max_abs_diff,
        diff.mean_abs_diff,
    );
    println!(
        "amostra saída[0..8]={:?}",
        &gpu_output[..gpu_output.len().min(8)]
    );

    Ok(())
}

fn demo_backbone(args: DemoBackboneArgs) -> Result<()> {
    if args.base_channels == 0 || args.base_depth == 0 {
        bail!("base_channels e base_depth devem ser maiores que zero");
    }
    if args.input_h == 0 || args.input_w == 0 {
        bail!("input_h e input_w devem ser maiores que zero");
    }
    if !args.input_h.is_multiple_of(32) || !args.input_w.is_multiple_of(32) {
        bail!("demo-backbone exige input_h e input_w múltiplos de 32");
    }

    let input_shape = TensorShape::new(3, args.input_h, args.input_w);
    let input = make_demo_tensor(input_shape, 31);
    let backbone = CspDarknetDemo::demo(args.base_channels, args.base_depth, args.depthwise)?;

    let cpu = backbone.forward_cpu(&input, input_shape)?;
    let gpu = run_demo_backbone(&input, input_shape, &backbone, args.device_index)?;

    if cpu.f1_shape != gpu.f1_shape || cpu.f2_shape != gpu.f2_shape || cpu.f3_shape != gpu.f3_shape
    {
        bail!(
            "shapes finais divergiram: cpu=({}, {}, {}) gpu=({}, {}, {})",
            cpu.f1_shape.display_nchw(),
            cpu.f2_shape.display_nchw(),
            cpu.f3_shape.display_nchw(),
            gpu.f1_shape.display_nchw(),
            gpu.f2_shape.display_nchw(),
            gpu.f3_shape.display_nchw(),
        );
    }

    let diff_f1 = compare_slices(&cpu.f1, &gpu.f1)?;
    let diff_f2 = compare_slices(&cpu.f2, &gpu.f2)?;
    let diff_f3 = compare_slices(&cpu.f3, &gpu.f3)?;

    println!(
        "demo backbone: input={} f1={} f2={} f3={} base_channels={} base_depth={} depthwise={}",
        input_shape.display_nchw(),
        cpu.f1_shape.display_nchw(),
        cpu.f2_shape.display_nchw(),
        cpu.f3_shape.display_nchw(),
        args.base_channels,
        args.base_depth,
        args.depthwise
    );
    println!("pipeline GPU: stem -> dark2 -> dark3 -> dark4 -> dark5");
    println!(
        "f1 CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu.f1.len(),
        diff_f1.max_abs_diff,
        diff_f1.mean_abs_diff,
    );
    println!(
        "f2 CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu.f2.len(),
        diff_f2.max_abs_diff,
        diff_f2.mean_abs_diff,
    );
    println!(
        "f3 CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu.f3.len(),
        diff_f3.max_abs_diff,
        diff_f3.mean_abs_diff,
    );
    println!("amostra f3[0..8]={:?}", &gpu.f3[..gpu.f3.len().min(8)]);

    Ok(())
}

fn demo_pafpn(args: DemoPafpnArgs) -> Result<()> {
    if args.base_channels == 0 || args.base_depth == 0 {
        bail!("base_channels e base_depth devem ser maiores que zero");
    }
    if args.input_h == 0 || args.input_w == 0 {
        bail!("input_h e input_w devem ser maiores que zero");
    }
    if !args.input_h.is_multiple_of(32) || !args.input_w.is_multiple_of(32) {
        bail!("demo-pafpn exige input_h e input_w múltiplos de 32");
    }

    let input_shape = TensorShape::new(3, args.input_h, args.input_w);
    let input = make_demo_tensor(input_shape, 37);
    let backbone = CspDarknetDemo::demo(args.base_channels, args.base_depth, args.depthwise)?;
    let pafpn = YoloxPafpnDemo::demo(args.base_channels, args.base_depth, args.depthwise)?;

    let backbone_cpu = backbone.forward_cpu(&input, input_shape)?;
    let cpu = pafpn.forward_cpu(&backbone_cpu)?;
    let gpu = run_demo_pafpn(&input, input_shape, &backbone, &pafpn, args.device_index)?;

    if cpu.p3_shape != gpu.p3_shape || cpu.p4_shape != gpu.p4_shape || cpu.p5_shape != gpu.p5_shape
    {
        bail!(
            "shapes finais divergiram: cpu=({}, {}, {}) gpu=({}, {}, {})",
            cpu.p3_shape.display_nchw(),
            cpu.p4_shape.display_nchw(),
            cpu.p5_shape.display_nchw(),
            gpu.p3_shape.display_nchw(),
            gpu.p4_shape.display_nchw(),
            gpu.p5_shape.display_nchw(),
        );
    }

    let diff_p3 = compare_slices(&cpu.p3, &gpu.p3)?;
    let diff_p4 = compare_slices(&cpu.p4, &gpu.p4)?;
    let diff_p5 = compare_slices(&cpu.p5, &gpu.p5)?;

    println!(
        "demo pafpn: input={} p3={} p4={} p5={} base_channels={} base_depth={} depthwise={}",
        input_shape.display_nchw(),
        cpu.p3_shape.display_nchw(),
        cpu.p4_shape.display_nchw(),
        cpu.p5_shape.display_nchw(),
        args.base_channels,
        args.base_depth,
        args.depthwise
    );
    println!("pipeline GPU: backbone -> pafpn");
    println!(
        "p3 CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu.p3.len(),
        diff_p3.max_abs_diff,
        diff_p3.mean_abs_diff,
    );
    println!(
        "p4 CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu.p4.len(),
        diff_p4.max_abs_diff,
        diff_p4.mean_abs_diff,
    );
    println!(
        "p5 CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu.p5.len(),
        diff_p5.max_abs_diff,
        diff_p5.mean_abs_diff,
    );
    println!("amostra p5[0..8]={:?}", &gpu.p5[..gpu.p5.len().min(8)]);

    Ok(())
}

fn demo_head(args: DemoHeadArgs) -> Result<()> {
    if args.base_channels == 0 || args.base_depth == 0 || args.num_classes == 0 {
        bail!("base_channels, base_depth e num_classes devem ser maiores que zero");
    }
    if args.input_h == 0 || args.input_w == 0 {
        bail!("input_h e input_w devem ser maiores que zero");
    }
    if !args.input_h.is_multiple_of(32) || !args.input_w.is_multiple_of(32) {
        bail!("demo-head exige input_h e input_w múltiplos de 32");
    }

    let input_shape = TensorShape::new(3, args.input_h, args.input_w);
    let input = make_demo_tensor(input_shape, 41);
    let backbone = CspDarknetDemo::demo(args.base_channels, args.base_depth, args.depthwise)?;
    let pafpn = YoloxPafpnDemo::demo(args.base_channels, args.base_depth, args.depthwise)?;
    let head = YoloxHeadDemo::demo(args.base_channels, args.num_classes, args.depthwise)?;

    let backbone_cpu = backbone.forward_cpu(&input, input_shape)?;
    let pafpn_cpu = pafpn.forward_cpu(&backbone_cpu)?;
    let cpu = head.forward_cpu(&pafpn_cpu)?;
    let gpu = run_demo_head(
        &input,
        input_shape,
        &backbone,
        &pafpn,
        &head,
        args.device_index,
    )?;

    if cpu.s8_shape != gpu.s8_shape
        || cpu.s16_shape != gpu.s16_shape
        || cpu.s32_shape != gpu.s32_shape
    {
        bail!(
            "shapes finais divergiram: cpu=({}, {}, {}) gpu=({}, {}, {})",
            cpu.s8_shape.display_nchw(),
            cpu.s16_shape.display_nchw(),
            cpu.s32_shape.display_nchw(),
            gpu.s8_shape.display_nchw(),
            gpu.s16_shape.display_nchw(),
            gpu.s32_shape.display_nchw(),
        );
    }

    let diff_s8 = compare_slices(&cpu.s8, &gpu.s8)?;
    let diff_s16 = compare_slices(&cpu.s16, &gpu.s16)?;
    let diff_s32 = compare_slices(&cpu.s32, &gpu.s32)?;

    println!(
        "demo head: input={} s8={} s16={} s32={} num_classes={} depthwise={}",
        input_shape.display_nchw(),
        cpu.s8_shape.display_nchw(),
        cpu.s16_shape.display_nchw(),
        cpu.s32_shape.display_nchw(),
        args.num_classes,
        args.depthwise
    );
    println!("pipeline GPU: backbone -> pafpn -> yolox-head");
    println!(
        "s8 CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu.s8.len(),
        diff_s8.max_abs_diff,
        diff_s8.mean_abs_diff,
    );
    println!(
        "s16 CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu.s16.len(),
        diff_s16.max_abs_diff,
        diff_s16.mean_abs_diff,
    );
    println!(
        "s32 CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu.s32.len(),
        diff_s32.max_abs_diff,
        diff_s32.mean_abs_diff,
    );
    println!("amostra s32[0..8]={:?}", &gpu.s32[..gpu.s32.len().min(8)]);

    Ok(())
}

fn demo_decode(args: DemoDecodeArgs) -> Result<()> {
    if args.base_channels == 0 || args.base_depth == 0 || args.num_classes == 0 {
        bail!("base_channels, base_depth e num_classes devem ser maiores que zero");
    }
    if args.input_h == 0 || args.input_w == 0 {
        bail!("input_h e input_w devem ser maiores que zero");
    }
    if !args.input_h.is_multiple_of(32) || !args.input_w.is_multiple_of(32) {
        bail!("demo-decode exige input_h e input_w múltiplos de 32");
    }

    let input_shape = TensorShape::new(3, args.input_h, args.input_w);
    let input = make_demo_tensor(input_shape, 43);
    let backbone = CspDarknetDemo::demo(args.base_channels, args.base_depth, args.depthwise)?;
    let pafpn = YoloxPafpnDemo::demo(args.base_channels, args.base_depth, args.depthwise)?;
    let head = YoloxHeadDemo::demo(args.base_channels, args.num_classes, args.depthwise)?;
    let decode = YoloxDecodeDemo::new(args.num_classes);

    let backbone_cpu = backbone.forward_cpu(&input, input_shape)?;
    let pafpn_cpu = pafpn.forward_cpu(&backbone_cpu)?;
    let head_cpu = head.forward_cpu(&pafpn_cpu)?;
    let cpu = decode.forward_cpu(&head_cpu)?;
    let gpu = run_demo_decode(
        &input,
        input_shape,
        &backbone,
        &pafpn,
        &head,
        &decode,
        args.device_index,
    )?;

    if cpu.rows != gpu.rows || cpu.cols != gpu.cols {
        bail!(
            "shape final divergente: cpu=[{}x{}] gpu=[{}x{}]",
            cpu.rows,
            cpu.cols,
            gpu.rows,
            gpu.cols
        );
    }

    let diff = compare_slices(&cpu.data, &gpu.data)?;
    println!(
        "demo decode: input={} output=[{}x{}] num_classes={} depthwise={}",
        input_shape.display_nchw(),
        cpu.rows,
        cpu.cols,
        args.num_classes,
        args.depthwise
    );
    println!("pipeline GPU: backbone -> pafpn -> yolox-head -> decode");
    println!(
        "CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu.data.len(),
        diff.max_abs_diff,
        diff.mean_abs_diff,
    );
    println!(
        "amostra decode[0..8]={:?}",
        &gpu.data[..gpu.data.len().min(8)]
    );

    Ok(())
}

fn demo_detect(args: DemoDetectArgs) -> Result<()> {
    if args.base_channels == 0 || args.base_depth == 0 {
        bail!("base_channels e base_depth devem ser maiores que zero");
    }
    validate_detect_args(
        "demo-detect",
        args.input_h,
        args.input_w,
        args.num_classes,
        args.max_detections,
        args.confidence_threshold,
        args.nms_threshold,
    )?;

    let input_shape = TensorShape::new(3, args.input_h, args.input_w);
    let input = make_demo_tensor(input_shape, 47);
    let backbone = CspDarknetDemo::demo(args.base_channels, args.base_depth, args.depthwise)?;
    let pafpn = YoloxPafpnDemo::demo(args.base_channels, args.base_depth, args.depthwise)?;
    let head = YoloxHeadDemo::demo(args.base_channels, args.num_classes, args.depthwise)?;
    let decode = YoloxDecodeDemo::new(args.num_classes);
    let postprocess = YoloxPostprocessDemo::new(
        args.num_classes,
        args.confidence_threshold,
        args.nms_threshold,
        args.max_detections,
    );

    let (cpu_decoded, _) = run_cpu_detect_demo(
        &input,
        input_shape,
        &backbone,
        &pafpn,
        &head,
        &decode,
        &postprocess,
    )?;
    let gpu_decoded = run_demo_decode(
        &input,
        input_shape,
        &backbone,
        &pafpn,
        &head,
        &decode,
        args.device_index,
    )?;

    if cpu_decoded.rows != gpu_decoded.rows || cpu_decoded.cols != gpu_decoded.cols {
        bail!(
            "shape final divergente: cpu=[{}x{}] gpu=[{}x{}]",
            cpu_decoded.rows,
            cpu_decoded.cols,
            gpu_decoded.rows,
            gpu_decoded.cols
        );
    }

    let diff = compare_slices(&cpu_decoded.data, &gpu_decoded.data)?;
    let detections = postprocess.forward_cpu(&gpu_decoded)?;

    print_detections(
        "demo detect",
        input_shape,
        &gpu_decoded,
        &detections,
        args.confidence_threshold,
        args.nms_threshold,
    );
    println!("pipeline GPU+CPU: backbone -> pafpn -> yolox-head -> decode -> nms(host)");
    println!(
        "decode CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu_decoded.data.len(),
        diff.max_abs_diff,
        diff.mean_abs_diff,
    );

    Ok(())
}

fn demo_detect_cpu(args: DemoDetectArgs) -> Result<()> {
    if args.base_channels == 0 || args.base_depth == 0 {
        bail!("base_channels e base_depth devem ser maiores que zero");
    }
    validate_detect_args(
        "demo-detect-cpu",
        args.input_h,
        args.input_w,
        args.num_classes,
        args.max_detections,
        args.confidence_threshold,
        args.nms_threshold,
    )?;

    let input_shape = TensorShape::new(3, args.input_h, args.input_w);
    let input = make_demo_tensor(input_shape, 61);
    let backbone = CspDarknetDemo::demo(args.base_channels, args.base_depth, args.depthwise)?;
    let pafpn = YoloxPafpnDemo::demo(args.base_channels, args.base_depth, args.depthwise)?;
    let head = YoloxHeadDemo::demo(args.base_channels, args.num_classes, args.depthwise)?;
    let decode = YoloxDecodeDemo::new(args.num_classes);
    let postprocess = YoloxPostprocessDemo::new(
        args.num_classes,
        args.confidence_threshold,
        args.nms_threshold,
        args.max_detections,
    );

    let (decoded, detections) = run_cpu_detect_demo(
        &input,
        input_shape,
        &backbone,
        &pafpn,
        &head,
        &decode,
        &postprocess,
    )?;

    print_detections(
        "demo detect cpu",
        input_shape,
        &decoded,
        &detections,
        args.confidence_threshold,
        args.nms_threshold,
    );
    println!("pipeline CPU: backbone -> pafpn -> yolox-head -> decode -> nms(host)");

    Ok(())
}

fn demo_detect_resident(args: DemoDetectArgs) -> Result<()> {
    if args.base_channels == 0 || args.base_depth == 0 {
        bail!("base_channels e base_depth devem ser maiores que zero");
    }
    validate_detect_args(
        "demo-detect-resident",
        args.input_h,
        args.input_w,
        args.num_classes,
        args.max_detections,
        args.confidence_threshold,
        args.nms_threshold,
    )?;

    let input_shape = TensorShape::new(3, args.input_h, args.input_w);
    let input = make_demo_tensor(input_shape, 53);
    let backbone = CspDarknetDemo::demo(args.base_channels, args.base_depth, args.depthwise)?;
    let pafpn = YoloxPafpnDemo::demo(args.base_channels, args.base_depth, args.depthwise)?;
    let head = YoloxHeadDemo::demo(args.base_channels, args.num_classes, args.depthwise)?;
    let decode = YoloxDecodeDemo::new(args.num_classes);
    let postprocess = YoloxPostprocessDemo::new(
        args.num_classes,
        args.confidence_threshold,
        args.nms_threshold,
        args.max_detections,
    );

    let (cpu_decoded, _) = run_cpu_detect_demo(
        &input,
        input_shape,
        &backbone,
        &pafpn,
        &head,
        &decode,
        &postprocess,
    )?;
    let gpu_decoded = run_demo_decode_resident(
        &input,
        input_shape,
        &backbone,
        &pafpn,
        &head,
        &decode,
        args.device_index,
    )?;

    if cpu_decoded.rows != gpu_decoded.rows || cpu_decoded.cols != gpu_decoded.cols {
        bail!(
            "shape final divergente: cpu=[{}x{}] gpu=[{}x{}]",
            cpu_decoded.rows,
            cpu_decoded.cols,
            gpu_decoded.rows,
            gpu_decoded.cols
        );
    }

    let diff = compare_slices(&cpu_decoded.data, &gpu_decoded.data)?;
    let detections = postprocess.forward_cpu(&gpu_decoded)?;

    print_detections(
        "demo detect resident",
        input_shape,
        &gpu_decoded,
        &detections,
        args.confidence_threshold,
        args.nms_threshold,
    );
    println!(
        "pipeline GPU+CPU: preload-weights(device-local) -> backbone -> pafpn -> yolox-head -> decode -> nms(host)"
    );
    println!(
        "decode CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu_decoded.data.len(),
        diff.max_abs_diff,
        diff.mean_abs_diff,
    );

    Ok(())
}

fn export_demo_bundle(args: ExportDemoBundleArgs) -> Result<()> {
    let bundle = DemoModelBundle::demo(
        args.base_channels,
        args.base_depth,
        args.num_classes,
        args.depthwise,
    )?;
    bundle.save_json(&args.output)?;
    println!("{}", bundle.summary());
    println!("bundle salvo em {}", args.output.display());
    Ok(())
}

fn inspect_bundle(args: InspectBundleArgs) -> Result<()> {
    let bundle = DemoModelBundle::load_json(&args.input)?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&bundle).context("falha ao serializar o bundle")?
        );
    } else {
        println!("{}", bundle.summary());
        println!("arquivo={}", args.input.display());
        if args.layers {
            println!("camadas:");
            for layer in bundle.named_layers() {
                println!(
                    "- {} {} params={:.3} KiB",
                    layer.name,
                    format!(
                        "[{}->{}, k={}x{}, s={}x{}, p={}x{}, g={}]",
                        layer.spec.in_channels,
                        layer.spec.out_channels,
                        layer.spec.kernel_h,
                        layer.spec.kernel_w,
                        layer.spec.stride_h,
                        layer.spec.stride_w,
                        layer.spec.pad_h,
                        layer.spec.pad_w,
                        layer.spec.groups,
                    ),
                    layer.parameter_bytes as f64 / 1024.0
                );
            }
        }
    }
    Ok(())
}

fn export_bundle_manifest(args: ExportBundleManifestArgs) -> Result<()> {
    let bundle = DemoModelBundle::load_json(&args.input)?;
    bundle.save_manifest_json(&args.output)?;
    println!("manifesto salvo em {}", args.output.display());
    println!("camadas={}", bundle.named_layers().len());
    Ok(())
}

fn export_burn_mapping_manifest(args: ExportBundleManifestArgs) -> Result<()> {
    let bundle = DemoModelBundle::load_json(&args.input)?;
    bundle.save_burn_mapping_manifest_json(&args.output)?;
    println!("mapeamento Burn salvo em {}", args.output.display());
    println!("camadas={}", bundle.named_layers().len());
    Ok(())
}

fn export_bundle_weights(args: ExportBundleWeightsArgs) -> Result<()> {
    let bundle = DemoModelBundle::load_json(&args.input)?;
    bundle.save_weight_patch_json(&args.output)?;
    println!("patch de pesos salvo em {}", args.output.display());
    println!("camadas={}", bundle.named_layers().len());
    Ok(())
}

fn build_raw_patch_from_burn_manifest(args: BuildRawPatchFromBurnManifestArgs) -> Result<()> {
    let bundle = DemoModelBundle::load_json(&args.bundle)?;
    let patch = bundle.build_raw_patch_from_external_manifest(&args.tensor_manifest)?;
    bundle.save_raw_weight_patch_json(&patch, &args.output)?;
    println!("patch raw gerado em {}", args.output.display());
    println!("camadas={}", patch.layers.len());
    Ok(())
}

fn apply_bundle_weights(args: ApplyBundleWeightsArgs) -> Result<()> {
    let mut bundle = DemoModelBundle::load_json(&args.bundle)?;
    let serialized = fs::read_to_string(&args.weights)
        .with_context(|| format!("falha ao ler patch {}", args.weights.display()))?;
    let patch =
        serde_json::from_str(&serialized).context("falha ao desserializar patch de pesos")?;
    bundle.apply_weight_patch(&patch)?;
    bundle.save_json(&args.output)?;
    println!("bundle atualizado salvo em {}", args.output.display());
    println!("{}", bundle.summary());
    Ok(())
}

fn apply_bundle_raw_weights(args: ApplyBundleRawWeightsArgs) -> Result<()> {
    let mut bundle = DemoModelBundle::load_json(&args.bundle)?;
    let patch: BundleRawWeightPatch = DemoModelBundle::load_raw_weight_patch_json(&args.raw_weights)
        .with_context(|| format!("falha ao carregar patch raw {}", args.raw_weights.display()))?;
    bundle.apply_raw_weight_patch(&patch)?;
    bundle.save_json(&args.output)?;
    println!("bundle atualizado com pesos raw salvo em {}", args.output.display());
    println!("{}", bundle.summary());
    Ok(())
}

fn export_bundle_weight_dir(args: ExportBundleWeightDirArgs) -> Result<()> {
    let bundle = DemoModelBundle::load_json(&args.input)?;
    bundle.save_weight_directory(&args.output_dir)?;
    println!("diretório de pesos salvo em {}", args.output_dir.display());
    println!("camadas={}", bundle.named_layers().len());
    Ok(())
}

fn apply_bundle_weight_dir(args: ApplyBundleWeightDirArgs) -> Result<()> {
    let mut bundle = DemoModelBundle::load_json(&args.bundle)?;
    let patch = bundle.load_weight_patch_from_directory(&args.weights_dir)?;
    bundle.apply_weight_patch(&patch)?;
    bundle.save_json(&args.output)?;
    println!("bundle atualizado salvo em {}", args.output.display());
    println!("{}", bundle.summary());
    Ok(())
}

fn demo_detect_bundle(args: DemoDetectBundleArgs) -> Result<()> {
    let bundle = DemoModelBundle::load_json(&args.input)?;
    validate_detect_args(
        "demo-detect-bundle",
        args.input_h,
        args.input_w,
        bundle.meta.num_classes,
        args.max_detections,
        args.confidence_threshold,
        args.nms_threshold,
    )?;
    let input_shape = TensorShape::new(3, args.input_h, args.input_w);
    let input = make_demo_tensor(input_shape, 59);
    let decode = YoloxDecodeDemo::new(bundle.meta.num_classes);
    let postprocess = YoloxPostprocessDemo::new(
        bundle.meta.num_classes,
        args.confidence_threshold,
        args.nms_threshold,
        args.max_detections,
    );

    let (cpu_decoded, cpu_detections) = run_cpu_detect_demo(
        &input,
        input_shape,
        &bundle.backbone,
        &bundle.pafpn,
        &bundle.head,
        &decode,
        &postprocess,
    )?;

    if args.cpu_only {
        print_detections(
            "demo detect bundle cpu",
            input_shape,
            &cpu_decoded,
            &cpu_detections,
            args.confidence_threshold,
            args.nms_threshold,
        );
        println!("pipeline CPU: bundle -> backbone -> pafpn -> yolox-head -> decode -> nms(host)");
        return Ok(());
    }

    let gpu_decoded = if args.resident_weights {
        run_demo_decode_resident(
            &input,
            input_shape,
            &bundle.backbone,
            &bundle.pafpn,
            &bundle.head,
            &decode,
            args.device_index,
        )?
    } else {
        run_demo_decode(
            &input,
            input_shape,
            &bundle.backbone,
            &bundle.pafpn,
            &bundle.head,
            &decode,
            args.device_index,
        )?
    };

    if cpu_decoded.rows != gpu_decoded.rows || cpu_decoded.cols != gpu_decoded.cols {
        bail!(
            "shape final divergente: cpu=[{}x{}] gpu=[{}x{}]",
            cpu_decoded.rows,
            cpu_decoded.cols,
            gpu_decoded.rows,
            gpu_decoded.cols
        );
    }

    let diff = compare_slices(&cpu_decoded.data, &gpu_decoded.data)?;
    let detections = postprocess.forward_cpu(&gpu_decoded)?;

    print_detections(
        "demo detect bundle",
        input_shape,
        &gpu_decoded,
        &detections,
        args.confidence_threshold,
        args.nms_threshold,
    );
    println!("resident_weights={}", args.resident_weights);
    println!(
        "decode CPU vs GPU: elementos={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu_decoded.data.len(),
        diff.max_abs_diff,
        diff.mean_abs_diff,
    );

    Ok(())
}

fn infer_bundle(args: InferBundleArgs) -> Result<()> {
    let bundle = DemoModelBundle::load_json(&args.bundle)?;
    validate_detect_args(
        "infer-bundle",
        args.input_h,
        args.input_w,
        bundle.meta.num_classes,
        args.max_detections,
        args.confidence_threshold,
        args.nms_threshold,
    )?;

    let (original_image, input_shape, input) =
        load_image_as_nchw_f32(&args.image, args.input_h, args.input_w)?;
    let (decoded, detections) = run_bundle_inference(
        &bundle,
        &input,
        input_shape,
        args.confidence_threshold,
        args.nms_threshold,
        args.max_detections,
        args.cpu_only,
        args.resident_weights,
        args.device_index,
    )?;

    print_detections(
        "infer bundle",
        input_shape,
        &decoded,
        &detections,
        args.confidence_threshold,
        args.nms_threshold,
    );

    let original_width = original_image.width();
    let original_height = original_image.height();
    let serializable = detections
        .iter()
        .map(|detection| {
            scale_detection_to_original(
                detection,
                original_width,
                original_height,
                args.input_w,
                args.input_h,
            )
        })
        .collect::<Vec<_>>();

    let report = InferenceReport {
        image: args.image.display().to_string(),
        original_width,
        original_height,
        input_h: args.input_h,
        input_w: args.input_w,
        cpu_only: args.cpu_only,
        resident_weights: args.resident_weights,
        detections: serializable,
    };

    if let Some(path) = &args.output_json {
        let serialized = serde_json::to_string_pretty(&report)
            .context("falha ao serializar relatório de inferência")?;
        fs::write(path, serialized)
            .with_context(|| format!("falha ao escrever {}", path.display()))?;
        println!("relatório salvo em {}", path.display());
    }

    if let Some(path) = &args.output_image {
        let annotated = draw_detection_rectangles(original_image, &report.detections);
        annotated
            .save(path)
            .with_context(|| format!("falha ao salvar {}", path.display()))?;
        println!("imagem anotada salva em {}", path.display());
    }

    Ok(())
}

fn bench_infer_bundle(args: BenchInferBundleArgs) -> Result<()> {
    if args.cpu_only && args.gpu_only {
        bail!("use apenas um entre --cpu-only e --gpu-only");
    }
    if args.iterations == 0 {
        bail!("iterations deve ser maior que zero");
    }

    let bundle_started = Instant::now();
    let bundle = DemoModelBundle::load_json(&args.bundle)?;
    let bundle_load_elapsed = bundle_started.elapsed();

    validate_detect_args(
        "bench-infer-bundle",
        args.input_h,
        args.input_w,
        bundle.meta.num_classes,
        args.max_detections,
        args.confidence_threshold,
        args.nms_threshold,
    )?;

    let input_started = Instant::now();
    let (_original_image, input_shape, input) =
        load_image_as_nchw_f32(&args.image, args.input_h, args.input_w)?;
    let input_prepare_elapsed = input_started.elapsed();

    let run_cpu = !args.gpu_only;
    let run_gpu = !args.cpu_only;

    let cpu = if run_cpu {
        Some(bench_inference_path(
            "cpu",
            &bundle,
            &input,
            input_shape,
            args.confidence_threshold,
            args.nms_threshold,
            args.max_detections,
            true,
            false,
            args.device_index,
            args.warmup_iterations,
            args.iterations,
        )?)
    } else {
        None
    };

    let gpu = if run_gpu {
        Some(bench_inference_path(
            if args.resident_weights {
                "gpu-resident"
            } else {
                "gpu"
            },
            &bundle,
            &input,
            input_shape,
            args.confidence_threshold,
            args.nms_threshold,
            args.max_detections,
            false,
            args.resident_weights,
            args.device_index,
            args.warmup_iterations,
            args.iterations,
        )?)
    } else {
        None
    };

    println!(
        "bench infer bundle: bundle_load={:.3}ms input_prepare={:.3}ms warmup={} iterations={}",
        duration_ms(bundle_load_elapsed),
        duration_ms(input_prepare_elapsed),
        args.warmup_iterations,
        args.iterations,
    );

    if let Some(report) = &cpu {
        println!(
            "{}: mean={:.3}ms median={:.3}ms min={:.3}ms max={:.3}ms fps={:.2} detections={}",
            report.label,
            report.mean_ms,
            report.median_ms,
            report.min_ms,
            report.max_ms,
            report.fps,
            report.detection_count,
        );
    }

    if let Some(report) = &gpu {
        println!(
            "{}: mean={:.3}ms median={:.3}ms min={:.3}ms max={:.3}ms fps={:.2} detections={}",
            report.label,
            report.mean_ms,
            report.median_ms,
            report.min_ms,
            report.max_ms,
            report.fps,
            report.detection_count,
        );
    }

    if let (Some(cpu), Some(gpu)) = (&cpu, &gpu) {
        println!(
            "speedup gpu/cpu: {:.2}x",
            if gpu.mean_ms > 0.0 {
                cpu.mean_ms / gpu.mean_ms
            } else {
                0.0
            }
        );
    }

    if let Some(path) = &args.output_json {
        let report = BenchInferenceReport {
            image: args.image.display().to_string(),
            input_h: args.input_h,
            input_w: args.input_w,
            bundle_load_ms: duration_ms(bundle_load_elapsed),
            input_prepare_ms: duration_ms(input_prepare_elapsed),
            cpu,
            gpu,
        };
        let serialized =
            serde_json::to_string_pretty(&report).context("falha ao serializar benchmark")?;
        fs::write(path, serialized)
            .with_context(|| format!("falha ao escrever {}", path.display()))?;
        println!("relatório salvo em {}", path.display());
    }

    Ok(())
}

fn compare_infer_bundle(args: CompareInferBundleArgs) -> Result<()> {
    let bundle = DemoModelBundle::load_json(&args.bundle)?;
    validate_detect_args(
        "compare-infer-bundle",
        args.input_h,
        args.input_w,
        bundle.meta.num_classes,
        args.max_detections,
        args.confidence_threshold,
        args.nms_threshold,
    )?;

    let (_original_image, input_shape, input) =
        load_image_as_nchw_f32(&args.image, args.input_h, args.input_w)?;

    let (cpu_decoded, cpu_detections) = run_bundle_inference(
        &bundle,
        &input,
        input_shape,
        args.confidence_threshold,
        args.nms_threshold,
        args.max_detections,
        true,
        false,
        args.device_index,
    )?;
    let (gpu_decoded, gpu_detections) = run_bundle_inference(
        &bundle,
        &input,
        input_shape,
        args.confidence_threshold,
        args.nms_threshold,
        args.max_detections,
        false,
        args.resident_weights,
        args.device_index,
    )?;

    if cpu_decoded.rows != gpu_decoded.rows || cpu_decoded.cols != gpu_decoded.cols {
        bail!(
            "shape final divergente: cpu=[{}x{}] gpu=[{}x{}]",
            cpu_decoded.rows,
            cpu_decoded.cols,
            gpu_decoded.rows,
            gpu_decoded.cols
        );
    }

    let diff = compare_slices(&cpu_decoded.data, &gpu_decoded.data)?;
    let compared = cpu_detections.len().min(gpu_detections.len());
    let detection_comparisons = (0..compared)
        .map(|index| {
            let cpu = &cpu_detections[index];
            let gpu = &gpu_detections[index];
            DetectionComparison {
                index,
                class_id_cpu: cpu.class_id,
                class_id_gpu: gpu.class_id,
                score_abs_diff: (cpu.score - gpu.score).abs(),
                x0_abs_diff: (cpu.x0 - gpu.x0).abs(),
                y0_abs_diff: (cpu.y0 - gpu.y0).abs(),
                x1_abs_diff: (cpu.x1 - gpu.x1).abs(),
                y1_abs_diff: (cpu.y1 - gpu.y1).abs(),
            }
        })
        .collect::<Vec<_>>();

    println!(
        "compare infer bundle: decoded_elements={} max_abs_diff={:.8} mean_abs_diff={:.8}",
        cpu_decoded.data.len(),
        diff.max_abs_diff,
        diff.mean_abs_diff,
    );
    println!(
        "detections cpu={} gpu={} compared={}",
        cpu_detections.len(),
        gpu_detections.len(),
        compared,
    );
    for item in detection_comparisons.iter().take(5) {
        println!(
            "det {:02}: class cpu/gpu={}/{} score_diff={:.6} box_diff=[{:.3}, {:.3}, {:.3}, {:.3}]",
            item.index,
            item.class_id_cpu,
            item.class_id_gpu,
            item.score_abs_diff,
            item.x0_abs_diff,
            item.y0_abs_diff,
            item.x1_abs_diff,
            item.y1_abs_diff,
        );
    }

    if let Some(path) = &args.output_json {
        let report = CompareInferenceReport {
            image: args.image.display().to_string(),
            input_h: args.input_h,
            input_w: args.input_w,
            resident_weights: args.resident_weights,
            decoded_elements: cpu_decoded.data.len(),
            decoded_max_abs_diff: diff.max_abs_diff,
            decoded_mean_abs_diff: diff.mean_abs_diff,
            cpu_detection_count: cpu_detections.len(),
            gpu_detection_count: gpu_detections.len(),
            compared_detections: compared,
            detection_comparisons,
        };
        let serialized = serde_json::to_string_pretty(&report)
            .context("falha ao serializar relatório de comparação")?;
        fs::write(path, serialized)
            .with_context(|| format!("falha ao escrever {}", path.display()))?;
        println!("relatório salvo em {}", path.display());
    }

    Ok(())
}
