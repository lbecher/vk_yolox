use anyhow::{Result, bail};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Conv2dSpec {
    pub in_channels: usize,
    pub out_channels: usize,
    pub groups: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
}

impl Conv2dSpec {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Self {
        Self::new_grouped(
            in_channels,
            out_channels,
            1,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        )
    }

    pub fn new_grouped(
        in_channels: usize,
        out_channels: usize,
        groups: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            groups,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        }
    }

    pub fn output_hw(&self, input_h: usize, input_w: usize) -> Result<(usize, usize)> {
        self.validate()?;
        let padded_h = input_h + self.pad_h * 2;
        let padded_w = input_w + self.pad_w * 2;
        if padded_h < self.kernel_h || padded_w < self.kernel_w {
            bail!("kernel maior que a entrada efetiva após padding");
        }

        let out_h = (padded_h - self.kernel_h) / self.stride_h + 1;
        let out_w = (padded_w - self.kernel_w) / self.stride_w + 1;
        Ok((out_h, out_w))
    }

    pub fn validate(&self) -> Result<()> {
        if self.groups == 0 {
            bail!("groups deve ser maior que zero");
        }
        if self.in_channels == 0 || self.out_channels == 0 {
            bail!("in_channels e out_channels devem ser maiores que zero");
        }
        if !self.in_channels.is_multiple_of(self.groups) {
            bail!(
                "in_channels={} deve ser múltiplo de groups={}",
                self.in_channels,
                self.groups
            );
        }
        if !self.out_channels.is_multiple_of(self.groups) {
            bail!(
                "out_channels={} deve ser múltiplo de groups={}",
                self.out_channels,
                self.groups
            );
        }

        Ok(())
    }

    pub fn in_channels_per_group(&self) -> usize {
        self.in_channels / self.groups
    }

    pub fn out_channels_per_group(&self) -> usize {
        self.out_channels / self.groups
    }

    pub fn weight_len(&self) -> usize {
        self.out_channels * self.in_channels_per_group() * self.kernel_h * self.kernel_w
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawConv2dWeights {
    pub spec: Conv2dSpec,
    pub weights: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl RawConv2dWeights {
    pub fn demo(spec: &Conv2dSpec) -> Self {
        let weights = (0..spec.weight_len())
            .map(|index| {
                let value = ((index * 17 + 29) % 127) as f32;
                (value - 63.0) / 90.0
            })
            .collect();

        Self {
            spec: *spec,
            weights,
            bias: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchNorm1d {
    pub scale: Vec<f32>,
    pub bias: Vec<f32>,
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
    pub epsilon: f32,
}

impl BatchNorm1d {
    pub fn demo(channels: usize) -> Self {
        let scale = (0..channels)
            .map(|index| 0.75 + (index % 7) as f32 * 0.05)
            .collect();
        let bias = (0..channels)
            .map(|index| -0.2 + (index % 5) as f32 * 0.03)
            .collect();
        let mean = (0..channels)
            .map(|index| -0.15 + (index % 9) as f32 * 0.02)
            .collect();
        let var = (0..channels)
            .map(|index| 0.9 + (index % 11) as f32 * 0.04)
            .collect();

        Self {
            scale,
            bias,
            mean,
            var,
            epsilon: 1e-3,
        }
    }

    pub fn identity(channels: usize) -> Self {
        let epsilon = 1e-3;
        Self {
            scale: vec![1.0; channels],
            bias: vec![0.0; channels],
            mean: vec![0.0; channels],
            var: vec![1.0 - epsilon; channels],
            epsilon,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FusedConv2dWeights {
    pub spec: Conv2dSpec,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl FusedConv2dWeights {
    pub fn convolve_nchw_f32(
        &self,
        input: &[f32],
        input_h: usize,
        input_w: usize,
    ) -> Result<Vec<f32>> {
        self.spec.validate()?;
        let expected_input = self.spec.in_channels * input_h * input_w;
        if input.len() != expected_input {
            bail!(
                "entrada inválida: esperado {} elementos, recebido {}",
                expected_input,
                input.len()
            );
        }

        let (output_h, output_w) = self.spec.output_hw(input_h, input_w)?;
        let mut output = vec![0.0; self.spec.out_channels * output_h * output_w];
        let plane = output_h * output_w;
        let work = self.spec.out_channels * plane * self.spec.in_channels_per_group();

        if work >= 16_384 {
            output
                .par_chunks_mut(plane)
                .enumerate()
                .for_each(|(oc, plane_out)| {
                    self.convolve_output_channel(
                        input, input_h, input_w, output_h, output_w, oc, plane_out,
                    );
                });
            return Ok(output);
        }

        for (oc, plane_out) in output.chunks_mut(plane).enumerate() {
            self.convolve_output_channel(
                input, input_h, input_w, output_h, output_w, oc, plane_out,
            );
        }

        Ok(output)
    }

    fn convolve_output_channel(
        &self,
        input: &[f32],
        input_h: usize,
        input_w: usize,
        output_h: usize,
        output_w: usize,
        oc: usize,
        plane_out: &mut [f32],
    ) {
        let in_channels_per_group = self.spec.in_channels_per_group();
        let out_channels_per_group = self.spec.out_channels_per_group();
        let group = oc / out_channels_per_group;
        let ic_start = group * in_channels_per_group;
        let ic_end = ic_start + in_channels_per_group;

        for oy in 0..output_h {
            for ox in 0..output_w {
                let mut acc = self.bias[oc];
                for ic in ic_start..ic_end {
                    let group_ic = ic - ic_start;
                    for ky in 0..self.spec.kernel_h {
                        let in_y = oy * self.spec.stride_h + ky;
                        if in_y < self.spec.pad_h {
                            continue;
                        }
                        let in_y = in_y - self.spec.pad_h;
                        if in_y >= input_h {
                            continue;
                        }

                        for kx in 0..self.spec.kernel_w {
                            let in_x = ox * self.spec.stride_w + kx;
                            if in_x < self.spec.pad_w {
                                continue;
                            }
                            let in_x = in_x - self.spec.pad_w;
                            if in_x >= input_w {
                                continue;
                            }

                            let input_idx = ((ic * input_h + in_y) * input_w) + in_x;
                            let weight_idx = (((oc * in_channels_per_group + group_ic)
                                * self.spec.kernel_h
                                + ky)
                                * self.spec.kernel_w)
                                + kx;
                            acc += input[input_idx] * self.weights[weight_idx];
                        }
                    }
                }

                plane_out[oy * output_w + ox] = acc;
            }
        }
    }
}

pub fn fuse_conv2d_bn(raw: &RawConv2dWeights, bn: &BatchNorm1d) -> Result<FusedConv2dWeights> {
    raw.spec.validate()?;
    let channels = raw.spec.out_channels;
    if raw.weights.len() != raw.spec.weight_len() {
        bail!("tensor de pesos conv inválido");
    }
    if bn.scale.len() != channels
        || bn.bias.len() != channels
        || bn.mean.len() != channels
        || bn.var.len() != channels
    {
        bail!("batchnorm incompatível com out_channels={channels}");
    }
    if let Some(bias) = &raw.bias
        && bias.len() != channels
    {
        bail!("bias conv incompatível com out_channels={channels}");
    }

    let kernel_span = raw.spec.in_channels_per_group() * raw.spec.kernel_h * raw.spec.kernel_w;
    let mut fused_weights = raw.weights.clone();
    let mut fused_bias = vec![0.0; channels];

    for oc in 0..channels {
        let scale = bn.scale[oc] / (bn.var[oc] + bn.epsilon).sqrt();
        let base = oc * kernel_span;
        for weight in &mut fused_weights[base..base + kernel_span] {
            *weight *= scale;
        }

        let conv_bias = raw.bias.as_ref().map(|bias| bias[oc]).unwrap_or(0.0);
        fused_bias[oc] = bn.bias[oc] + (conv_bias - bn.mean[oc]) * scale;
    }

    Ok(FusedConv2dWeights {
        spec: raw.spec,
        weights: fused_weights,
        bias: fused_bias,
    })
}
