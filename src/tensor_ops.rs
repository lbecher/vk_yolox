use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorShape {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl TensorShape {
    pub const fn new(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
        }
    }

    pub fn len(self) -> usize {
        self.channels * self.height * self.width
    }

    pub fn display_nchw(self) -> String {
        format!("[1x{}x{}x{}]", self.channels, self.height, self.width)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DiffStats {
    pub max_abs_diff: f32,
    pub mean_abs_diff: f64,
}

pub fn compare_slices(lhs: &[f32], rhs: &[f32]) -> Result<DiffStats> {
    if lhs.len() != rhs.len() {
        bail!("slices incompatíveis: lhs={} rhs={}", lhs.len(), rhs.len());
    }

    let mut max_abs_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f64;
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        let diff = (lhs - rhs).abs();
        max_abs_diff = max_abs_diff.max(diff);
        sum_abs_diff += diff as f64;
    }

    Ok(DiffStats {
        max_abs_diff,
        mean_abs_diff: sum_abs_diff / lhs.len() as f64,
    })
}

pub fn make_demo_tensor(shape: TensorShape, seed: usize) -> Vec<f32> {
    (0..shape.len())
        .map(|index| {
            let value = ((index * 37 + seed * 17 + 11) % 251) as f32;
            (value - 125.0) / 64.0
        })
        .collect()
}

#[inline]
pub fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn silu_nchw(input: &[f32]) -> Vec<f32> {
    input.iter().map(|value| silu_scalar(*value)).collect()
}

pub fn upsample_nearest_nchw(
    input: &[f32],
    shape: TensorShape,
    scale: usize,
) -> Result<(TensorShape, Vec<f32>)> {
    if scale == 0 {
        bail!("scale deve ser maior que zero");
    }
    if input.len() != shape.len() {
        bail!(
            "entrada inválida para upsample: esperado {} elementos, recebido {}",
            shape.len(),
            input.len()
        );
    }

    let output_shape = TensorShape::new(shape.channels, shape.height * scale, shape.width * scale);
    let mut output = vec![0.0; output_shape.len()];

    for c in 0..shape.channels {
        for oy in 0..output_shape.height {
            let iy = oy / scale;
            for ox in 0..output_shape.width {
                let ix = ox / scale;
                let src = ((c * shape.height + iy) * shape.width) + ix;
                let dst = ((c * output_shape.height + oy) * output_shape.width) + ox;
                output[dst] = input[src];
            }
        }
    }

    Ok((output_shape, output))
}

pub fn concat_channels_nchw(
    lhs: &[f32],
    lhs_shape: TensorShape,
    rhs: &[f32],
    rhs_shape: TensorShape,
) -> Result<(TensorShape, Vec<f32>)> {
    if lhs.len() != lhs_shape.len() || rhs.len() != rhs_shape.len() {
        bail!("buffers incompatíveis com os shapes informados");
    }
    if lhs_shape.height != rhs_shape.height || lhs_shape.width != rhs_shape.width {
        bail!(
            "concat exige mesmo HxW: lhs={}x{} rhs={}x{}",
            lhs_shape.height,
            lhs_shape.width,
            rhs_shape.height,
            rhs_shape.width
        );
    }

    let out_shape = TensorShape::new(
        lhs_shape.channels + rhs_shape.channels,
        lhs_shape.height,
        lhs_shape.width,
    );
    let plane = lhs_shape.height * lhs_shape.width;
    let mut output = vec![0.0; out_shape.len()];

    for channel in 0..lhs_shape.channels {
        let src = channel * plane;
        let dst = channel * plane;
        output[dst..dst + plane].copy_from_slice(&lhs[src..src + plane]);
    }

    for channel in 0..rhs_shape.channels {
        let src = channel * plane;
        let dst = (lhs_shape.channels + channel) * plane;
        output[dst..dst + plane].copy_from_slice(&rhs[src..src + plane]);
    }

    Ok((out_shape, output))
}

#[inline]
pub fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_nchw(input: &[f32]) -> Vec<f32> {
    input.iter().map(|value| sigmoid_scalar(*value)).collect()
}

pub fn add_nchw(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>> {
    if lhs.len() != rhs.len() {
        bail!("add incompatível: lhs={} rhs={}", lhs.len(), rhs.len());
    }

    Ok(lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs + rhs).collect())
}

pub fn maxpool2d_nchw(
    input: &[f32],
    shape: TensorShape,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<(TensorShape, Vec<f32>)> {
    if kernel == 0 || stride == 0 {
        bail!("kernel e stride devem ser maiores que zero");
    }
    if input.len() != shape.len() {
        bail!(
            "entrada inválida para maxpool: esperado {} elementos, recebido {}",
            shape.len(),
            input.len()
        );
    }

    let padded_h = shape.height + padding * 2;
    let padded_w = shape.width + padding * 2;
    if padded_h < kernel || padded_w < kernel {
        bail!("kernel maior que a entrada efetiva após padding");
    }

    let output_h = (padded_h - kernel) / stride + 1;
    let output_w = (padded_w - kernel) / stride + 1;
    let output_shape = TensorShape::new(shape.channels, output_h, output_w);
    let mut output = vec![0.0; output_shape.len()];

    for c in 0..shape.channels {
        for oy in 0..output_h {
            for ox in 0..output_w {
                let mut max_value = f32::NEG_INFINITY;
                for ky in 0..kernel {
                    let in_y = oy * stride + ky;
                    if in_y < padding {
                        continue;
                    }
                    let in_y = in_y - padding;
                    if in_y >= shape.height {
                        continue;
                    }

                    for kx in 0..kernel {
                        let in_x = ox * stride + kx;
                        if in_x < padding {
                            continue;
                        }
                        let in_x = in_x - padding;
                        if in_x >= shape.width {
                            continue;
                        }

                        let idx = ((c * shape.height + in_y) * shape.width) + in_x;
                        max_value = max_value.max(input[idx]);
                    }
                }

                let out_idx = ((c * output_h + oy) * output_w) + ox;
                output[out_idx] = max_value;
            }
        }
    }

    Ok((output_shape, output))
}

pub fn focus_nchw(input: &[f32], shape: TensorShape) -> Result<(TensorShape, Vec<f32>)> {
    if shape.height % 2 != 0 || shape.width % 2 != 0 {
        bail!(
            "focus exige H e W pares, recebido {}x{}",
            shape.height,
            shape.width
        );
    }
    if input.len() != shape.len() {
        bail!(
            "entrada inválida para focus: esperado {} elementos, recebido {}",
            shape.len(),
            input.len()
        );
    }

    let output_shape = TensorShape::new(shape.channels * 4, shape.height / 2, shape.width / 2);
    let mut output = vec![0.0; output_shape.len()];
    let plane = output_shape.height * output_shape.width;

    for c in 0..shape.channels {
        for oy in 0..output_shape.height {
            for ox in 0..output_shape.width {
                let src_y = oy * 2;
                let src_x = ox * 2;
                let patches = [
                    (src_y, src_x),
                    (src_y + 1, src_x),
                    (src_y, src_x + 1),
                    (src_y + 1, src_x + 1),
                ];

                for (patch_idx, (iy, ix)) in patches.into_iter().enumerate() {
                    let src = ((c * shape.height + iy) * shape.width) + ix;
                    let out_channel = c + patch_idx * shape.channels;
                    let dst = out_channel * plane + oy * output_shape.width + ox;
                    output[dst] = input[src];
                }
            }
        }
    }

    Ok((output_shape, output))
}
