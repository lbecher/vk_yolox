use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

use crate::{
    fused_weights::{
        BatchNorm1d, Conv2dSpec, FusedConv2dWeights, RawConv2dWeights, fuse_conv2d_bn,
    },
    tensor_ops::{
        TensorShape, add_nchw, concat_channels_nchw, focus_nchw, maxpool2d_nchw, sigmoid_scalar,
        silu_nchw, upsample_nearest_nchw,
    },
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BaseConvBlock {
    pub conv: FusedConv2dWeights,
}

impl BaseConvBlock {
    pub fn demo(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let spec = Conv2dSpec::new(
            in_channels,
            out_channels,
            kernel,
            kernel,
            stride,
            stride,
            padding,
            padding,
        );
        let raw = RawConv2dWeights::demo(&spec);
        let bn = BatchNorm1d::demo(out_channels);

        Ok(Self {
            conv: fuse_conv2d_bn(&raw, &bn)?,
        })
    }

    pub fn forward_cpu(
        &self,
        input: &[f32],
        input_shape: TensorShape,
    ) -> Result<(TensorShape, Vec<f32>)> {
        if input_shape.channels != self.conv.spec.in_channels {
            bail!(
                "BaseConv incompatível: input C={} conv C={}",
                input_shape.channels,
                self.conv.spec.in_channels
            );
        }

        let (output_h, output_w) = self
            .conv
            .spec
            .output_hw(input_shape.height, input_shape.width)?;
        let output = self
            .conv
            .convolve_nchw_f32(input, input_shape.height, input_shape.width)?;

        Ok((
            TensorShape::new(self.conv.spec.out_channels, output_h, output_w),
            silu_nchw(&output),
        ))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DwsConvBlock {
    pub depthwise: FusedConv2dWeights,
    pub pointwise: FusedConv2dWeights,
}

impl DwsConvBlock {
    pub fn demo(
        channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self> {
        let depthwise_spec = Conv2dSpec::new_grouped(
            channels,
            channels,
            channels,
            kernel,
            kernel,
            stride,
            stride,
            kernel / 2,
            kernel / 2,
        );
        let pointwise_spec = Conv2dSpec::new(channels, out_channels, 1, 1, 1, 1, 0, 0);

        let depthwise = fuse_conv2d_bn(
            &RawConv2dWeights::demo(&depthwise_spec),
            &BatchNorm1d::demo(channels),
        )?;
        let pointwise = fuse_conv2d_bn(
            &RawConv2dWeights::demo(&pointwise_spec),
            &BatchNorm1d::demo(out_channels),
        )?;

        Ok(Self {
            depthwise,
            pointwise,
        })
    }

    pub fn forward_cpu(
        &self,
        input: &[f32],
        input_shape: TensorShape,
    ) -> Result<(TensorShape, Vec<f32>)> {
        let (depthwise_h, depthwise_w) = self
            .depthwise
            .spec
            .output_hw(input_shape.height, input_shape.width)?;
        let depthwise_shape =
            TensorShape::new(self.depthwise.spec.out_channels, depthwise_h, depthwise_w);
        let depthwise =
            self.depthwise
                .convolve_nchw_f32(input, input_shape.height, input_shape.width)?;
        let depthwise = silu_nchw(&depthwise);

        let (pointwise_h, pointwise_w) = self
            .pointwise
            .spec
            .output_hw(depthwise_shape.height, depthwise_shape.width)?;
        let pointwise_shape =
            TensorShape::new(self.pointwise.spec.out_channels, pointwise_h, pointwise_w);
        let pointwise = self.pointwise.convolve_nchw_f32(
            &depthwise,
            depthwise_shape.height,
            depthwise_shape.width,
        )?;

        Ok((pointwise_shape, silu_nchw(&pointwise)))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConvBlock {
    Base(BaseConvBlock),
    Dws(DwsConvBlock),
}

impl ConvBlock {
    pub fn demo(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        depthwise: bool,
    ) -> Result<Self> {
        if depthwise {
            Ok(Self::Dws(DwsConvBlock::demo(
                in_channels,
                out_channels,
                kernel,
                stride,
            )?))
        } else {
            Ok(Self::Base(BaseConvBlock::demo(
                in_channels,
                out_channels,
                kernel,
                stride,
                padding,
            )?))
        }
    }

    pub fn forward_cpu(
        &self,
        input: &[f32],
        input_shape: TensorShape,
    ) -> Result<(TensorShape, Vec<f32>)> {
        match self {
            Self::Base(block) => block.forward_cpu(input, input_shape),
            Self::Dws(block) => block.forward_cpu(input, input_shape),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BottleneckBlock {
    pub conv1: BaseConvBlock,
    pub conv2: ConvBlock,
    pub shortcut: bool,
}

impl BottleneckBlock {
    pub fn demo(channels: usize, shortcut: bool, depthwise: bool) -> Result<Self> {
        let conv1 = BaseConvBlock::demo(channels, channels, 1, 1, 0)?;
        let conv2 = if depthwise {
            ConvBlock::Dws(DwsConvBlock::demo(channels, channels, 3, 1)?)
        } else {
            ConvBlock::Base(BaseConvBlock::demo(channels, channels, 3, 1, 1)?)
        };

        Ok(Self {
            conv1,
            conv2,
            shortcut,
        })
    }

    pub fn forward_cpu(
        &self,
        input: &[f32],
        input_shape: TensorShape,
    ) -> Result<(TensorShape, Vec<f32>)> {
        let identity = input.to_vec();
        let (shape1, x1) = self.conv1.forward_cpu(input, input_shape)?;
        let (shape2, mut x2) = self.conv2.forward_cpu(&x1, shape1)?;

        if self.shortcut {
            if shape2 != input_shape {
                bail!(
                    "shortcut inválido: saída={} entrada={}",
                    shape2.display_nchw(),
                    input_shape.display_nchw()
                );
            }
            x2 = add_nchw(&x2, &identity)?;
        }

        Ok((shape2, x2))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CspBottleneckBlock {
    pub conv1: BaseConvBlock,
    pub conv2: BaseConvBlock,
    pub conv3: BaseConvBlock,
    pub blocks: Vec<BottleneckBlock>,
}

impl CspBottleneckBlock {
    pub fn demo(
        in_channels: usize,
        out_channels: usize,
        num_blocks: usize,
        expansion: f64,
        shortcut: bool,
        depthwise: bool,
    ) -> Result<Self> {
        if !(0.0 < expansion && expansion <= 1.0) {
            bail!("expansion deve estar em (0, 1]");
        }

        let hidden_channels = ((out_channels as f64) * expansion).floor() as usize;
        let conv1 = BaseConvBlock::demo(in_channels, hidden_channels, 1, 1, 0)?;
        let conv2 = BaseConvBlock::demo(in_channels, hidden_channels, 1, 1, 0)?;
        let conv3 = BaseConvBlock::demo(hidden_channels * 2, out_channels, 1, 1, 0)?;
        let blocks = (0..num_blocks)
            .map(|_| BottleneckBlock::demo(hidden_channels, shortcut, depthwise))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            conv1,
            conv2,
            conv3,
            blocks,
        })
    }

    pub fn forward_cpu(
        &self,
        input: &[f32],
        input_shape: TensorShape,
    ) -> Result<(TensorShape, Vec<f32>)> {
        let (mut x1_shape, mut x1) = self.conv1.forward_cpu(input, input_shape)?;
        let (x2_shape, x2) = self.conv2.forward_cpu(input, input_shape)?;

        for block in &self.blocks {
            let (shape, out) = block.forward_cpu(&x1, x1_shape)?;
            x1_shape = shape;
            x1 = out;
        }

        let (cat_shape, cat) = concat_channels_nchw(&x1, x1_shape, &x2, x2_shape)?;
        self.conv3.forward_cpu(&cat, cat_shape)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CspStageBlock {
    pub conv: ConvBlock,
    pub c3: CspBottleneckBlock,
}

impl CspStageBlock {
    pub fn demo(
        in_channels: usize,
        out_channels: usize,
        depth: usize,
        depthwise: bool,
    ) -> Result<Self> {
        let conv = if depthwise {
            ConvBlock::Dws(DwsConvBlock::demo(in_channels, out_channels, 3, 2)?)
        } else {
            ConvBlock::Base(BaseConvBlock::demo(in_channels, out_channels, 3, 2, 1)?)
        };
        let c3 = CspBottleneckBlock::demo(out_channels, out_channels, depth, 0.5, true, depthwise)?;

        Ok(Self { conv, c3 })
    }

    pub fn forward_cpu(
        &self,
        input: &[f32],
        input_shape: TensorShape,
    ) -> Result<(TensorShape, Vec<f32>)> {
        let (shape, x) = self.conv.forward_cpu(input, input_shape)?;
        self.c3.forward_cpu(&x, shape)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SppBottleneckBlock {
    pub conv1: BaseConvBlock,
    pub conv2: BaseConvBlock,
    pub pooling: [usize; 3],
}

impl SppBottleneckBlock {
    pub fn demo(in_channels: usize, out_channels: usize) -> Result<Self> {
        let hidden_channels = in_channels / 2;
        let conv1 = BaseConvBlock::demo(in_channels, hidden_channels, 1, 1, 0)?;
        let conv2 = BaseConvBlock::demo(hidden_channels * 4, out_channels, 1, 1, 0)?;

        Ok(Self {
            conv1,
            conv2,
            pooling: [5, 9, 13],
        })
    }

    pub fn forward_cpu(
        &self,
        input: &[f32],
        input_shape: TensorShape,
    ) -> Result<(TensorShape, Vec<f32>)> {
        let (base_shape, base) = self.conv1.forward_cpu(input, input_shape)?;
        let (_, p0) = maxpool2d_nchw(&base, base_shape, self.pooling[0], 1, self.pooling[0] / 2)?;
        let (_, p1) = maxpool2d_nchw(&base, base_shape, self.pooling[1], 1, self.pooling[1] / 2)?;
        let (_, p2) = maxpool2d_nchw(&base, base_shape, self.pooling[2], 1, self.pooling[2] / 2)?;

        let (cat01_shape, cat01) = concat_channels_nchw(&base, base_shape, &p0, base_shape)?;
        let (cat012_shape, cat012) = concat_channels_nchw(&cat01, cat01_shape, &p1, base_shape)?;
        let (cat_shape, cat) = concat_channels_nchw(&cat012, cat012_shape, &p2, base_shape)?;

        self.conv2.forward_cpu(&cat, cat_shape)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dark5Block {
    pub conv: ConvBlock,
    pub spp: SppBottleneckBlock,
    pub c3: CspBottleneckBlock,
}

impl Dark5Block {
    pub fn demo(
        in_channels: usize,
        out_channels: usize,
        depth: usize,
        depthwise: bool,
    ) -> Result<Self> {
        let conv = if depthwise {
            ConvBlock::Dws(DwsConvBlock::demo(in_channels, out_channels, 3, 2)?)
        } else {
            ConvBlock::Base(BaseConvBlock::demo(in_channels, out_channels, 3, 2, 1)?)
        };
        let spp = SppBottleneckBlock::demo(out_channels, out_channels)?;
        let c3 =
            CspBottleneckBlock::demo(out_channels, out_channels, depth, 0.5, false, depthwise)?;

        Ok(Self { conv, spp, c3 })
    }

    pub fn forward_cpu(
        &self,
        input: &[f32],
        input_shape: TensorShape,
    ) -> Result<(TensorShape, Vec<f32>)> {
        let (shape, x) = self.conv.forward_cpu(input, input_shape)?;
        let (shape, x) = self.spp.forward_cpu(&x, shape)?;
        self.c3.forward_cpu(&x, shape)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FocusStemBlock {
    pub conv: BaseConvBlock,
}

impl FocusStemBlock {
    pub fn demo(out_channels: usize) -> Result<Self> {
        Ok(Self {
            conv: BaseConvBlock::demo(12, out_channels, 3, 1, 1)?,
        })
    }

    pub fn forward_cpu(
        &self,
        input: &[f32],
        input_shape: TensorShape,
    ) -> Result<(TensorShape, Vec<f32>)> {
        let (focused_shape, focused) = focus_nchw(input, input_shape)?;
        self.conv.forward_cpu(&focused, focused_shape)
    }
}

#[derive(Clone, Debug)]
pub struct DarknetFeatures {
    pub f1_shape: TensorShape,
    pub f1: Vec<f32>,
    pub f2_shape: TensorShape,
    pub f2: Vec<f32>,
    pub f3_shape: TensorShape,
    pub f3: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CspDarknetDemo {
    pub stem: FocusStemBlock,
    pub dark2: CspStageBlock,
    pub dark3: CspStageBlock,
    pub dark4: CspStageBlock,
    pub dark5: Dark5Block,
}

impl CspDarknetDemo {
    pub fn demo(base_channels: usize, base_depth: usize, depthwise: bool) -> Result<Self> {
        Ok(Self {
            stem: FocusStemBlock::demo(base_channels)?,
            dark2: CspStageBlock::demo(base_channels, base_channels * 2, base_depth, depthwise)?,
            dark3: CspStageBlock::demo(
                base_channels * 2,
                base_channels * 4,
                base_depth * 3,
                depthwise,
            )?,
            dark4: CspStageBlock::demo(
                base_channels * 4,
                base_channels * 8,
                base_depth * 3,
                depthwise,
            )?,
            dark5: Dark5Block::demo(base_channels * 8, base_channels * 16, base_depth, depthwise)?,
        })
    }

    pub fn forward_cpu(&self, input: &[f32], input_shape: TensorShape) -> Result<DarknetFeatures> {
        let (stem_shape, stem) = self.stem.forward_cpu(input, input_shape)?;
        let (dark2_shape, dark2) = self.dark2.forward_cpu(&stem, stem_shape)?;
        let (f1_shape, f1) = self.dark3.forward_cpu(&dark2, dark2_shape)?;
        let (f2_shape, f2) = self.dark4.forward_cpu(&f1, f1_shape)?;
        let (f3_shape, f3) = self.dark5.forward_cpu(&f2, f2_shape)?;

        Ok(DarknetFeatures {
            f1_shape,
            f1,
            f2_shape,
            f2,
            f3_shape,
            f3,
        })
    }
}

#[derive(Clone, Debug)]
pub struct PafpnFeatures {
    pub p3_shape: TensorShape,
    pub p3: Vec<f32>,
    pub p4_shape: TensorShape,
    pub p4: Vec<f32>,
    pub p5_shape: TensorShape,
    pub p5: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct YoloxPafpnDemo {
    pub lateral_conv0: BaseConvBlock,
    pub c3_p4: CspBottleneckBlock,
    pub reduce_conv1: BaseConvBlock,
    pub c3_p3: CspBottleneckBlock,
    pub bu_conv2: ConvBlock,
    pub c3_n3: CspBottleneckBlock,
    pub bu_conv1: ConvBlock,
    pub c3_n4: CspBottleneckBlock,
}

impl YoloxPafpnDemo {
    pub fn demo(base_channels: usize, base_depth: usize, depthwise: bool) -> Result<Self> {
        let neck_depth = base_depth * 3;
        let lateral_conv0 = BaseConvBlock::demo(base_channels * 16, base_channels * 8, 1, 1, 0)?;
        let c3_p4 = CspBottleneckBlock::demo(
            base_channels * 16,
            base_channels * 8,
            neck_depth,
            0.5,
            false,
            depthwise,
        )?;
        let reduce_conv1 = BaseConvBlock::demo(base_channels * 8, base_channels * 4, 1, 1, 0)?;
        let c3_p3 = CspBottleneckBlock::demo(
            base_channels * 8,
            base_channels * 4,
            neck_depth,
            0.5,
            false,
            depthwise,
        )?;
        let bu_conv2 = if depthwise {
            ConvBlock::Dws(DwsConvBlock::demo(
                base_channels * 4,
                base_channels * 4,
                3,
                2,
            )?)
        } else {
            ConvBlock::Base(BaseConvBlock::demo(
                base_channels * 4,
                base_channels * 4,
                3,
                2,
                1,
            )?)
        };
        let c3_n3 = CspBottleneckBlock::demo(
            base_channels * 8,
            base_channels * 8,
            neck_depth,
            0.5,
            false,
            depthwise,
        )?;
        let bu_conv1 = if depthwise {
            ConvBlock::Dws(DwsConvBlock::demo(
                base_channels * 8,
                base_channels * 8,
                3,
                2,
            )?)
        } else {
            ConvBlock::Base(BaseConvBlock::demo(
                base_channels * 8,
                base_channels * 8,
                3,
                2,
                1,
            )?)
        };
        let c3_n4 = CspBottleneckBlock::demo(
            base_channels * 16,
            base_channels * 16,
            neck_depth,
            0.5,
            false,
            depthwise,
        )?;

        Ok(Self {
            lateral_conv0,
            c3_p4,
            reduce_conv1,
            c3_p3,
            bu_conv2,
            c3_n3,
            bu_conv1,
            c3_n4,
        })
    }

    pub fn forward_cpu(&self, features: &DarknetFeatures) -> Result<PafpnFeatures> {
        let (fpn_out0_shape, fpn_out0) = self
            .lateral_conv0
            .forward_cpu(&features.f3, features.f3_shape)?;
        let (f_out0_up_shape, f_out0_up) = upsample_nearest_nchw(&fpn_out0, fpn_out0_shape, 2)?;
        let (cat_p4_shape, cat_p4) =
            concat_channels_nchw(&f_out0_up, f_out0_up_shape, &features.f2, features.f2_shape)?;
        let (f_out0_shape, f_out0) = self.c3_p4.forward_cpu(&cat_p4, cat_p4_shape)?;

        let (fpn_out1_shape, fpn_out1) = self.reduce_conv1.forward_cpu(&f_out0, f_out0_shape)?;
        let (f_out1_up_shape, f_out1_up) = upsample_nearest_nchw(&fpn_out1, fpn_out1_shape, 2)?;
        let (cat_p3_shape, cat_p3) =
            concat_channels_nchw(&f_out1_up, f_out1_up_shape, &features.f1, features.f1_shape)?;
        let (p3_shape, p3) = self.c3_p3.forward_cpu(&cat_p3, cat_p3_shape)?;

        let (p_out1_down_shape, p_out1_down) = self.bu_conv2.forward_cpu(&p3, p3_shape)?;
        let (cat_n3_shape, cat_n3) =
            concat_channels_nchw(&p_out1_down, p_out1_down_shape, &fpn_out1, fpn_out1_shape)?;
        let (p4_shape, p4) = self.c3_n3.forward_cpu(&cat_n3, cat_n3_shape)?;

        let (p_out0_down_shape, p_out0_down) = self.bu_conv1.forward_cpu(&p4, p4_shape)?;
        let (cat_n4_shape, cat_n4) =
            concat_channels_nchw(&p_out0_down, p_out0_down_shape, &fpn_out0, fpn_out0_shape)?;
        let (p5_shape, p5) = self.c3_n4.forward_cpu(&cat_n4, cat_n4_shape)?;

        Ok(PafpnFeatures {
            p3_shape,
            p3,
            p4_shape,
            p4,
            p5_shape,
            p5,
        })
    }
}

#[derive(Clone, Debug)]
pub struct HeadFeatures {
    pub s8_shape: TensorShape,
    pub s8: Vec<f32>,
    pub s16_shape: TensorShape,
    pub s16: Vec<f32>,
    pub s32_shape: TensorShape,
    pub s32: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct DecodedPredictions {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub class_id: usize,
    pub score: f32,
    pub objectness: f32,
    pub class_confidence: f32,
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct YoloxHeadScaleDemo {
    pub stem: BaseConvBlock,
    pub cls_conv1: ConvBlock,
    pub cls_conv2: ConvBlock,
    pub reg_conv1: ConvBlock,
    pub reg_conv2: ConvBlock,
    pub cls_pred: FusedConv2dWeights,
    pub reg_pred: FusedConv2dWeights,
    pub obj_pred: FusedConv2dWeights,
    pub num_classes: usize,
}

impl YoloxHeadScaleDemo {
    pub fn demo(
        in_channels: usize,
        hidden_channels: usize,
        num_classes: usize,
        depthwise: bool,
    ) -> Result<Self> {
        Ok(Self {
            stem: BaseConvBlock::demo(in_channels, hidden_channels, 1, 1, 0)?,
            cls_conv1: ConvBlock::demo(hidden_channels, hidden_channels, 3, 1, 1, depthwise)?,
            cls_conv2: ConvBlock::demo(hidden_channels, hidden_channels, 3, 1, 1, depthwise)?,
            reg_conv1: ConvBlock::demo(hidden_channels, hidden_channels, 3, 1, 1, depthwise)?,
            reg_conv2: ConvBlock::demo(hidden_channels, hidden_channels, 3, 1, 1, depthwise)?,
            cls_pred: demo_linear_conv(hidden_channels, num_classes, 1, 1, 0)?,
            reg_pred: demo_linear_conv(hidden_channels, 4, 1, 1, 0)?,
            obj_pred: demo_linear_conv(hidden_channels, 1, 1, 1, 0)?,
            num_classes,
        })
    }

    pub fn forward_cpu(
        &self,
        input: &[f32],
        input_shape: TensorShape,
    ) -> Result<(TensorShape, Vec<f32>)> {
        let (stem_shape, stem) = self.stem.forward_cpu(input, input_shape)?;

        let (cls_shape, cls) = self.cls_conv1.forward_cpu(&stem, stem_shape)?;
        let (cls_shape, cls) = self.cls_conv2.forward_cpu(&cls, cls_shape)?;
        let (cls_pred_shape, cls_pred) = forward_linear_conv(&self.cls_pred, &cls, cls_shape)?;

        let (reg_shape, reg) = self.reg_conv1.forward_cpu(&stem, stem_shape)?;
        let (reg_shape, reg) = self.reg_conv2.forward_cpu(&reg, reg_shape)?;
        let (reg_pred_shape, reg_pred) = forward_linear_conv(&self.reg_pred, &reg, reg_shape)?;
        let (obj_pred_shape, obj_pred) = forward_linear_conv(&self.obj_pred, &reg, reg_shape)?;

        let (reg_obj_shape, reg_obj) =
            concat_channels_nchw(&reg_pred, reg_pred_shape, &obj_pred, obj_pred_shape)?;
        let (output_shape, output) =
            concat_channels_nchw(&reg_obj, reg_obj_shape, &cls_pred, cls_pred_shape)?;

        if output_shape.channels != self.num_classes + 5 {
            bail!(
                "head output inválida: esperado C={} recebido C={}",
                self.num_classes + 5,
                output_shape.channels
            );
        }

        Ok((output_shape, output))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct YoloxHeadDemo {
    pub head_s8: YoloxHeadScaleDemo,
    pub head_s16: YoloxHeadScaleDemo,
    pub head_s32: YoloxHeadScaleDemo,
}

impl YoloxHeadDemo {
    pub fn demo(base_channels: usize, num_classes: usize, depthwise: bool) -> Result<Self> {
        let hidden_channels = base_channels * 4;

        Ok(Self {
            head_s8: YoloxHeadScaleDemo::demo(
                base_channels * 4,
                hidden_channels,
                num_classes,
                depthwise,
            )?,
            head_s16: YoloxHeadScaleDemo::demo(
                base_channels * 8,
                hidden_channels,
                num_classes,
                depthwise,
            )?,
            head_s32: YoloxHeadScaleDemo::demo(
                base_channels * 16,
                hidden_channels,
                num_classes,
                depthwise,
            )?,
        })
    }

    pub fn forward_cpu(&self, features: &PafpnFeatures) -> Result<HeadFeatures> {
        let (s8_shape, s8) = self.head_s8.forward_cpu(&features.p3, features.p3_shape)?;
        let (s16_shape, s16) = self.head_s16.forward_cpu(&features.p4, features.p4_shape)?;
        let (s32_shape, s32) = self.head_s32.forward_cpu(&features.p5, features.p5_shape)?;

        Ok(HeadFeatures {
            s8_shape,
            s8,
            s16_shape,
            s16,
            s32_shape,
            s32,
        })
    }
}

#[derive(Clone, Debug)]
pub struct YoloxDecodeDemo {
    pub num_classes: usize,
}

impl YoloxDecodeDemo {
    pub const fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }

    pub fn forward_cpu(&self, features: &HeadFeatures) -> Result<DecodedPredictions> {
        let s8 = decode_head_scale(&features.s8, features.s8_shape, 8, self.num_classes)?;
        let s16 = decode_head_scale(&features.s16, features.s16_shape, 16, self.num_classes)?;
        let s32 = decode_head_scale(&features.s32, features.s32_shape, 32, self.num_classes)?;

        let s8_s16 = concat_decoded_rows(&s8, &s16)?;
        concat_decoded_rows(&s8_s16, &s32)
    }
}

#[derive(Clone, Debug)]
pub struct YoloxPostprocessDemo {
    pub num_classes: usize,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub max_detections: usize,
}

impl YoloxPostprocessDemo {
    pub const fn new(
        num_classes: usize,
        confidence_threshold: f32,
        nms_threshold: f32,
        max_detections: usize,
    ) -> Self {
        Self {
            num_classes,
            confidence_threshold,
            nms_threshold,
            max_detections,
        }
    }

    pub fn forward_cpu(&self, decoded: &DecodedPredictions) -> Result<Vec<Detection>> {
        if decoded.cols != self.num_classes + 5 {
            bail!(
                "postprocess incompatível: esperado cols={} recebido cols={}",
                self.num_classes + 5,
                decoded.cols
            );
        }
        if decoded.data.len() != decoded.rows * decoded.cols {
            bail!(
                "postprocess recebeu buffer inválido: esperado {} elementos, recebido {}",
                decoded.rows * decoded.cols,
                decoded.data.len()
            );
        }

        let mut candidates = Vec::new();
        for row in 0..decoded.rows {
            let base = row * decoded.cols;
            let objectness = decoded.data[base + 4];

            let mut best_class = 0usize;
            let mut best_class_confidence = 0.0f32;
            for class_id in 0..self.num_classes {
                let class_confidence = decoded.data[base + 5 + class_id];
                if class_confidence > best_class_confidence {
                    best_class_confidence = class_confidence;
                    best_class = class_id;
                }
            }

            let score = objectness * best_class_confidence;
            if score < self.confidence_threshold {
                continue;
            }

            let cx = decoded.data[base];
            let cy = decoded.data[base + 1];
            let w = decoded.data[base + 2];
            let h = decoded.data[base + 3];

            candidates.push(Detection {
                class_id: best_class,
                score,
                objectness,
                class_confidence: best_class_confidence,
                x0: cx - w * 0.5,
                y0: cy - h * 0.5,
                x1: cx + w * 0.5,
                y1: cy + h * 0.5,
            });
        }

        candidates.sort_by(|lhs, rhs| rhs.score.partial_cmp(&lhs.score).unwrap_or(Ordering::Equal));

        let mut selected: Vec<Detection> = Vec::new();
        'candidate: for candidate in candidates {
            for existing in &selected {
                if existing.class_id != candidate.class_id {
                    continue;
                }
                if detection_iou(existing, &candidate) > self.nms_threshold {
                    continue 'candidate;
                }
            }

            selected.push(candidate);
            if selected.len() >= self.max_detections {
                break;
            }
        }

        Ok(selected)
    }
}

fn demo_linear_conv(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<FusedConv2dWeights> {
    let spec = Conv2dSpec::new(
        in_channels,
        out_channels,
        kernel,
        kernel,
        stride,
        stride,
        padding,
        padding,
    );
    let raw = RawConv2dWeights::demo(&spec);
    fuse_conv2d_bn(&raw, &BatchNorm1d::identity(out_channels))
}

fn forward_linear_conv(
    conv: &FusedConv2dWeights,
    input: &[f32],
    input_shape: TensorShape,
) -> Result<(TensorShape, Vec<f32>)> {
    let (output_h, output_w) = conv.spec.output_hw(input_shape.height, input_shape.width)?;
    let output = conv.convolve_nchw_f32(input, input_shape.height, input_shape.width)?;
    Ok((
        TensorShape::new(conv.spec.out_channels, output_h, output_w),
        output,
    ))
}

fn decode_head_scale(
    input: &[f32],
    shape: TensorShape,
    stride: usize,
    num_classes: usize,
) -> Result<DecodedPredictions> {
    let cols = num_classes + 5;
    if shape.channels != cols {
        bail!(
            "decode incompatível: esperado C={} recebido C={}",
            cols,
            shape.channels
        );
    }
    if input.len() != shape.len() {
        bail!(
            "decode recebeu buffer inválido: esperado {} elementos, recebido {}",
            shape.len(),
            input.len()
        );
    }

    let rows = shape.height * shape.width;
    let mut output = vec![0.0; rows * cols];

    for y in 0..shape.height {
        for x in 0..shape.width {
            let row = y * shape.width + x;
            let base = row * cols;
            let load = |channel: usize| -> f32 {
                let idx = ((channel * shape.height + y) * shape.width) + x;
                input[idx]
            };

            output[base] = (sigmoid_scalar(load(0)) + x as f32) * stride as f32;
            output[base + 1] = (sigmoid_scalar(load(1)) + y as f32) * stride as f32;
            output[base + 2] = load(2).clamp(-16.0, 16.0).exp() * stride as f32;
            output[base + 3] = load(3).clamp(-16.0, 16.0).exp() * stride as f32;
            output[base + 4] = sigmoid_scalar(load(4));

            for cls in 0..num_classes {
                output[base + 5 + cls] = sigmoid_scalar(load(5 + cls));
            }
        }
    }

    Ok(DecodedPredictions {
        rows,
        cols,
        data: output,
    })
}

fn concat_decoded_rows(
    lhs: &DecodedPredictions,
    rhs: &DecodedPredictions,
) -> Result<DecodedPredictions> {
    if lhs.cols != rhs.cols {
        bail!(
            "concat decoded incompatível: lhs.cols={} rhs.cols={}",
            lhs.cols,
            rhs.cols
        );
    }

    let mut data = Vec::with_capacity(lhs.data.len() + rhs.data.len());
    data.extend_from_slice(&lhs.data);
    data.extend_from_slice(&rhs.data);

    Ok(DecodedPredictions {
        rows: lhs.rows + rhs.rows,
        cols: lhs.cols,
        data,
    })
}

fn detection_iou(lhs: &Detection, rhs: &Detection) -> f32 {
    let inter_x0 = lhs.x0.max(rhs.x0);
    let inter_y0 = lhs.y0.max(rhs.y0);
    let inter_x1 = lhs.x1.min(rhs.x1);
    let inter_y1 = lhs.y1.min(rhs.y1);

    let inter_w = (inter_x1 - inter_x0).max(0.0);
    let inter_h = (inter_y1 - inter_y0).max(0.0);
    let inter_area = inter_w * inter_h;

    let lhs_area = (lhs.x1 - lhs.x0).max(0.0) * (lhs.y1 - lhs.y0).max(0.0);
    let rhs_area = (rhs.x1 - rhs.x0).max(0.0) * (rhs.y1 - rhs.y0).max(0.0);
    let union_area = lhs_area + rhs_area - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}
