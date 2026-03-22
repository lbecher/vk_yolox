use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::Path};

use crate::{
    fused_weights::FusedConv2dWeights,
    yolox_blocks::{CspDarknetDemo, YoloxHeadDemo, YoloxPafpnDemo},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DemoModelBundleMeta {
    pub base_channels: usize,
    pub base_depth: usize,
    pub num_classes: usize,
    pub depthwise: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DemoModelBundle {
    pub meta: DemoModelBundleMeta,
    pub backbone: CspDarknetDemo,
    pub pafpn: YoloxPafpnDemo,
    pub head: YoloxHeadDemo,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NamedConvLayer {
    pub name: String,
    pub spec: crate::fused_weights::Conv2dSpec,
    pub parameter_bytes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NamedLayerWeights {
    pub name: String,
    pub weights: FusedConv2dWeights,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BundleWeightPatch {
    pub meta: DemoModelBundleMeta,
    pub layers: Vec<NamedLayerWeights>,
}

impl DemoModelBundle {
    pub fn demo(
        base_channels: usize,
        base_depth: usize,
        num_classes: usize,
        depthwise: bool,
    ) -> Result<Self> {
        if base_channels == 0 || base_depth == 0 || num_classes == 0 {
            bail!("base_channels, base_depth e num_classes devem ser maiores que zero");
        }

        Ok(Self {
            meta: DemoModelBundleMeta {
                base_channels,
                base_depth,
                num_classes,
                depthwise,
            },
            backbone: CspDarknetDemo::demo(base_channels, base_depth, depthwise)?,
            pafpn: YoloxPafpnDemo::demo(base_channels, base_depth, depthwise)?,
            head: YoloxHeadDemo::demo(base_channels, num_classes, depthwise)?,
        })
    }

    pub fn save_json(&self, path: &Path) -> Result<()> {
        let serialized =
            serde_json::to_string_pretty(self).context("falha ao serializar o bundle")?;
        fs::write(path, serialized)
            .with_context(|| format!("falha ao escrever bundle em {}", path.display()))
    }

    pub fn load_json(path: &Path) -> Result<Self> {
        let serialized = fs::read_to_string(path)
            .with_context(|| format!("falha ao ler bundle {}", path.display()))?;
        let bundle: Self =
            serde_json::from_str(&serialized).context("falha ao desserializar o bundle")?;
        bundle.validate()?;
        Ok(bundle)
    }

    pub fn validate(&self) -> Result<()> {
        if self.meta.base_channels == 0 || self.meta.base_depth == 0 || self.meta.num_classes == 0 {
            bail!("bundle inválido: metadados com zero");
        }

        let hidden_channels = self.meta.base_channels * 4;
        if self.head.head_s8.num_classes != self.meta.num_classes
            || self.head.head_s16.num_classes != self.meta.num_classes
            || self.head.head_s32.num_classes != self.meta.num_classes
        {
            bail!("bundle inválido: num_classes da head não coincide com os metadados");
        }
        if self.head.head_s8.stem.conv.spec.out_channels != hidden_channels
            || self.head.head_s16.stem.conv.spec.out_channels != hidden_channels
            || self.head.head_s32.stem.conv.spec.out_channels != hidden_channels
        {
            bail!("bundle inválido: hidden_channels da head não coincide com base_channels");
        }

        Ok(())
    }

    pub fn parameter_bytes(&self) -> usize {
        fn fused_bytes(conv: &crate::fused_weights::FusedConv2dWeights) -> usize {
            (conv.weights.len() + conv.bias.len()) * std::mem::size_of::<f32>()
        }

        fn base(block: &crate::yolox_blocks::BaseConvBlock) -> usize {
            fused_bytes(&block.conv)
        }

        fn dws(block: &crate::yolox_blocks::DwsConvBlock) -> usize {
            fused_bytes(&block.depthwise) + fused_bytes(&block.pointwise)
        }

        fn conv(block: &crate::yolox_blocks::ConvBlock) -> usize {
            match block {
                crate::yolox_blocks::ConvBlock::Base(block) => base(block),
                crate::yolox_blocks::ConvBlock::Dws(block) => dws(block),
            }
        }

        fn bottleneck(block: &crate::yolox_blocks::BottleneckBlock) -> usize {
            base(&block.conv1) + conv(&block.conv2)
        }

        fn csp_bottleneck(block: &crate::yolox_blocks::CspBottleneckBlock) -> usize {
            base(&block.conv1)
                + base(&block.conv2)
                + base(&block.conv3)
                + block.blocks.iter().map(bottleneck).sum::<usize>()
        }

        fn csp_stage(block: &crate::yolox_blocks::CspStageBlock) -> usize {
            conv(&block.conv) + csp_bottleneck(&block.c3)
        }

        fn spp(block: &crate::yolox_blocks::SppBottleneckBlock) -> usize {
            base(&block.conv1) + base(&block.conv2)
        }

        fn dark5(block: &crate::yolox_blocks::Dark5Block) -> usize {
            conv(&block.conv) + spp(&block.spp) + csp_bottleneck(&block.c3)
        }

        fn head_scale(block: &crate::yolox_blocks::YoloxHeadScaleDemo) -> usize {
            base(&block.stem)
                + conv(&block.cls_conv1)
                + conv(&block.cls_conv2)
                + conv(&block.reg_conv1)
                + conv(&block.reg_conv2)
                + fused_bytes(&block.cls_pred)
                + fused_bytes(&block.reg_pred)
                + fused_bytes(&block.obj_pred)
        }

        base(&self.backbone.stem.conv)
            + csp_stage(&self.backbone.dark2)
            + csp_stage(&self.backbone.dark3)
            + csp_stage(&self.backbone.dark4)
            + dark5(&self.backbone.dark5)
            + base(&self.pafpn.lateral_conv0)
            + csp_bottleneck(&self.pafpn.c3_p4)
            + base(&self.pafpn.reduce_conv1)
            + csp_bottleneck(&self.pafpn.c3_p3)
            + conv(&self.pafpn.bu_conv2)
            + csp_bottleneck(&self.pafpn.c3_n3)
            + conv(&self.pafpn.bu_conv1)
            + csp_bottleneck(&self.pafpn.c3_n4)
            + head_scale(&self.head.head_s8)
            + head_scale(&self.head.head_s16)
            + head_scale(&self.head.head_s32)
    }

    pub fn named_layers(&self) -> Vec<NamedConvLayer> {
        fn fused_bytes(conv: &crate::fused_weights::FusedConv2dWeights) -> usize {
            (conv.weights.len() + conv.bias.len()) * std::mem::size_of::<f32>()
        }

        fn push_base(
            layers: &mut Vec<NamedConvLayer>,
            name: &str,
            block: &crate::yolox_blocks::BaseConvBlock,
        ) {
            layers.push(NamedConvLayer {
                name: name.to_string(),
                spec: block.conv.spec,
                parameter_bytes: fused_bytes(&block.conv),
            });
        }

        fn push_conv(
            layers: &mut Vec<NamedConvLayer>,
            name: &str,
            block: &crate::yolox_blocks::ConvBlock,
        ) {
            match block {
                crate::yolox_blocks::ConvBlock::Base(block) => push_base(layers, name, block),
                crate::yolox_blocks::ConvBlock::Dws(block) => {
                    layers.push(NamedConvLayer {
                        name: format!("{name}.depthwise"),
                        spec: block.depthwise.spec,
                        parameter_bytes: fused_bytes(&block.depthwise),
                    });
                    layers.push(NamedConvLayer {
                        name: format!("{name}.pointwise"),
                        spec: block.pointwise.spec,
                        parameter_bytes: fused_bytes(&block.pointwise),
                    });
                }
            }
        }

        fn push_bottleneck(
            layers: &mut Vec<NamedConvLayer>,
            name: &str,
            block: &crate::yolox_blocks::BottleneckBlock,
        ) {
            push_base(layers, &format!("{name}.conv1"), &block.conv1);
            push_conv(layers, &format!("{name}.conv2"), &block.conv2);
        }

        fn push_csp_bottleneck(
            layers: &mut Vec<NamedConvLayer>,
            name: &str,
            block: &crate::yolox_blocks::CspBottleneckBlock,
        ) {
            push_base(layers, &format!("{name}.conv1"), &block.conv1);
            push_base(layers, &format!("{name}.conv2"), &block.conv2);
            push_base(layers, &format!("{name}.conv3"), &block.conv3);
            for (index, block) in block.blocks.iter().enumerate() {
                push_bottleneck(layers, &format!("{name}.blocks.{index}"), block);
            }
        }

        fn push_csp_stage(
            layers: &mut Vec<NamedConvLayer>,
            name: &str,
            block: &crate::yolox_blocks::CspStageBlock,
        ) {
            push_conv(layers, &format!("{name}.conv"), &block.conv);
            push_csp_bottleneck(layers, &format!("{name}.c3"), &block.c3);
        }

        fn push_spp(
            layers: &mut Vec<NamedConvLayer>,
            name: &str,
            block: &crate::yolox_blocks::SppBottleneckBlock,
        ) {
            push_base(layers, &format!("{name}.conv1"), &block.conv1);
            push_base(layers, &format!("{name}.conv2"), &block.conv2);
        }

        fn push_dark5(
            layers: &mut Vec<NamedConvLayer>,
            name: &str,
            block: &crate::yolox_blocks::Dark5Block,
        ) {
            push_conv(layers, &format!("{name}.conv"), &block.conv);
            push_spp(layers, &format!("{name}.spp"), &block.spp);
            push_csp_bottleneck(layers, &format!("{name}.c3"), &block.c3);
        }

        fn push_head_scale(
            layers: &mut Vec<NamedConvLayer>,
            name: &str,
            block: &crate::yolox_blocks::YoloxHeadScaleDemo,
        ) {
            push_base(layers, &format!("{name}.stem"), &block.stem);
            push_conv(layers, &format!("{name}.cls_conv1"), &block.cls_conv1);
            push_conv(layers, &format!("{name}.cls_conv2"), &block.cls_conv2);
            push_conv(layers, &format!("{name}.reg_conv1"), &block.reg_conv1);
            push_conv(layers, &format!("{name}.reg_conv2"), &block.reg_conv2);
            layers.push(NamedConvLayer {
                name: format!("{name}.cls_pred"),
                spec: block.cls_pred.spec,
                parameter_bytes: fused_bytes(&block.cls_pred),
            });
            layers.push(NamedConvLayer {
                name: format!("{name}.reg_pred"),
                spec: block.reg_pred.spec,
                parameter_bytes: fused_bytes(&block.reg_pred),
            });
            layers.push(NamedConvLayer {
                name: format!("{name}.obj_pred"),
                spec: block.obj_pred.spec,
                parameter_bytes: fused_bytes(&block.obj_pred),
            });
        }

        let mut layers = Vec::new();
        push_base(&mut layers, "backbone.stem.conv", &self.backbone.stem.conv);
        push_csp_stage(&mut layers, "backbone.dark2", &self.backbone.dark2);
        push_csp_stage(&mut layers, "backbone.dark3", &self.backbone.dark3);
        push_csp_stage(&mut layers, "backbone.dark4", &self.backbone.dark4);
        push_dark5(&mut layers, "backbone.dark5", &self.backbone.dark5);

        push_base(
            &mut layers,
            "pafpn.lateral_conv0",
            &self.pafpn.lateral_conv0,
        );
        push_csp_bottleneck(&mut layers, "pafpn.c3_p4", &self.pafpn.c3_p4);
        push_base(&mut layers, "pafpn.reduce_conv1", &self.pafpn.reduce_conv1);
        push_csp_bottleneck(&mut layers, "pafpn.c3_p3", &self.pafpn.c3_p3);
        push_conv(&mut layers, "pafpn.bu_conv2", &self.pafpn.bu_conv2);
        push_csp_bottleneck(&mut layers, "pafpn.c3_n3", &self.pafpn.c3_n3);
        push_conv(&mut layers, "pafpn.bu_conv1", &self.pafpn.bu_conv1);
        push_csp_bottleneck(&mut layers, "pafpn.c3_n4", &self.pafpn.c3_n4);

        push_head_scale(&mut layers, "head.s8", &self.head.head_s8);
        push_head_scale(&mut layers, "head.s16", &self.head.head_s16);
        push_head_scale(&mut layers, "head.s32", &self.head.head_s32);
        layers
    }

    pub fn save_manifest_json(&self, path: &Path) -> Result<()> {
        let serialized = serde_json::to_string_pretty(&self.named_layers())
            .context("falha ao serializar o manifesto de camadas")?;
        fs::write(path, serialized)
            .with_context(|| format!("falha ao escrever manifesto em {}", path.display()))
    }

    pub fn export_weight_patch(&self) -> BundleWeightPatch {
        let mut layers = Vec::new();
        self.for_each_named_fused_conv(|name, weights| {
            layers.push(NamedLayerWeights {
                name,
                weights: weights.clone(),
            });
            Ok(())
        })
        .expect("for_each_named_fused_conv não deve falhar durante export");

        BundleWeightPatch {
            meta: self.meta.clone(),
            layers,
        }
    }

    pub fn save_weight_patch_json(&self, path: &Path) -> Result<()> {
        let serialized = serde_json::to_string_pretty(&self.export_weight_patch())
            .context("falha ao serializar patch de pesos")?;
        fs::write(path, serialized)
            .with_context(|| format!("falha ao escrever patch de pesos em {}", path.display()))
    }

    pub fn save_weight_directory(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)
            .with_context(|| format!("falha ao criar diretório {}", dir.display()))?;
        self.save_manifest_json(&dir.join("manifest.json"))?;

        for layer in self.export_weight_patch().layers {
            let stem = layer_file_stem(&layer.name);
            write_f32_bin(
                &dir.join(format!("{stem}.weights.bin")),
                &layer.weights.weights,
            )?;
            write_f32_bin(&dir.join(format!("{stem}.bias.bin")), &layer.weights.bias)?;
        }

        Ok(())
    }

    pub fn apply_weight_patch(&mut self, patch: &BundleWeightPatch) -> Result<()> {
        if patch.meta.num_classes != self.meta.num_classes {
            bail!(
                "patch incompatível: num_classes={} bundle={}",
                patch.meta.num_classes,
                self.meta.num_classes
            );
        }

        let mut patch_by_name = patch
            .layers
            .iter()
            .map(|layer| (layer.name.as_str(), &layer.weights))
            .collect::<HashMap<_, _>>();
        let expected = self.named_layers();

        for layer in &expected {
            let Some(patched) = patch_by_name.remove(layer.name.as_str()) else {
                bail!("patch incompleto: camada `{}` ausente", layer.name);
            };
            if patched.spec.in_channels != layer.spec.in_channels
                || patched.spec.out_channels != layer.spec.out_channels
                || patched.spec.groups != layer.spec.groups
                || patched.spec.kernel_h != layer.spec.kernel_h
                || patched.spec.kernel_w != layer.spec.kernel_w
                || patched.spec.stride_h != layer.spec.stride_h
                || patched.spec.stride_w != layer.spec.stride_w
                || patched.spec.pad_h != layer.spec.pad_h
                || patched.spec.pad_w != layer.spec.pad_w
            {
                bail!("patch incompatível com a camada `{}`", layer.name);
            }
        }

        if let Some(extra) = patch_by_name.keys().next() {
            bail!("patch contém camada extra não reconhecida: `{extra}`");
        }

        let mut owned = patch
            .layers
            .iter()
            .map(|layer| (layer.name.clone(), layer.weights.clone()))
            .collect::<HashMap<_, _>>();

        self.for_each_named_fused_conv_mut(|name, weights| {
            let patched = owned
                .remove(&name)
                .ok_or_else(|| anyhow::anyhow!("patch interno ausente para `{name}`"))?;
            *weights = patched;
            Ok(())
        })?;

        self.validate()
    }

    pub fn load_weight_patch_from_directory(&self, dir: &Path) -> Result<BundleWeightPatch> {
        let manifest_path = dir.join("manifest.json");
        if manifest_path.exists() {
            let serialized = fs::read_to_string(&manifest_path).with_context(|| {
                format!(
                    "falha ao ler manifesto de pesos {}",
                    manifest_path.display()
                )
            })?;
            let manifest: Vec<NamedConvLayer> = serde_json::from_str(&serialized)
                .context("falha ao desserializar manifesto de pesos")?;

            let expected = self.named_layers();
            if manifest.len() != expected.len() {
                bail!(
                    "manifesto incompatível: esperado {} camadas, recebido {}",
                    expected.len(),
                    manifest.len()
                );
            }
            for (expected, received) in expected.iter().zip(&manifest) {
                if expected.name != received.name {
                    bail!(
                        "manifesto incompatível: esperado camada `{}` recebido `{}`",
                        expected.name,
                        received.name
                    );
                }
            }
        }

        let mut layers = Vec::new();
        for layer in self.named_layers() {
            let stem = layer_file_stem(&layer.name);
            let weights_path = dir.join(format!("{stem}.weights.bin"));
            let bias_path = dir.join(format!("{stem}.bias.bin"));
            let weights = read_f32_bin(&weights_path)
                .with_context(|| format!("falha ao ler pesos de `{}`", layer.name))?;
            let bias = read_f32_bin(&bias_path)
                .with_context(|| format!("falha ao ler bias de `{}`", layer.name))?;

            let expected_weights = layer.spec.weight_len();
            let expected_bias = layer.spec.out_channels;
            if weights.len() != expected_weights || bias.len() != expected_bias {
                bail!(
                    "arquivos incompatíveis para `{}`: weights={} bias={} esperado weights={} bias={}",
                    layer.name,
                    weights.len(),
                    bias.len(),
                    expected_weights,
                    expected_bias
                );
            }

            layers.push(NamedLayerWeights {
                name: layer.name,
                weights: FusedConv2dWeights {
                    spec: layer.spec,
                    weights,
                    bias,
                },
            });
        }

        Ok(BundleWeightPatch {
            meta: self.meta.clone(),
            layers,
        })
    }

    fn for_each_named_fused_conv(
        &self,
        mut f: impl FnMut(String, &FusedConv2dWeights) -> Result<()>,
    ) -> Result<()> {
        fn visit_conv(
            f: &mut impl FnMut(String, &FusedConv2dWeights) -> Result<()>,
            name: String,
            block: &crate::yolox_blocks::ConvBlock,
        ) -> Result<()> {
            match block {
                crate::yolox_blocks::ConvBlock::Base(block) => f(name, &block.conv),
                crate::yolox_blocks::ConvBlock::Dws(block) => {
                    f(format!("{name}.depthwise"), &block.depthwise)?;
                    f(format!("{name}.pointwise"), &block.pointwise)
                }
            }
        }

        fn visit_bottleneck(
            f: &mut impl FnMut(String, &FusedConv2dWeights) -> Result<()>,
            name: String,
            block: &crate::yolox_blocks::BottleneckBlock,
        ) -> Result<()> {
            f(format!("{name}.conv1"), &block.conv1.conv)?;
            visit_conv(f, format!("{name}.conv2"), &block.conv2)
        }

        fn visit_csp_bottleneck(
            f: &mut impl FnMut(String, &FusedConv2dWeights) -> Result<()>,
            name: String,
            block: &crate::yolox_blocks::CspBottleneckBlock,
        ) -> Result<()> {
            f(format!("{name}.conv1"), &block.conv1.conv)?;
            f(format!("{name}.conv2"), &block.conv2.conv)?;
            f(format!("{name}.conv3"), &block.conv3.conv)?;
            for (index, block) in block.blocks.iter().enumerate() {
                visit_bottleneck(f, format!("{name}.blocks.{index}"), block)?;
            }
            Ok(())
        }

        f(
            "backbone.stem.conv".to_string(),
            &self.backbone.stem.conv.conv,
        )?;
        visit_conv(
            &mut f,
            "backbone.dark2.conv".to_string(),
            &self.backbone.dark2.conv,
        )?;
        visit_csp_bottleneck(
            &mut f,
            "backbone.dark2.c3".to_string(),
            &self.backbone.dark2.c3,
        )?;
        visit_conv(
            &mut f,
            "backbone.dark3.conv".to_string(),
            &self.backbone.dark3.conv,
        )?;
        visit_csp_bottleneck(
            &mut f,
            "backbone.dark3.c3".to_string(),
            &self.backbone.dark3.c3,
        )?;
        visit_conv(
            &mut f,
            "backbone.dark4.conv".to_string(),
            &self.backbone.dark4.conv,
        )?;
        visit_csp_bottleneck(
            &mut f,
            "backbone.dark4.c3".to_string(),
            &self.backbone.dark4.c3,
        )?;
        visit_conv(
            &mut f,
            "backbone.dark5.conv".to_string(),
            &self.backbone.dark5.conv,
        )?;
        f(
            "backbone.dark5.spp.conv1".to_string(),
            &self.backbone.dark5.spp.conv1.conv,
        )?;
        f(
            "backbone.dark5.spp.conv2".to_string(),
            &self.backbone.dark5.spp.conv2.conv,
        )?;
        visit_csp_bottleneck(
            &mut f,
            "backbone.dark5.c3".to_string(),
            &self.backbone.dark5.c3,
        )?;

        f(
            "pafpn.lateral_conv0".to_string(),
            &self.pafpn.lateral_conv0.conv,
        )?;
        visit_csp_bottleneck(&mut f, "pafpn.c3_p4".to_string(), &self.pafpn.c3_p4)?;
        f(
            "pafpn.reduce_conv1".to_string(),
            &self.pafpn.reduce_conv1.conv,
        )?;
        visit_csp_bottleneck(&mut f, "pafpn.c3_p3".to_string(), &self.pafpn.c3_p3)?;
        visit_conv(&mut f, "pafpn.bu_conv2".to_string(), &self.pafpn.bu_conv2)?;
        visit_csp_bottleneck(&mut f, "pafpn.c3_n3".to_string(), &self.pafpn.c3_n3)?;
        visit_conv(&mut f, "pafpn.bu_conv1".to_string(), &self.pafpn.bu_conv1)?;
        visit_csp_bottleneck(&mut f, "pafpn.c3_n4".to_string(), &self.pafpn.c3_n4)?;

        f("head.s8.stem".to_string(), &self.head.head_s8.stem.conv)?;
        visit_conv(
            &mut f,
            "head.s8.cls_conv1".to_string(),
            &self.head.head_s8.cls_conv1,
        )?;
        visit_conv(
            &mut f,
            "head.s8.cls_conv2".to_string(),
            &self.head.head_s8.cls_conv2,
        )?;
        visit_conv(
            &mut f,
            "head.s8.reg_conv1".to_string(),
            &self.head.head_s8.reg_conv1,
        )?;
        visit_conv(
            &mut f,
            "head.s8.reg_conv2".to_string(),
            &self.head.head_s8.reg_conv2,
        )?;
        f("head.s8.cls_pred".to_string(), &self.head.head_s8.cls_pred)?;
        f("head.s8.reg_pred".to_string(), &self.head.head_s8.reg_pred)?;
        f("head.s8.obj_pred".to_string(), &self.head.head_s8.obj_pred)?;

        f("head.s16.stem".to_string(), &self.head.head_s16.stem.conv)?;
        visit_conv(
            &mut f,
            "head.s16.cls_conv1".to_string(),
            &self.head.head_s16.cls_conv1,
        )?;
        visit_conv(
            &mut f,
            "head.s16.cls_conv2".to_string(),
            &self.head.head_s16.cls_conv2,
        )?;
        visit_conv(
            &mut f,
            "head.s16.reg_conv1".to_string(),
            &self.head.head_s16.reg_conv1,
        )?;
        visit_conv(
            &mut f,
            "head.s16.reg_conv2".to_string(),
            &self.head.head_s16.reg_conv2,
        )?;
        f(
            "head.s16.cls_pred".to_string(),
            &self.head.head_s16.cls_pred,
        )?;
        f(
            "head.s16.reg_pred".to_string(),
            &self.head.head_s16.reg_pred,
        )?;
        f(
            "head.s16.obj_pred".to_string(),
            &self.head.head_s16.obj_pred,
        )?;

        f("head.s32.stem".to_string(), &self.head.head_s32.stem.conv)?;
        visit_conv(
            &mut f,
            "head.s32.cls_conv1".to_string(),
            &self.head.head_s32.cls_conv1,
        )?;
        visit_conv(
            &mut f,
            "head.s32.cls_conv2".to_string(),
            &self.head.head_s32.cls_conv2,
        )?;
        visit_conv(
            &mut f,
            "head.s32.reg_conv1".to_string(),
            &self.head.head_s32.reg_conv1,
        )?;
        visit_conv(
            &mut f,
            "head.s32.reg_conv2".to_string(),
            &self.head.head_s32.reg_conv2,
        )?;
        f(
            "head.s32.cls_pred".to_string(),
            &self.head.head_s32.cls_pred,
        )?;
        f(
            "head.s32.reg_pred".to_string(),
            &self.head.head_s32.reg_pred,
        )?;
        f(
            "head.s32.obj_pred".to_string(),
            &self.head.head_s32.obj_pred,
        )?;
        Ok(())
    }

    fn for_each_named_fused_conv_mut(
        &mut self,
        mut f: impl FnMut(String, &mut FusedConv2dWeights) -> Result<()>,
    ) -> Result<()> {
        fn visit_conv(
            f: &mut impl FnMut(String, &mut FusedConv2dWeights) -> Result<()>,
            name: String,
            block: &mut crate::yolox_blocks::ConvBlock,
        ) -> Result<()> {
            match block {
                crate::yolox_blocks::ConvBlock::Base(block) => f(name, &mut block.conv),
                crate::yolox_blocks::ConvBlock::Dws(block) => {
                    f(format!("{name}.depthwise"), &mut block.depthwise)?;
                    f(format!("{name}.pointwise"), &mut block.pointwise)
                }
            }
        }

        fn visit_bottleneck(
            f: &mut impl FnMut(String, &mut FusedConv2dWeights) -> Result<()>,
            name: String,
            block: &mut crate::yolox_blocks::BottleneckBlock,
        ) -> Result<()> {
            f(format!("{name}.conv1"), &mut block.conv1.conv)?;
            visit_conv(f, format!("{name}.conv2"), &mut block.conv2)
        }

        fn visit_csp_bottleneck(
            f: &mut impl FnMut(String, &mut FusedConv2dWeights) -> Result<()>,
            name: String,
            block: &mut crate::yolox_blocks::CspBottleneckBlock,
        ) -> Result<()> {
            f(format!("{name}.conv1"), &mut block.conv1.conv)?;
            f(format!("{name}.conv2"), &mut block.conv2.conv)?;
            f(format!("{name}.conv3"), &mut block.conv3.conv)?;
            for (index, block) in block.blocks.iter_mut().enumerate() {
                visit_bottleneck(f, format!("{name}.blocks.{index}"), block)?;
            }
            Ok(())
        }

        f(
            "backbone.stem.conv".to_string(),
            &mut self.backbone.stem.conv.conv,
        )?;
        visit_conv(
            &mut f,
            "backbone.dark2.conv".to_string(),
            &mut self.backbone.dark2.conv,
        )?;
        visit_csp_bottleneck(
            &mut f,
            "backbone.dark2.c3".to_string(),
            &mut self.backbone.dark2.c3,
        )?;
        visit_conv(
            &mut f,
            "backbone.dark3.conv".to_string(),
            &mut self.backbone.dark3.conv,
        )?;
        visit_csp_bottleneck(
            &mut f,
            "backbone.dark3.c3".to_string(),
            &mut self.backbone.dark3.c3,
        )?;
        visit_conv(
            &mut f,
            "backbone.dark4.conv".to_string(),
            &mut self.backbone.dark4.conv,
        )?;
        visit_csp_bottleneck(
            &mut f,
            "backbone.dark4.c3".to_string(),
            &mut self.backbone.dark4.c3,
        )?;
        visit_conv(
            &mut f,
            "backbone.dark5.conv".to_string(),
            &mut self.backbone.dark5.conv,
        )?;
        f(
            "backbone.dark5.spp.conv1".to_string(),
            &mut self.backbone.dark5.spp.conv1.conv,
        )?;
        f(
            "backbone.dark5.spp.conv2".to_string(),
            &mut self.backbone.dark5.spp.conv2.conv,
        )?;
        visit_csp_bottleneck(
            &mut f,
            "backbone.dark5.c3".to_string(),
            &mut self.backbone.dark5.c3,
        )?;

        f(
            "pafpn.lateral_conv0".to_string(),
            &mut self.pafpn.lateral_conv0.conv,
        )?;
        visit_csp_bottleneck(&mut f, "pafpn.c3_p4".to_string(), &mut self.pafpn.c3_p4)?;
        f(
            "pafpn.reduce_conv1".to_string(),
            &mut self.pafpn.reduce_conv1.conv,
        )?;
        visit_csp_bottleneck(&mut f, "pafpn.c3_p3".to_string(), &mut self.pafpn.c3_p3)?;
        visit_conv(
            &mut f,
            "pafpn.bu_conv2".to_string(),
            &mut self.pafpn.bu_conv2,
        )?;
        visit_csp_bottleneck(&mut f, "pafpn.c3_n3".to_string(), &mut self.pafpn.c3_n3)?;
        visit_conv(
            &mut f,
            "pafpn.bu_conv1".to_string(),
            &mut self.pafpn.bu_conv1,
        )?;
        visit_csp_bottleneck(&mut f, "pafpn.c3_n4".to_string(), &mut self.pafpn.c3_n4)?;

        f("head.s8.stem".to_string(), &mut self.head.head_s8.stem.conv)?;
        visit_conv(
            &mut f,
            "head.s8.cls_conv1".to_string(),
            &mut self.head.head_s8.cls_conv1,
        )?;
        visit_conv(
            &mut f,
            "head.s8.cls_conv2".to_string(),
            &mut self.head.head_s8.cls_conv2,
        )?;
        visit_conv(
            &mut f,
            "head.s8.reg_conv1".to_string(),
            &mut self.head.head_s8.reg_conv1,
        )?;
        visit_conv(
            &mut f,
            "head.s8.reg_conv2".to_string(),
            &mut self.head.head_s8.reg_conv2,
        )?;
        f(
            "head.s8.cls_pred".to_string(),
            &mut self.head.head_s8.cls_pred,
        )?;
        f(
            "head.s8.reg_pred".to_string(),
            &mut self.head.head_s8.reg_pred,
        )?;
        f(
            "head.s8.obj_pred".to_string(),
            &mut self.head.head_s8.obj_pred,
        )?;

        f(
            "head.s16.stem".to_string(),
            &mut self.head.head_s16.stem.conv,
        )?;
        visit_conv(
            &mut f,
            "head.s16.cls_conv1".to_string(),
            &mut self.head.head_s16.cls_conv1,
        )?;
        visit_conv(
            &mut f,
            "head.s16.cls_conv2".to_string(),
            &mut self.head.head_s16.cls_conv2,
        )?;
        visit_conv(
            &mut f,
            "head.s16.reg_conv1".to_string(),
            &mut self.head.head_s16.reg_conv1,
        )?;
        visit_conv(
            &mut f,
            "head.s16.reg_conv2".to_string(),
            &mut self.head.head_s16.reg_conv2,
        )?;
        f(
            "head.s16.cls_pred".to_string(),
            &mut self.head.head_s16.cls_pred,
        )?;
        f(
            "head.s16.reg_pred".to_string(),
            &mut self.head.head_s16.reg_pred,
        )?;
        f(
            "head.s16.obj_pred".to_string(),
            &mut self.head.head_s16.obj_pred,
        )?;

        f(
            "head.s32.stem".to_string(),
            &mut self.head.head_s32.stem.conv,
        )?;
        visit_conv(
            &mut f,
            "head.s32.cls_conv1".to_string(),
            &mut self.head.head_s32.cls_conv1,
        )?;
        visit_conv(
            &mut f,
            "head.s32.cls_conv2".to_string(),
            &mut self.head.head_s32.cls_conv2,
        )?;
        visit_conv(
            &mut f,
            "head.s32.reg_conv1".to_string(),
            &mut self.head.head_s32.reg_conv1,
        )?;
        visit_conv(
            &mut f,
            "head.s32.reg_conv2".to_string(),
            &mut self.head.head_s32.reg_conv2,
        )?;
        f(
            "head.s32.cls_pred".to_string(),
            &mut self.head.head_s32.cls_pred,
        )?;
        f(
            "head.s32.reg_pred".to_string(),
            &mut self.head.head_s32.reg_pred,
        )?;
        f(
            "head.s32.obj_pred".to_string(),
            &mut self.head.head_s32.obj_pred,
        )?;
        Ok(())
    }

    pub fn summary(&self) -> String {
        format!(
            "bundle demo: base_channels={} base_depth={} num_classes={} depthwise={} layers={} pesos_fused={:.2} MiB",
            self.meta.base_channels,
            self.meta.base_depth,
            self.meta.num_classes,
            self.meta.depthwise,
            self.named_layers().len(),
            self.parameter_bytes() as f64 / (1024.0 * 1024.0)
        )
    }
}

fn layer_file_stem(name: &str) -> String {
    name.chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => ch,
            _ => '_',
        })
        .collect()
}

fn write_f32_bin(path: &Path, values: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).with_context(|| format!("falha ao escrever {}", path.display()))
}

fn read_f32_bin(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("falha ao ler {}", path.display()))?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        bail!(
            "arquivo binário inválido {}: tamanho {} não é múltiplo de 4",
            path.display(),
            bytes.len()
        );
    }

    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}
