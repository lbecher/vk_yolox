use clap::ValueEnum;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};

const BYTES_PER_F32: usize = std::mem::size_of::<f32>();

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum ModelVariant {
    Nano,
    Tiny,
    S,
    M,
    L,
    X,
}

impl ModelVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nano => "yolox-nano",
            Self::Tiny => "yolox-tiny",
            Self::S => "yolox-s",
            Self::M => "yolox-m",
            Self::L => "yolox-l",
            Self::X => "yolox-x",
        }
    }

    fn spec(self) -> ModelSpec {
        match self {
            Self::Nano => ModelSpec::new(0.33, 0.25, true),
            Self::Tiny => ModelSpec::new(0.33, 0.375, false),
            Self::S => ModelSpec::new(0.33, 0.50, false),
            Self::M => ModelSpec::new(0.67, 0.75, false),
            Self::L => ModelSpec::new(1.0, 1.0, false),
            Self::X => ModelSpec::new(1.33, 1.25, false),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ModelSpec {
    depth: f64,
    width: f64,
    depthwise: bool,
}

impl ModelSpec {
    const fn new(depth: f64, width: f64, depthwise: bool) -> Self {
        Self {
            depth,
            width,
            depthwise,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelPlan {
    pub model: ModelMetadata,
    pub parameters: ParameterSummary,
    pub memory: MemorySummary,
    pub primitives: Vec<PrimitiveRequirement>,
    pub execution: Vec<ExecutionNode>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelMetadata {
    pub variant: ModelVariant,
    pub input_shape: TensorShape,
    pub num_classes: usize,
    pub depth_multiplier: f64,
    pub width_multiplier: f64,
    pub depthwise: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ParameterSummary {
    pub raw_bytes: usize,
    pub fused_bytes: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct PrimitiveRequirement {
    pub primitive: String,
    pub count: usize,
    pub notes: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExecutionNode {
    pub id: String,
    pub name: String,
    pub from_burn: String,
    pub op: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub output_shape: TensorShape,
    pub output_bytes: usize,
    pub raw_parameter_bytes: usize,
    pub fused_parameter_bytes: usize,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemorySummary {
    pub peak_live_bytes: usize,
    pub peak_at: String,
    pub live_tensors_at_peak: Vec<String>,
    pub max_single_tensor_name: String,
    pub max_single_tensor_bytes: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct TensorShape {
    pub dims: Vec<usize>,
}

impl TensorShape {
    fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    fn nchw(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self::new(vec![n, c, h, w])
    }

    fn bytes(&self) -> usize {
        self.dims.iter().product::<usize>() * BYTES_PER_F32
    }

    pub fn display(&self) -> String {
        let dims: Vec<String> = self.dims.iter().map(|dim| dim.to_string()).collect();
        format!("[{}]", dims.join("x"))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ParamFootprint {
    raw_bytes: usize,
    fused_bytes: usize,
}

impl std::ops::Add for ParamFootprint {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            raw_bytes: self.raw_bytes + rhs.raw_bytes,
            fused_bytes: self.fused_bytes + rhs.fused_bytes,
        }
    }
}

impl std::ops::AddAssign for ParamFootprint {
    fn add_assign(&mut self, rhs: Self) {
        self.raw_bytes += rhs.raw_bytes;
        self.fused_bytes += rhs.fused_bytes;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum PrimitiveKind {
    Conv2d,
    DepthwiseConv2d,
    BatchNormFold,
    SiLU,
    Sigmoid,
    Exp,
    ElementwiseAdd,
    ElementwiseMultiply,
    MaxPool2d,
    UpsampleNearest,
    Concat,
    SliceStrided,
    Flatten,
    SwapDims,
    GridGeneration,
}

impl PrimitiveKind {
    fn label(self) -> &'static str {
        match self {
            Self::Conv2d => "conv2d",
            Self::DepthwiseConv2d => "depthwise-conv2d",
            Self::BatchNormFold => "batchnorm-fold",
            Self::SiLU => "silu",
            Self::Sigmoid => "sigmoid",
            Self::Exp => "exp",
            Self::ElementwiseAdd => "elementwise-add",
            Self::ElementwiseMultiply => "elementwise-multiply",
            Self::MaxPool2d => "maxpool2d",
            Self::UpsampleNearest => "upsample-nearest",
            Self::Concat => "concat",
            Self::SliceStrided => "slice-strided",
            Self::Flatten => "flatten",
            Self::SwapDims => "swap-dims",
            Self::GridGeneration => "grid-generation",
        }
    }

    fn note(self) -> &'static str {
        match self {
            Self::Conv2d => "kernel principal para convoluções normais 1x1 e 3x3",
            Self::DepthwiseConv2d => "necessário apenas para YOLOX-Nano e blocos depthwise",
            Self::BatchNormFold => {
                "fold offline recomendado para reduzir kernels em tempo de inferência"
            }
            Self::SiLU => "ativação usada após cada BaseConv",
            Self::Sigmoid => "aplicada nas heads de classe e objectness",
            Self::Exp => "decodifica largura e altura na head YOLOX",
            Self::ElementwiseAdd => "usado em atalhos residuais e na decodificação",
            Self::ElementwiseMultiply => "usado na decodificação e composição final de scores",
            Self::MaxPool2d => "usado no bloco SPP do backbone",
            Self::UpsampleNearest => "usado no caminho top-down do PAFPN",
            Self::Concat => "concatena caminhos laterais, SPP e saídas da head",
            Self::SliceStrided => "implementa o Focus sem copiar para CPU",
            Self::Flatten => "normaliza a saída por escala antes do concat global",
            Self::SwapDims => "transforma [B,C,N] em [B,N,C] para o decode",
            Self::GridGeneration => "gera grids e strides por escala para decode em GPU",
        }
    }
}

pub fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

pub fn build_model_plan(variant: ModelVariant, input_size: usize, num_classes: usize) -> ModelPlan {
    let spec = variant.spec();
    let base_channels = expand(64, spec.width);
    let base_depth = ((spec.depth * 3.0).round() as usize).max(1);
    let num_blocks = (3.0 * spec.depth).round() as usize;
    let head_hidden = expand(256, spec.width);

    let c3 = base_channels * 4;
    let c4 = base_channels * 8;
    let c5 = base_channels * 16;
    let s2 = input_size / 2;
    let s4 = input_size / 4;
    let s8 = input_size / 8;
    let s16 = input_size / 16;
    let s32 = input_size / 32;
    let total_anchors = s8 * s8 + s16 * s16 + s32 * s32;
    let per_anchor_outputs = 5 + num_classes;

    let mut primitives = BTreeMap::new();
    let mut nodes = Vec::new();

    add_node(
        &mut nodes,
        "00",
        "input_image",
        "yolox-burn/src/main.rs",
        "InputTensor",
        &[],
        TensorShape::nchw(1, 3, input_size, input_size),
        ParamFootprint::default(),
        &[],
    );

    let stem_fp = focus_footprint(3, base_channels, spec.depthwise, &mut primitives);
    add_node(
        &mut nodes,
        "01",
        "stem_focus",
        "yolox-burn/src/blocks.rs",
        "Focus",
        &["input_image"],
        TensorShape::nchw(1, base_channels, s2, s2),
        stem_fp,
        &["slice + concat + conv/bn/silu"],
    );

    let dark2_fp = csp_block_footprint(
        base_channels,
        base_channels * 2,
        base_depth,
        false,
        true,
        spec.depthwise,
        &mut primitives,
    );
    add_node(
        &mut nodes,
        "02",
        "dark2",
        "yolox-burn/src/darknet.rs",
        "CspBlock",
        &["stem_focus"],
        TensorShape::nchw(1, base_channels * 2, s4, s4),
        dark2_fp,
        &[],
    );

    let dark3_fp = csp_block_footprint(
        base_channels * 2,
        c3,
        base_depth * 3,
        false,
        true,
        spec.depthwise,
        &mut primitives,
    );
    add_node(
        &mut nodes,
        "03",
        "dark3",
        "yolox-burn/src/darknet.rs",
        "CspBlock",
        &["dark2"],
        TensorShape::nchw(1, c3, s8, s8),
        dark3_fp,
        &["feature map P3"],
    );

    let dark4_fp = csp_block_footprint(
        c3,
        c4,
        base_depth * 3,
        false,
        true,
        spec.depthwise,
        &mut primitives,
    );
    add_node(
        &mut nodes,
        "04",
        "dark4",
        "yolox-burn/src/darknet.rs",
        "CspBlock",
        &["dark3"],
        TensorShape::nchw(1, c4, s16, s16),
        dark4_fp,
        &["feature map P4"],
    );

    let dark5_fp = csp_block_footprint(
        c4,
        c5,
        base_depth,
        true,
        false,
        spec.depthwise,
        &mut primitives,
    );
    add_node(
        &mut nodes,
        "05",
        "dark5",
        "yolox-burn/src/darknet.rs",
        "CspBlock+SPP",
        &["dark4"],
        TensorShape::nchw(1, c5, s32, s32),
        dark5_fp,
        &["feature map P5"],
    );

    let lateral_fp = base_conv_footprint(c5, c4, 1, 1, 1, &mut primitives);
    add_node(
        &mut nodes,
        "06",
        "fpn_out0",
        "yolox-burn/src/pafpn.rs",
        "BaseConv",
        &["dark5"],
        TensorShape::nchw(1, c4, s32, s32),
        lateral_fp,
        &[],
    );

    bump(&mut primitives, PrimitiveKind::UpsampleNearest, 1);
    add_node(
        &mut nodes,
        "07",
        "fpn_out0_up",
        "yolox-burn/src/pafpn.rs",
        "UpsampleNearest",
        &["fpn_out0"],
        TensorShape::nchw(1, c4, s16, s16),
        ParamFootprint::default(),
        &[],
    );

    bump(&mut primitives, PrimitiveKind::Concat, 1);
    add_node(
        &mut nodes,
        "08",
        "cat_p4",
        "yolox-burn/src/pafpn.rs",
        "Concat",
        &["fpn_out0_up", "dark4"],
        TensorShape::nchw(1, c4 * 2, s16, s16),
        ParamFootprint::default(),
        &[],
    );

    let c3_p4_fp = csp_bottleneck_footprint(
        c4 * 2,
        c4,
        num_blocks,
        false,
        spec.depthwise,
        &mut primitives,
    );
    add_node(
        &mut nodes,
        "09",
        "f_out0",
        "yolox-burn/src/pafpn.rs",
        "CspBottleneck",
        &["cat_p4"],
        TensorShape::nchw(1, c4, s16, s16),
        c3_p4_fp,
        &[],
    );

    let reduce_fp = base_conv_footprint(c4, c3, 1, 1, 1, &mut primitives);
    add_node(
        &mut nodes,
        "10",
        "fpn_out1",
        "yolox-burn/src/pafpn.rs",
        "BaseConv",
        &["f_out0"],
        TensorShape::nchw(1, c3, s16, s16),
        reduce_fp,
        &[],
    );

    bump(&mut primitives, PrimitiveKind::UpsampleNearest, 1);
    add_node(
        &mut nodes,
        "11",
        "fpn_out1_up",
        "yolox-burn/src/pafpn.rs",
        "UpsampleNearest",
        &["fpn_out1"],
        TensorShape::nchw(1, c3, s8, s8),
        ParamFootprint::default(),
        &[],
    );

    bump(&mut primitives, PrimitiveKind::Concat, 1);
    add_node(
        &mut nodes,
        "12",
        "cat_p3",
        "yolox-burn/src/pafpn.rs",
        "Concat",
        &["fpn_out1_up", "dark3"],
        TensorShape::nchw(1, c3 * 2, s8, s8),
        ParamFootprint::default(),
        &[],
    );

    let c3_p3_fp = csp_bottleneck_footprint(
        c3 * 2,
        c3,
        num_blocks,
        false,
        spec.depthwise,
        &mut primitives,
    );
    add_node(
        &mut nodes,
        "13",
        "pan_out2",
        "yolox-burn/src/pafpn.rs",
        "CspBottleneck",
        &["cat_p3"],
        TensorShape::nchw(1, c3, s8, s8),
        c3_p3_fp,
        &["feature map de saída stride=8"],
    );

    let bu_conv2_fp = conv_module_footprint(c3, c3, 3, spec.depthwise, &mut primitives);
    add_node(
        &mut nodes,
        "14",
        "p_out1_down",
        "yolox-burn/src/pafpn.rs",
        "BottomUpConv",
        &["pan_out2"],
        TensorShape::nchw(1, c3, s16, s16),
        bu_conv2_fp,
        &[],
    );

    let head_s8_fp = head_scale_footprint(
        c3,
        head_hidden,
        num_classes,
        spec.depthwise,
        &mut primitives,
    );
    add_node(
        &mut nodes,
        "15",
        "head_s8",
        "yolox-burn/src/head.rs",
        "HeadScale",
        &["pan_out2"],
        TensorShape::new(vec![1, per_anchor_outputs, s8 * s8]),
        head_s8_fp,
        &[],
    );

    bump(&mut primitives, PrimitiveKind::Concat, 1);
    add_node(
        &mut nodes,
        "16",
        "cat_n3",
        "yolox-burn/src/pafpn.rs",
        "Concat",
        &["p_out1_down", "fpn_out1"],
        TensorShape::nchw(1, c3 * 2, s16, s16),
        ParamFootprint::default(),
        &[],
    );

    let c3_n3_fp = csp_bottleneck_footprint(
        c3 * 2,
        c4,
        num_blocks,
        false,
        spec.depthwise,
        &mut primitives,
    );
    add_node(
        &mut nodes,
        "17",
        "pan_out1",
        "yolox-burn/src/pafpn.rs",
        "CspBottleneck",
        &["cat_n3"],
        TensorShape::nchw(1, c4, s16, s16),
        c3_n3_fp,
        &["feature map de saída stride=16"],
    );

    let bu_conv1_fp = conv_module_footprint(c4, c4, 3, spec.depthwise, &mut primitives);
    add_node(
        &mut nodes,
        "18",
        "p_out0_down",
        "yolox-burn/src/pafpn.rs",
        "BottomUpConv",
        &["pan_out1"],
        TensorShape::nchw(1, c4, s32, s32),
        bu_conv1_fp,
        &[],
    );

    let head_s16_fp = head_scale_footprint(
        c4,
        head_hidden,
        num_classes,
        spec.depthwise,
        &mut primitives,
    );
    add_node(
        &mut nodes,
        "19",
        "head_s16",
        "yolox-burn/src/head.rs",
        "HeadScale",
        &["pan_out1"],
        TensorShape::new(vec![1, per_anchor_outputs, s16 * s16]),
        head_s16_fp,
        &[],
    );

    bump(&mut primitives, PrimitiveKind::Concat, 1);
    add_node(
        &mut nodes,
        "20",
        "cat_n4",
        "yolox-burn/src/pafpn.rs",
        "Concat",
        &["p_out0_down", "fpn_out0"],
        TensorShape::nchw(1, c4 * 2, s32, s32),
        ParamFootprint::default(),
        &[],
    );

    let c3_n4_fp = csp_bottleneck_footprint(
        c4 * 2,
        c5,
        num_blocks,
        false,
        spec.depthwise,
        &mut primitives,
    );
    add_node(
        &mut nodes,
        "21",
        "pan_out0",
        "yolox-burn/src/pafpn.rs",
        "CspBottleneck",
        &["cat_n4"],
        TensorShape::nchw(1, c5, s32, s32),
        c3_n4_fp,
        &["feature map de saída stride=32"],
    );

    let head_s32_fp = head_scale_footprint(
        c5,
        head_hidden,
        num_classes,
        spec.depthwise,
        &mut primitives,
    );
    add_node(
        &mut nodes,
        "22",
        "head_s32",
        "yolox-burn/src/head.rs",
        "HeadScale",
        &["pan_out0"],
        TensorShape::new(vec![1, per_anchor_outputs, s32 * s32]),
        head_s32_fp,
        &[],
    );

    bump(&mut primitives, PrimitiveKind::Concat, 1);
    add_node(
        &mut nodes,
        "23",
        "head_concat",
        "yolox-burn/src/head.rs",
        "ConcatOutputs",
        &["head_s8", "head_s16", "head_s32"],
        TensorShape::new(vec![1, per_anchor_outputs, total_anchors]),
        ParamFootprint::default(),
        &[],
    );

    bump(&mut primitives, PrimitiveKind::SwapDims, 1);
    bump(&mut primitives, PrimitiveKind::GridGeneration, 3);
    bump(&mut primitives, PrimitiveKind::Exp, 1);
    bump(&mut primitives, PrimitiveKind::ElementwiseAdd, 1);
    bump(&mut primitives, PrimitiveKind::ElementwiseMultiply, 2);
    bump(&mut primitives, PrimitiveKind::Concat, 1);
    add_node(
        &mut nodes,
        "24",
        "decode",
        "yolox-burn/src/head.rs",
        "DecodeHead",
        &["head_concat"],
        TensorShape::new(vec![1, total_anchors, per_anchor_outputs]),
        ParamFootprint::default(),
        &["inclui swap_dims, grid, add, mul, exp e concat final"],
    );

    let parameters = ParameterSummary {
        raw_bytes: nodes.iter().map(|node| node.raw_parameter_bytes).sum(),
        fused_bytes: nodes.iter().map(|node| node.fused_parameter_bytes).sum(),
    };

    let memory = estimate_memory(&nodes);
    let primitives = primitives
        .into_iter()
        .map(|(primitive, count)| PrimitiveRequirement {
            primitive: primitive.label().to_string(),
            count,
            notes: primitive.note().to_string(),
        })
        .collect();

    ModelPlan {
        model: ModelMetadata {
            variant,
            input_shape: TensorShape::nchw(1, 3, input_size, input_size),
            num_classes,
            depth_multiplier: spec.depth,
            width_multiplier: spec.width,
            depthwise: spec.depthwise,
        },
        parameters,
        memory,
        primitives,
        execution: nodes,
        notes: vec![
            "O plano assume pesos em f32 carregados uma vez em buffer device-local; staging só na inicialização.".to_string(),
            "BatchNorm deve ser foldado em convolução antes do upload para eliminar leituras extras e kernels dedicados.".to_string(),
            "O decode da head está incluído no plano GPU, mas o NMS continua fora porque a referência Burn faz essa etapa no host.".to_string(),
            "Para Vulkan 1.0, o caminho mais direto é compilar shaders SPIR-V por primitiva e encadear dispatches com buffers NCHW reutilizáveis.".to_string(),
            "A ordem das nodes foi levemente reescalonada em relação ao código Burn para reduzir pico de ativações sem alterar a semântica.".to_string(),
        ],
    }
}

fn add_node(
    nodes: &mut Vec<ExecutionNode>,
    id: &str,
    name: &str,
    from_burn: &str,
    op: &str,
    inputs: &[&str],
    output_shape: TensorShape,
    footprint: ParamFootprint,
    notes: &[&str],
) {
    nodes.push(ExecutionNode {
        id: id.to_string(),
        name: name.to_string(),
        from_burn: from_burn.to_string(),
        op: op.to_string(),
        inputs: inputs.iter().map(|input| input.to_string()).collect(),
        output: name.to_string(),
        output_bytes: output_shape.bytes(),
        output_shape,
        raw_parameter_bytes: footprint.raw_bytes,
        fused_parameter_bytes: footprint.fused_bytes,
        notes: notes.iter().map(|note| note.to_string()).collect(),
    });
}

fn estimate_memory(nodes: &[ExecutionNode]) -> MemorySummary {
    let mut sizes = HashMap::new();
    let mut remaining_uses = HashMap::new();

    for node in nodes {
        sizes.insert(node.output.clone(), node.output_bytes);
        for input in &node.inputs {
            *remaining_uses.entry(input.clone()).or_insert(0usize) += 1;
        }
    }

    let first = &nodes[0];
    let mut live: HashSet<String> = HashSet::from([first.output.clone()]);
    let mut live_bytes = first.output_bytes;
    let mut peak_live_bytes = live_bytes;
    let mut peak_at = first.name.clone();
    let mut live_tensors_at_peak = vec![first.output.clone()];

    let mut max_single_tensor_name = first.output.clone();
    let mut max_single_tensor_bytes = first.output_bytes;
    for node in nodes {
        if node.output_bytes > max_single_tensor_bytes {
            max_single_tensor_bytes = node.output_bytes;
            max_single_tensor_name = node.output.clone();
        }
    }

    for node in nodes.iter().skip(1) {
        live.insert(node.output.clone());
        live_bytes += node.output_bytes;

        if live_bytes > peak_live_bytes {
            peak_live_bytes = live_bytes;
            peak_at = node.name.clone();
            let mut snapshot: Vec<String> = live.iter().cloned().collect();
            snapshot.sort();
            live_tensors_at_peak = snapshot;
        }

        for input in &node.inputs {
            if let Some(uses) = remaining_uses.get_mut(input) {
                *uses -= 1;
                if *uses == 0 && live.remove(input) {
                    live_bytes -= sizes[input];
                }
            }
        }
    }

    MemorySummary {
        peak_live_bytes,
        peak_at,
        live_tensors_at_peak,
        max_single_tensor_name,
        max_single_tensor_bytes,
    }
}

fn expand(num_channels: usize, factor: f64) -> usize {
    (num_channels as f64 * factor).floor() as usize
}

fn bump(primitives: &mut BTreeMap<PrimitiveKind, usize>, primitive: PrimitiveKind, count: usize) {
    *primitives.entry(primitive).or_insert(0) += count;
}

fn base_conv_footprint(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    groups: usize,
    count_bn: usize,
    primitives: &mut BTreeMap<PrimitiveKind, usize>,
) -> ParamFootprint {
    let weights = out_channels * (in_channels / groups) * kernel_size * kernel_size;
    let raw_elems = weights + count_bn * out_channels * 4;
    let fused_elems = weights + out_channels;

    if groups == in_channels && out_channels == in_channels && kernel_size > 1 {
        bump(primitives, PrimitiveKind::DepthwiseConv2d, 1);
    } else {
        bump(primitives, PrimitiveKind::Conv2d, 1);
    }
    if count_bn > 0 {
        bump(primitives, PrimitiveKind::BatchNormFold, count_bn);
        bump(primitives, PrimitiveKind::SiLU, 1);
    }

    ParamFootprint {
        raw_bytes: raw_elems * BYTES_PER_F32,
        fused_bytes: fused_elems * BYTES_PER_F32,
    }
}

fn pred_conv_footprint(
    in_channels: usize,
    out_channels: usize,
    primitives: &mut BTreeMap<PrimitiveKind, usize>,
) -> ParamFootprint {
    let elems = out_channels * in_channels + out_channels;
    bump(primitives, PrimitiveKind::Conv2d, 1);

    ParamFootprint {
        raw_bytes: elems * BYTES_PER_F32,
        fused_bytes: elems * BYTES_PER_F32,
    }
}

fn conv_module_footprint(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    depthwise: bool,
    primitives: &mut BTreeMap<PrimitiveKind, usize>,
) -> ParamFootprint {
    if depthwise {
        let depthwise_fp = base_conv_footprint(
            in_channels,
            in_channels,
            kernel_size,
            in_channels,
            1,
            primitives,
        );
        let pointwise_fp = base_conv_footprint(in_channels, out_channels, 1, 1, 1, primitives);
        depthwise_fp + pointwise_fp
    } else {
        base_conv_footprint(in_channels, out_channels, kernel_size, 1, 1, primitives)
    }
}

fn focus_footprint(
    in_channels: usize,
    out_channels: usize,
    _depthwise: bool,
    primitives: &mut BTreeMap<PrimitiveKind, usize>,
) -> ParamFootprint {
    bump(primitives, PrimitiveKind::SliceStrided, 4);
    bump(primitives, PrimitiveKind::Concat, 1);
    base_conv_footprint(in_channels * 4, out_channels, 3, 1, 1, primitives)
}

fn bottleneck_footprint(
    channels: usize,
    shortcut: bool,
    depthwise: bool,
    primitives: &mut BTreeMap<PrimitiveKind, usize>,
) -> ParamFootprint {
    let mut footprint = base_conv_footprint(channels, channels, 1, 1, 1, primitives);
    footprint += conv_module_footprint(channels, channels, 3, depthwise, primitives);
    if shortcut {
        bump(primitives, PrimitiveKind::ElementwiseAdd, 1);
    }
    footprint
}

fn csp_bottleneck_footprint(
    in_channels: usize,
    out_channels: usize,
    num_blocks: usize,
    shortcut: bool,
    depthwise: bool,
    primitives: &mut BTreeMap<PrimitiveKind, usize>,
) -> ParamFootprint {
    let hidden_channels = expand(out_channels, 0.5);
    let mut footprint = base_conv_footprint(in_channels, hidden_channels, 1, 1, 1, primitives);
    footprint += base_conv_footprint(in_channels, hidden_channels, 1, 1, 1, primitives);
    footprint += base_conv_footprint(hidden_channels * 2, out_channels, 1, 1, 1, primitives);

    for _ in 0..num_blocks {
        footprint += bottleneck_footprint(hidden_channels, shortcut, depthwise, primitives);
    }

    bump(primitives, PrimitiveKind::Concat, 1);
    footprint
}

fn spp_bottleneck_footprint(
    in_channels: usize,
    out_channels: usize,
    primitives: &mut BTreeMap<PrimitiveKind, usize>,
) -> ParamFootprint {
    let hidden_channels = in_channels / 2;
    let mut footprint = base_conv_footprint(in_channels, hidden_channels, 1, 1, 1, primitives);
    bump(primitives, PrimitiveKind::MaxPool2d, 3);
    bump(primitives, PrimitiveKind::Concat, 1);
    footprint += base_conv_footprint(hidden_channels * 4, out_channels, 1, 1, 1, primitives);
    footprint
}

fn csp_block_footprint(
    in_channels: usize,
    out_channels: usize,
    depth: usize,
    spp: bool,
    shortcut: bool,
    depthwise: bool,
    primitives: &mut BTreeMap<PrimitiveKind, usize>,
) -> ParamFootprint {
    let mut footprint = conv_module_footprint(in_channels, out_channels, 3, depthwise, primitives);
    if spp {
        footprint += spp_bottleneck_footprint(out_channels, out_channels, primitives);
    }
    footprint += csp_bottleneck_footprint(
        out_channels,
        out_channels,
        depth,
        shortcut,
        depthwise,
        primitives,
    );
    footprint
}

fn conv_block_footprint(
    channels: usize,
    depthwise: bool,
    primitives: &mut BTreeMap<PrimitiveKind, usize>,
) -> ParamFootprint {
    let first = conv_module_footprint(channels, channels, 3, depthwise, primitives);
    let second = conv_module_footprint(channels, channels, 3, depthwise, primitives);
    first + second
}

fn head_scale_footprint(
    in_channels: usize,
    hidden_channels: usize,
    num_classes: usize,
    depthwise: bool,
    primitives: &mut BTreeMap<PrimitiveKind, usize>,
) -> ParamFootprint {
    let mut footprint = base_conv_footprint(in_channels, hidden_channels, 1, 1, 1, primitives);
    footprint += conv_block_footprint(hidden_channels, depthwise, primitives);
    footprint += conv_block_footprint(hidden_channels, depthwise, primitives);
    footprint += pred_conv_footprint(hidden_channels, num_classes, primitives);
    footprint += pred_conv_footprint(hidden_channels, 4, primitives);
    footprint += pred_conv_footprint(hidden_channels, 1, primitives);

    bump(primitives, PrimitiveKind::Sigmoid, 2);
    bump(primitives, PrimitiveKind::Concat, 1);
    bump(primitives, PrimitiveKind::Flatten, 1);
    footprint
}
