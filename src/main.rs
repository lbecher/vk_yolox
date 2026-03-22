mod fused_weights;
mod model_bundle;
mod model_plan;
mod tensor_ops;
mod vulkan_conv;
mod yolox_blocks;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use fused_weights::{BatchNorm1d, Conv2dSpec, RawConv2dWeights, fuse_conv2d_bn};
use model_bundle::DemoModelBundle;
use model_plan::{ModelPlan, ModelVariant, build_model_plan, bytes_to_mib};
use std::{fs, path::PathBuf};
use tensor_ops::{
    TensorShape, add_nchw, compare_slices, concat_channels_nchw, focus_nchw, make_demo_tensor,
    maxpool2d_nchw, sigmoid_nchw, silu_nchw, upsample_nearest_nchw,
};
use vulkan_conv::{
    run_conv2d_demo, run_demo_backbone, run_demo_block, run_demo_bottleneck, run_demo_csp,
    run_demo_dark5, run_demo_decode, run_demo_decode_resident, run_demo_head, run_demo_pafpn,
    run_demo_stem,
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
    ExportBundleWeights(ExportBundleWeightsArgs),
    ApplyBundleWeights(ApplyBundleWeightsArgs),
    ExportBundleWeightDir(ExportBundleWeightDirArgs),
    ApplyBundleWeightDir(ApplyBundleWeightDirArgs),
    DemoDetectBundle(DemoDetectBundleArgs),
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

fn main() -> Result<()> {
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
        Command::ExportBundleWeights(args) => export_bundle_weights(args),
        Command::ApplyBundleWeights(args) => apply_bundle_weights(args),
        Command::ExportBundleWeightDir(args) => export_bundle_weight_dir(args),
        Command::ApplyBundleWeightDir(args) => apply_bundle_weight_dir(args),
        Command::DemoDetectBundle(args) => demo_detect_bundle(args),
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

fn export_bundle_weights(args: ExportBundleWeightsArgs) -> Result<()> {
    let bundle = DemoModelBundle::load_json(&args.input)?;
    bundle.save_weight_patch_json(&args.output)?;
    println!("patch de pesos salvo em {}", args.output.display());
    println!("camadas={}", bundle.named_layers().len());
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
