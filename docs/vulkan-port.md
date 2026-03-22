# YOLOX em Vulkan 1.0

O código em `yolox-burn` serve como referência funcional do grafo e do pós-processamento.
O código em `vk_compute_test` já resolve a base de `vulkano` para Vulkan 1.0, criação de device,
buffers `device-local`, command buffers e dispatch de compute.

O crate raiz agora gera um plano explícito de porting para unir essas duas referências.

## O que precisa existir no runtime Vulkan

- Upload único dos pesos para buffers `device-local`, com staging apenas na inicialização.
- Fusão de `BatchNorm` nas convoluções antes do upload.
- Tensors NCHW em buffers lineares reutilizáveis.
- Encadeamento de kernels por primitiva, não por camada monolítica.
- Reuso agressivo de buffers temporários para reduzir o pico de memória.

## Primitivas mínimas

- `conv2d`
- `depthwise-conv2d`
- `silu`
- `sigmoid`
- `exp`
- `elementwise-add`
- `elementwise-multiply`
- `maxpool2d`
- `upsample-nearest`
- `concat`
- `slice-strided` para o `Focus`
- `flatten`
- `swap-dims`
- `grid-generation`

## Ordem prática de implementação

1. Exportar/fundir pesos do modelo YOLOX-Tiny para um layout estável de buffers.
2. Implementar `conv2d`, `depthwise-conv2d`, `silu` e `concat`.
3. Fechar backbone `CSPDarknet`.
4. Fechar `PAFPN` com `upsample-nearest`.
5. Fechar head, `decode` e leitura de saídas.
6. Decidir se `NMS` fica na CPU como na referência Burn ou se entra numa fase GPU posterior.

## Uso

```bash
cargo run -- inspect
cargo run -- inspect --model tiny --json
cargo run -- inspect --model nano --output plan.json
cargo run -- demo-conv
cargo run -- demo-depthwise
cargo run -- demo-block
cargo run -- demo-stem
cargo run -- demo-bottleneck
cargo run -- demo-csp
cargo run -- demo-dark5
cargo run -- demo-backbone
cargo run -- demo-pafpn
cargo run -- demo-head
cargo run -- demo-decode
cargo run -- demo-detect
cargo run -- demo-detect-cpu
cargo run -- demo-detect-resident
cargo run -- export-demo-bundle --output yolox_demo_bundle.json
cargo run -- inspect-bundle --input yolox_demo_bundle.json
cargo run -- inspect-bundle --input yolox_demo_bundle.json --layers
cargo run -- export-bundle-manifest --input yolox_demo_bundle.json --output yolox_layers.json
cargo run -- export-bundle-weights --input yolox_demo_bundle.json --output yolox_weights.json
cargo run -- apply-bundle-weights --bundle yolox_demo_bundle.json --weights yolox_weights.json --output yolox_bundle_patched.json
cargo run -- export-bundle-weight-dir --input yolox_demo_bundle.json --output-dir yolox_weight_dir
cargo run -- apply-bundle-weight-dir --bundle yolox_demo_bundle.json --weights-dir yolox_weight_dir --output yolox_bundle_patched_dir.json
cargo run -- demo-detect-bundle --input yolox_demo_bundle.json --cpu-only
cargo run -- demo-detect-bundle --input yolox_demo_bundle.json --resident-weights
```

## Requisito de ambiente

- Em macOS, o runtime `demo-conv` precisa de um loader Vulkan funcional. Na prática isso costuma
  significar `MoltenVK` ou o Vulkan SDK instalado e visível para o processo.
