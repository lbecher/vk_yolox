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

## Checklist atual de implementacao

### Ja implementado no crate atual

- Runtime Vulkan 1.0 com `vulkano`, staging, buffers `device-local` e dispatch compute em `src/vulkan_conv.rs`.
- Primitivas operacionais para o caminho principal do grafo: `conv2d`, depthwise via grouped conv, `silu`, `sigmoid`, `add`, `maxpool2d`, `upsample-nearest`, `concat`, `focus` e decode por escala em `src/vulkan_conv.rs`.
- Caminho CPU de referencia para backbone, `PAFPN`, head, decode e pos-processamento basico em `src/yolox_blocks.rs`.
- Paralelismo CPU para convolucao com `rayon` em `src/fused_weights.rs`.
- CLI de demonstracao e comparacao CPU/GPU em `src/main.rs`.
- Bundle de modelo demo, manifesto e patch de pesos em `src/model_bundle.rs`.
- Aplicacao de patch raw (`conv` + `BatchNorm`) com fusao offline para gerar bundle pronto em `src/model_bundle.rs` e `src/main.rs`.
- Manifesto explicito de mapeamento entre nomes do bundle e nomes esperados da referencia Burn em `src/model_bundle.rs`.
- Conversor de `manifest.json` + bins `f32` da referencia Burn para `BundleRawWeightPatch` em `src/model_bundle.rs` e `src/main.rs`.
- Exportador na referencia Burn para gerar `manifest.json + .bin` em `refs/yolox-burn/src/export.rs`.
- `PAFPN` demo alinhado com a profundidade da referencia para o caso Tiny/Nano em `src/yolox_blocks.rs`.
- Planejamento de footprint, primitivas e memoria em `src/model_plan.rs`.

### Faltando para fechar inferencia YOLOX real

1. Importacao de pesos pre-treinados reais.
   Referencias:
   `refs/yolox-burn/src/weights.rs`
   `refs/yolox-burn/src/yolox.rs`
   Trabalho:
   baixar/carregar `.pth`, remapear nomes de camadas e converter os pesos para `BundleRawWeightPatch`, deixando a fusao final por conta do crate atual.
   Arquivos alvo:
   `src/model_bundle.rs`
   `src/fused_weights.rs`
   `src/main.rs`

2. Pipeline de entrada de imagem real.
   Referencia:
   `refs/yolox-burn/src/main.rs`
   Trabalho:
   abrir imagem, redimensionar para a resolucao do modelo, converter HWC -> NCHW e preparar tensor de entrada real em vez de `make_demo_tensor`.
   Arquivos alvo:
   `src/main.rs`
   `src/tensor_ops.rs`

3. Comando de inferencia fim a fim.
   Trabalho:
   adicionar um subcomando tipo `infer` ou `detect-image` que receba `--input`, `--bundle` ou `--weights`, escolha CPU/GPU e produza deteccoes reais.
   Arquivos alvo:
   `src/main.rs`
   `src/model_bundle.rs`
   `src/vulkan_conv.rs`

4. Saida final utilizavel.
   Referencia:
   `refs/yolox-burn/src/main.rs`
   Trabalho:
   salvar imagem anotada, ou emitir JSON com boxes/class/score no espaco da imagem original.
   Arquivos alvo:
   `src/main.rs`
   possivelmente um novo `src/image_io.rs` ou `src/visualization.rs`

### Faltando para equivalencia com a referencia Burn

1. Validar NMS e pos-processamento contra a referencia.
   Referencia:
   `refs/yolox-burn/src/boxes.rs`
   Situacao atual:
   `src/yolox_blocks.rs` ja possui NMS simplificado por classe, mas ainda nao esta explicitamente validado contra a saida da referencia Burn.
   Arquivos alvo:
   `src/yolox_blocks.rs`
   `src/main.rs`

2. Validar decode/head com pesos reais.
   Referencia:
   `refs/yolox-burn/src/head.rs`
   Situacao atual:
   o crate atual implementa decode em CPU e GPU, inclusive em shader fundido, mas isso ainda esta validado apenas com pesos de demo.
   Arquivos alvo:
   `src/yolox_blocks.rs`
   `src/vulkan_conv.rs`
   `src/main.rs`

3. Testes de comparacao CPU x GPU com bundles reais.
   Trabalho:
   medir `max_abs_diff`, `mean_abs_diff` e divergencia de boxes finais usando pesos reais do YOLOX-Tiny/Nano.
   Arquivos alvo:
   `src/main.rs`
   possivelmente `tests/` para testes automatizados

### Diferencas atuais entre o plano e a implementacao

- O plano lista `elementwise-multiply`, `flatten`, `swap-dims` e `grid-generation` como primitivas separadas.
- A implementacao atual resolve boa parte disso dentro do shader `decode-head-scale` em `src/vulkan_conv.rs`, o que e funcional, mas foge do desenho original por primitivas.
- O caminho `resident_weights` existe para bundles demo em `src/main.rs` e `src/vulkan_conv.rs`, mas ainda nao esta ligado a pesos reais importados da referencia.

### Ordem recomendada a partir daqui

1. Implementar importador de pesos reais para YOLOX-Tiny.
2. Adicionar comando de inferencia com imagem real no CLI.
3. Validar saida CPU contra `refs/yolox-burn`.
4. Validar saida GPU contra CPU usando o mesmo bundle real.
5. Refinar NMS, exportar resultado final e so depois considerar NMS em GPU.

## TODO de otimizacao GPU

Objetivo: sair do estado atual de GPU funcional porem lenta para um runtime reutilizavel,
com menor overhead por inferencia e kernels mais adequados ao volume de trabalho do YOLOX.

### Fase 1: overhead estrutural do runtime

- [x] Corrigir divergencias de decode CPU/GPU antes de otimizar performance.
- [x] Adicionar benchmark dedicado (`bench-infer-bundle`) para medir CPU x GPU.
- [x] Adicionar cache de pipelines no runtime Vulkan para evitar recompilar WGSL/SPIR-V em toda operacao.
- [x] Criar sessoes GPU reutilizaveis para benchmark, evitando recriar `VulkanRuntime` a cada iteracao.
- [x] Criar sessao GPU com pesos preparados/residentes para reuso entre inferencias no benchmark.
- [ ] Usar sessoes persistentes tambem no caminho normal de inferencia (`infer-bundle`), nao so no benchmark.

### Fase 2: sincronizacao e gravacao de comandos

- [x] Reduzir `submit + wait` por operacao em `execute_compute`.
- [ ] Gravar varios dispatches no mesmo command buffer e sincronizar apenas no final da inferencia.
- [~] Prototipo de batching de dispatches no `AutoCommandBufferBuilder` foi testado e mantido desligado.
  Situacao atual: correto, mas sem ganho nesta GPU e com regressao quando ativado do jeito testado.
  Proximo passo, se retomarmos essa linha: medir batching mais granular ou pre-gravado, em vez de um batch monolitico.
- [x] Reduzir `submit + wait` nas copias de upload/readback (`copy_buffer_and_wait`).
- [x] Avaliar upload de parametros pequenos via buffers persistentes ou push-constant-like fallback quando possivel.

### Fase 3: memoria e buffers temporarios

- [x] Implementar pool simples de buffers temporarios `f32` por tamanho para evitar parte das realocacoes.
- [x] Reusar buffers intermediarios entre camadas do backbone, `PAFPN` e head no caminho preparado/residente.
  Impacto atual: correto, mas com ganho pequeno/ruidoso no benchmark principal desta maquina.
- [ ] Minimizar readbacks host, mantendo mais etapas no device antes do retorno final.

### Fase 4: otimizacao dos kernels

- [x] Fundir `conv + bias + silu` para reduzir trafego de memoria e numero de dispatches.
- [x] Fundir o `SPP` (`base + pool5 + pool9 + pool13`) em um unico kernel de saida concatenada.
  Impacto atual: correto, mas com ganho pequeno/ruidoso no benchmark principal.
- [x] Escrever o decode multiescala diretamente no buffer final, evitando `concat_rows`.
  Impacto atual: correto, mas sem ganho material na GPU usada para os testes.
- [ ] Especializar kernels para `1x1`, `3x3` e depthwise.
  Situacao atual: `1x1` e `3x3` padrao ja possuem caminho rapido; falta depthwise.
- [ ] Explorar tiling/workgroup memory na convolucao para reduzir leituras repetidas da memoria global.
- [ ] Revisar `workgroup_size` por kernel (`conv`, `concat`, `decode`, `upsample`) com base em benchmark real.
- [ ] Avaliar leituras/escritas vetorizadas (`vec4<f32>`) quando o layout permitir.

### Fase 5: otimizacoes avancadas

- [ ] Avaliar decode parcial e/ou NMS na GPU para reduzir custo host-side e readback.
- [ ] Avaliar `fp16` quando a GPU suportar e quando o erro numerico continuar aceitavel.
- [ ] Avaliar command buffers pre-gravados para shape fixo `640x640`.
- [ ] Criar benchmark segmentado por etapa: preprocess, backbone, neck, head, decode e NMS.

### Criterios de validacao

- Toda otimizacao precisa preservar a saida dentro da tolerancia de `compare-infer-bundle`.
- O benchmark principal deve ser sempre rodado em `--release`.
- Os resultados devem ser comparados com e sem `--resident-weights`.

### Formato simples para importar tensores externos

O crate agora aceita um manifesto JSON com a lista de tensores exportados da referencia Burn.
Cada tensor aponta para um arquivo `.bin` com `f32` little-endian.

Exemplo:

```json
{
  "source": "yolox-burn",
  "tensors": [
    {
      "name": "backbone.backbone.stem.conv.conv.weight",
      "file": "backbone_backbone_stem_conv_conv_weight.bin",
      "len": 1296
    },
    {
      "name": "backbone.backbone.stem.conv.bn.gamma",
      "file": "backbone_backbone_stem_conv_bn_gamma.bin",
      "len": 24
    }
  ]
}
```

Fluxo previsto:

1. Gerar um bundle base com a topologia esperada.
2. Exportar o mapeamento Burn para inspecao.
3. Exportar os tensores da referencia Burn para `.bin` + `manifest.json`.
4. Gerar `raw_patch.json` com o conversor interno.
5. Aplicar o patch raw ao bundle e gerar o bundle fusionado final.

## Uso

```bash
cd refs/yolox-burn
cargo run --features pretrained -- --export-tensors exported_tensors

cd ../..
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
cargo run -- export-burn-mapping-manifest --input yolox_demo_bundle.json --output yolox_burn_mapping.json
cargo run -- export-bundle-weights --input yolox_demo_bundle.json --output yolox_weights.json
cargo run -- build-raw-patch-from-burn-manifest --bundle yolox_demo_bundle.json --tensor-manifest burn_tensors_manifest.json --output yolox_raw_patch.json
cargo run -- apply-bundle-weights --bundle yolox_demo_bundle.json --weights yolox_weights.json --output yolox_bundle_patched.json
cargo run -- apply-bundle-raw-weights --bundle yolox_demo_bundle.json --raw-weights yolox_raw_patch.json --output yolox_bundle_fused.json
cargo run -- export-bundle-weight-dir --input yolox_demo_bundle.json --output-dir yolox_weight_dir
cargo run -- apply-bundle-weight-dir --bundle yolox_demo_bundle.json --weights-dir yolox_weight_dir --output yolox_bundle_patched_dir.json
cargo run -- demo-detect-bundle --input yolox_demo_bundle.json --cpu-only
cargo run -- demo-detect-bundle --input yolox_demo_bundle.json --resident-weights
cargo run -- infer-bundle --bundle yolox_tiny_bundle_real.json --image refs/yolox-burn/dog_bike_man.jpg --input-h 640 --input-w 640 --cpu-only --output-json infer_report_cpu.json --output-image infer_cpu.png
cargo run -- infer-bundle --bundle yolox_tiny_bundle_real.json --image refs/yolox-burn/dog_bike_man.jpg --input-h 640 --input-w 640 --output-json infer_report_gpu.json
cargo run -- compare-infer-bundle --bundle yolox_tiny_bundle_real.json --image refs/yolox-burn/dog_bike_man.jpg --input-h 640 --input-w 640 --output-json compare_infer_report.json
cargo run -- bench-infer-bundle --bundle yolox_tiny_bundle_real.json --image refs/yolox-burn/dog_bike_man.jpg --input-h 640 --input-w 640 --confidence-threshold 0.5 --nms-threshold 0.65 --warmup-iterations 3 --iterations 10 --output-json bench_infer_report.json
cargo run --release -- bench-infer-bundle --bundle yolox_tiny_bundle_real.json --image refs/yolox-burn/dog_bike_man.jpg --input-h 640 --input-w 640 --confidence-threshold 0.5 --nms-threshold 0.65 --warmup-iterations 3 --iterations 10 --output-json bench_infer_report_release.json
cargo run --release -- bench-infer-bundle --bundle yolox_tiny_bundle_real.json --image refs/yolox-burn/dog_bike_man.jpg --input-h 640 --input-w 640 --confidence-threshold 0.5 --nms-threshold 0.65 --resident-weights --warmup-iterations 3 --iterations 10 --output-json bench_infer_report_release_resident.json
```

## Benchmark e progresso das otimizacoes

O comando `bench-infer-bundle` agora carrega bundle e imagem uma vez, executa warmup,
roda multiplas iteracoes e mede CPU/GPU separadamente.

Melhorias implementadas nesta fase:

- cache de pipelines no runtime Vulkan;
- sessao GPU reutilizavel para benchmark;
- sessao GPU com pesos preparados/residentes para reuso entre iteracoes.
- upload de buffers pequenos de parametros em memoria host-visible, evitando `copy + wait` por kernel.
- fusao de `conv + bias + silu` nos blocos com ativacao.
- encadeamento de `GpuFuture` para evitar `wait` a cada dispatch e a cada copia intermediaria.
- kernel especializado para convolucoes `1x1` com e sem `SiLU`.
- kernel especializado para convolucoes `3x3` padrao com e sem `SiLU`.
- fusao do `SPP` em um unico kernel de concat+pooling.
- decode multiescala com escrita direta no buffer final.
- pool simples de buffers temporarios `f32` com reciclagem explicita dos intermediarios mais quentes.

Resultado medido em `--release` com `dog_bike_man.jpg`, `640x640`, `warmup=1`, `iterations=3`:

- antes das otimizacoes estruturais:
  CPU `2287.675 ms`, GPU `744.818 ms`, speedup `3.07x`
- depois de reutilizar runtime/pipelines no benchmark:
  CPU `2297.811 ms`, GPU `489.499 ms`, speedup `4.69x`
- depois com sessao GPU residente:
  CPU `2300.373 ms`, GPU residente `470.690 ms`, speedup `4.89x`
- depois de mover params pequenos para host-visible:
  CPU `2232.879 ms`, GPU residente `438.679 ms`, speedup `5.09x`
- depois de fundir `conv + bias + silu`:
  CPU `2321.510 ms`, GPU residente `424.669 ms`, speedup `5.47x`
- depois de encadear submissao/copia sem `wait` por operacao:
  CPU `2335.420 ms`, GPU residente `406.936 ms`, speedup `5.74x`
- depois de especializar `conv 1x1`:
  CPU `2205.338 ms`, GPU residente `356.409 ms`, speedup `6.19x`
- depois de especializar `conv 3x3`:
  CPU `2204.818 ms`, GPU residente `262.862 ms`, speedup `8.39x`
- depois de fundir `SPP` e simplificar o decode final:
  benchmark continuou na faixa de `~265-268 ms` para GPU residente nesta maquina.
  Conclusao: as proximas melhorias relevantes devem vir de `command buffer` unico por inferencia
  e de reuso de buffers temporarios, nao de mais pequenas fusoes isoladas.
- depois de ligar o pool simples de buffers temporarios no caminho preparado/residente:
  benchmark permaneceu na faixa de `~265 ms` para GPU residente nesta maquina.
  Conclusao: o proximo salto de performance deve vir mais de reduzir gravacao/submissao por kernel
  do que de alocacao isolada de buffers.
- tentativa de batching de dispatches em um command buffer unico:
  permaneceu correta, mas regrediu para a faixa de `~275 ms` nesta maquina.
  O prototipo ficou desativado por padrao para nao piorar o caminho principal.

Arquivos de referencia:

- `bench_infer_report_release.json`
- `bench_infer_report_release_resident.json`

Proximo alvo de maior impacto: gravar mais dispatches no mesmo command buffer e implementar
reuso de buffers temporarios. Depois disso, retomar depthwise e tuning de `workgroup_size`.

## Requisito de ambiente

- Em macOS, o runtime `demo-conv` precisa de um loader Vulkan funcional. Na prática isso costuma
  significar `MoltenVK` ou o Vulkan SDK instalado e visível para o processo.
