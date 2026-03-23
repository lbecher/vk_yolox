.PHONY: \
	bench \
	bench-gpu \
	bench-cpu \
	bench-adb-armhf-build \
	bench-adb-armhf-push \
	bench-adb-armhf \
	bench-adb-aarch64-build \
	bench-adb-aarch64-push \
	bench-adb-aarch64

CARGO ?= cargo
ADB ?= adb
BIN_NAME ?= vk_yolox
BUNDLE ?= yolox_tiny_bundle_real.json
IMAGE ?= dog_bike_man.jpg
INPUT_H ?= 640
INPUT_W ?= 640
CONFIDENCE_THRESHOLD ?= 0.5
NMS_THRESHOLD ?= 0.65
WARMUP_ITERATIONS ?= 1
ITERATIONS ?= 3
ADB_RUN_DIR ?= /data/local/tmp/$(BIN_NAME)
ADB_PULL_DIR ?= .
ADB_SERIAL_FLAG :=

ifdef ADB_SERIAL
ADB_SERIAL_FLAG := -s $(ADB_SERIAL)
endif

BENCH_BASE_ARGS = \
	bench-infer-bundle \
	--bundle $(BUNDLE) \
	--image $(IMAGE) \
	--input-h $(INPUT_H) \
	--input-w $(INPUT_W) \
	--confidence-threshold $(CONFIDENCE_THRESHOLD) \
	--nms-threshold $(NMS_THRESHOLD) \
	--warmup-iterations $(WARMUP_ITERATIONS) \
	--iterations $(ITERATIONS)

BENCH_ARGS = $(BENCH_BASE_ARGS) --resident-weights --output-json bench_infer_report.json
BENCH_GPU_ARGS = $(BENCH_BASE_ARGS) --resident-weights --gpu-only --output-json bench_infer_report_gpu.json
BENCH_CPU_ARGS = $(BENCH_BASE_ARGS) --cpu-only --output-json bench_infer_report_cpu.json

ANDROID_ARMHF_TARGET ?= armv7-linux-androideabi
ANDROID_AARCH64_TARGET ?= aarch64-linux-android
ANDROID_ARMHF_BIN := target/$(ANDROID_ARMHF_TARGET)/release/$(BIN_NAME)
ANDROID_AARCH64_BIN := target/$(ANDROID_AARCH64_TARGET)/release/$(BIN_NAME)
ARMHF_BENCH_JSON ?= bench_infer_report_armhf.json
ARMHF_OUTPUT_JSON ?= infer_report_armhf.json
ARMHF_OUTPUT_IMAGE ?= infer_output_armhf.png
AARCH64_BENCH_JSON ?= bench_infer_report_aarch64.json
AARCH64_OUTPUT_JSON ?= infer_report_aarch64.json
AARCH64_OUTPUT_IMAGE ?= infer_output_aarch64.png

bench:
	$(CARGO) run --release -- $(BENCH_ARGS)

bench-gpu:
	$(CARGO) run --release -- $(BENCH_GPU_ARGS)

bench-cpu:
	$(CARGO) run --release -- $(BENCH_CPU_ARGS)

bench-adb-armhf-build:
	$(CARGO) build --release --target $(ANDROID_ARMHF_TARGET)

bench-adb-armhf-push: bench-adb-armhf-build
	$(ADB) $(ADB_SERIAL_FLAG) shell mkdir -p $(ADB_RUN_DIR)
	$(ADB) $(ADB_SERIAL_FLAG) push $(ANDROID_ARMHF_BIN) $(ADB_RUN_DIR)/$(BIN_NAME)
	$(ADB) $(ADB_SERIAL_FLAG) push $(BUNDLE) $(ADB_RUN_DIR)/$(notdir $(BUNDLE))
	$(ADB) $(ADB_SERIAL_FLAG) push $(IMAGE) $(ADB_RUN_DIR)/$(notdir $(IMAGE))

bench-adb-armhf: bench-adb-armhf-push
	mkdir -p $(ADB_PULL_DIR)
	$(ADB) $(ADB_SERIAL_FLAG) shell "cd $(ADB_RUN_DIR) && chmod +x ./$(BIN_NAME) && ./$(BIN_NAME) $(BENCH_BASE_ARGS) --resident-weights --output-json $(ARMHF_BENCH_JSON) && ./$(BIN_NAME) infer-bundle --bundle $(BUNDLE) --image $(IMAGE) --input-h $(INPUT_H) --input-w $(INPUT_W) --confidence-threshold $(CONFIDENCE_THRESHOLD) --nms-threshold $(NMS_THRESHOLD) --resident-weights --output-json $(ARMHF_OUTPUT_JSON) --output-image $(ARMHF_OUTPUT_IMAGE)"
	$(ADB) $(ADB_SERIAL_FLAG) pull $(ADB_RUN_DIR)/$(ARMHF_BENCH_JSON) $(ADB_PULL_DIR)/$(ARMHF_BENCH_JSON)
	$(ADB) $(ADB_SERIAL_FLAG) pull $(ADB_RUN_DIR)/$(ARMHF_OUTPUT_JSON) $(ADB_PULL_DIR)/$(ARMHF_OUTPUT_JSON)
	$(ADB) $(ADB_SERIAL_FLAG) pull $(ADB_RUN_DIR)/$(ARMHF_OUTPUT_IMAGE) $(ADB_PULL_DIR)/$(ARMHF_OUTPUT_IMAGE)

bench-adb-aarch64-build:
	$(CARGO) build --release --target $(ANDROID_AARCH64_TARGET)

bench-adb-aarch64-push: bench-adb-aarch64-build
	$(ADB) $(ADB_SERIAL_FLAG) shell mkdir -p $(ADB_RUN_DIR)
	$(ADB) $(ADB_SERIAL_FLAG) push $(ANDROID_AARCH64_BIN) $(ADB_RUN_DIR)/$(BIN_NAME)
	$(ADB) $(ADB_SERIAL_FLAG) push $(BUNDLE) $(ADB_RUN_DIR)/$(notdir $(BUNDLE))
	$(ADB) $(ADB_SERIAL_FLAG) push $(IMAGE) $(ADB_RUN_DIR)/$(notdir $(IMAGE))

bench-adb-aarch64: bench-adb-aarch64-push
	mkdir -p $(ADB_PULL_DIR)
	$(ADB) $(ADB_SERIAL_FLAG) shell "cd $(ADB_RUN_DIR) && chmod +x ./$(BIN_NAME) && ./$(BIN_NAME) $(BENCH_BASE_ARGS) --resident-weights --output-json $(AARCH64_BENCH_JSON) && ./$(BIN_NAME) infer-bundle --bundle $(BUNDLE) --image $(IMAGE) --input-h $(INPUT_H) --input-w $(INPUT_W) --confidence-threshold $(CONFIDENCE_THRESHOLD) --nms-threshold $(NMS_THRESHOLD) --resident-weights --output-json $(AARCH64_OUTPUT_JSON) --output-image $(AARCH64_OUTPUT_IMAGE)"
	$(ADB) $(ADB_SERIAL_FLAG) pull $(ADB_RUN_DIR)/$(AARCH64_BENCH_JSON) $(ADB_PULL_DIR)/$(AARCH64_BENCH_JSON)
	$(ADB) $(ADB_SERIAL_FLAG) pull $(ADB_RUN_DIR)/$(AARCH64_OUTPUT_JSON) $(ADB_PULL_DIR)/$(AARCH64_OUTPUT_JSON)
	$(ADB) $(ADB_SERIAL_FLAG) pull $(ADB_RUN_DIR)/$(AARCH64_OUTPUT_IMAGE) $(ADB_PULL_DIR)/$(AARCH64_OUTPUT_IMAGE)
