.PHONY: bench

bench:
	cargo run --release -- bench-infer-bundle --bundle yolox_tiny_bundle_real.json --image dog_bike_man.jpg --input-h 640 --input-w 640 --confidence-threshold 0.5 --nms-threshold 0.65 --resident-weights --warmup-iterations 1 --iterations 3 --output-json bench_infer_report.json
