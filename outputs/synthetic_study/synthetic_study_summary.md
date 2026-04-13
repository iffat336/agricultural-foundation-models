# Domain-Inspired Synthetic Agricultural Study

This report is based on reproducible synthetic data with domain-inspired crop, weather, region, management, and shift structure. It is meant to demonstrate a realistic early-stage ML workflow, not to claim real agricultural performance.

## Why This Is Stronger Than A Placeholder Demo

- It trains three actual models instead of hand-writing benchmark numbers.
- It evaluates multiple OOD conditions: region, year, climate stress, sensor shift, and missing modality.
- It includes a physics-aware model with an auxiliary scientific-consistency target.
- It produces benchmark-compatible outputs that the app can load directly.

## baseline
- Final validation score: 0.723, F1: 0.716, ECE: 0.223, log_loss: 0.816, train_loss: 0.949

## multimodal_ssl
- Final validation score: 1.000, F1: 1.000, ECE: 0.004, log_loss: 0.004, train_loss: 0.027

## physics_aware_fm
- Final validation score: 1.000, F1: 1.000, ECE: 0.005, log_loss: 0.005, train_loss: 0.058, physics_mae: 0.134

## Benchmark-Compatible Rows

| run_name | split | model_variant | score | f1 | ece | notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| baseline_region_shift | Region transfer | baseline | 0.6562 | 0.5506 | 0.2566 | Image-only baseline trained on domain-inspired synthetic crop observations |
| baseline_year_shift | Year transfer | baseline | 0.7375 | 0.7278 | 0.2383 | Image-only baseline trained on domain-inspired synthetic crop observations |
| baseline_climate_stress | Climate-stress transfer | baseline | 0.7156 | 0.7223 | 0.2934 | Image-only baseline trained on domain-inspired synthetic crop observations |
| baseline_sensor_shift | Sensor transfer | baseline | 0.4969 | 0.4945 | 0.3149 | Image-only baseline trained on domain-inspired synthetic crop observations |
| baseline_missing_modality | Missing modality | baseline | 0.7094 | 0.7156 | 0.2236 | Image-only baseline trained on domain-inspired synthetic crop observations |
| multimodal_ssl_region_shift | Region transfer | multimodal_ssl | 1.0000 | 1.0000 | 0.0039 | Multimodal fusion model trained on synthetic image, weather, geo, management, and text features |
| multimodal_ssl_year_shift | Year transfer | multimodal_ssl | 1.0000 | 1.0000 | 0.0072 | Multimodal fusion model trained on synthetic image, weather, geo, management, and text features |
| multimodal_ssl_climate_stress | Climate-stress transfer | multimodal_ssl | 1.0000 | 1.0000 | 0.0050 | Multimodal fusion model trained on synthetic image, weather, geo, management, and text features |
| multimodal_ssl_sensor_shift | Sensor transfer | multimodal_ssl | 1.0000 | 1.0000 | 0.0034 | Multimodal fusion model trained on synthetic image, weather, geo, management, and text features |
| multimodal_ssl_missing_modality | Missing modality | multimodal_ssl | 0.3500 | 0.1728 | 0.6250 | Multimodal fusion model trained on synthetic image, weather, geo, management, and text features |
| physics_aware_fm_region_shift | Region transfer | physics_aware_fm | 1.0000 | 1.0000 | 0.0058 | Physics-aware multimodal model with auxiliary scientific-consistency target on synthetic agronomic signals; physics_mae=0.132 |
| physics_aware_fm_year_shift | Year transfer | physics_aware_fm | 1.0000 | 1.0000 | 0.0063 | Physics-aware multimodal model with auxiliary scientific-consistency target on synthetic agronomic signals; physics_mae=0.131 |
| physics_aware_fm_climate_stress | Climate-stress transfer | physics_aware_fm | 1.0000 | 1.0000 | 0.0073 | Physics-aware multimodal model with auxiliary scientific-consistency target on synthetic agronomic signals; physics_mae=0.355 |
| physics_aware_fm_sensor_shift | Sensor transfer | physics_aware_fm | 1.0000 | 1.0000 | 0.0053 | Physics-aware multimodal model with auxiliary scientific-consistency target on synthetic agronomic signals; physics_mae=0.134 |
| physics_aware_fm_missing_modality | Missing modality | physics_aware_fm | 0.3500 | 0.1728 | 0.5793 | Physics-aware multimodal model with auxiliary scientific-consistency target on synthetic agronomic signals; physics_mae=0.141 |
