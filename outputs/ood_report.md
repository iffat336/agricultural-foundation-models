# OOD Benchmark Summary

Source log: `data\experiment_runs.csv`

## Best Variant Per Split

- `Domain-inspired synthetic agricultural benchmark` / `Climate-stress transfer`: `multimodal_ssl` (score=1.00, f1=1.00, ece=0.01)
- `Domain-inspired synthetic agricultural benchmark` / `Missing modality`: `baseline` (score=0.71, f1=0.72, ece=0.22)
- `Domain-inspired synthetic agricultural benchmark` / `Region transfer`: `multimodal_ssl` (score=1.00, f1=1.00, ece=0.00)
- `Domain-inspired synthetic agricultural benchmark` / `Sensor transfer`: `multimodal_ssl` (score=1.00, f1=1.00, ece=0.00)
- `Domain-inspired synthetic agricultural benchmark` / `Year transfer`: `physics_aware_fm` (score=1.00, f1=1.00, ece=0.01)

## Variant Gains Vs Baseline

### Domain-inspired synthetic agricultural benchmark
- `Climate-stress transfer` / `multimodal_ssl`: score=1.00, gain_vs_baseline=+0.28
- `Climate-stress transfer` / `physics_aware_fm`: score=1.00, gain_vs_baseline=+0.28
- `Climate-stress transfer` / `baseline`: score=0.72, gain_vs_baseline=+0.00
- `Missing modality` / `baseline`: score=0.71, gain_vs_baseline=+0.00
- `Missing modality` / `multimodal_ssl`: score=0.35, gain_vs_baseline=-0.36
- `Missing modality` / `physics_aware_fm`: score=0.35, gain_vs_baseline=-0.36
- `Region transfer` / `multimodal_ssl`: score=1.00, gain_vs_baseline=+0.34
- `Region transfer` / `physics_aware_fm`: score=1.00, gain_vs_baseline=+0.34
- `Region transfer` / `baseline`: score=0.66, gain_vs_baseline=+0.00
- `Sensor transfer` / `multimodal_ssl`: score=1.00, gain_vs_baseline=+0.50
- `Sensor transfer` / `physics_aware_fm`: score=1.00, gain_vs_baseline=+0.50
- `Sensor transfer` / `baseline`: score=0.50, gain_vs_baseline=+0.00
- `Year transfer` / `multimodal_ssl`: score=1.00, gain_vs_baseline=+0.26
- `Year transfer` / `physics_aware_fm`: score=1.00, gain_vs_baseline=+0.26
- `Year transfer` / `baseline`: score=0.74, gain_vs_baseline=+0.00

## Table Snapshot

| use_case | split | variant | score | f1 | ece | runs |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Domain-inspired synthetic agricultural benchmark | Climate-stress transfer | multimodal_ssl | 1.00 | 1.00 | 0.01 | 1 |
| Domain-inspired synthetic agricultural benchmark | Climate-stress transfer | physics_aware_fm | 1.00 | 1.00 | 0.01 | 1 |
| Domain-inspired synthetic agricultural benchmark | Climate-stress transfer | baseline | 0.72 | 0.72 | 0.29 | 1 |
| Domain-inspired synthetic agricultural benchmark | Missing modality | baseline | 0.71 | 0.72 | 0.22 | 1 |
| Domain-inspired synthetic agricultural benchmark | Missing modality | multimodal_ssl | 0.35 | 0.17 | 0.62 | 1 |
| Domain-inspired synthetic agricultural benchmark | Missing modality | physics_aware_fm | 0.35 | 0.17 | 0.58 | 1 |
| Domain-inspired synthetic agricultural benchmark | Region transfer | multimodal_ssl | 1.00 | 1.00 | 0.00 | 1 |
| Domain-inspired synthetic agricultural benchmark | Region transfer | physics_aware_fm | 1.00 | 1.00 | 0.01 | 1 |
| Domain-inspired synthetic agricultural benchmark | Region transfer | baseline | 0.66 | 0.55 | 0.26 | 1 |
| Domain-inspired synthetic agricultural benchmark | Sensor transfer | multimodal_ssl | 1.00 | 1.00 | 0.00 | 1 |
| Domain-inspired synthetic agricultural benchmark | Sensor transfer | physics_aware_fm | 1.00 | 1.00 | 0.01 | 1 |
| Domain-inspired synthetic agricultural benchmark | Sensor transfer | baseline | 0.50 | 0.49 | 0.31 | 1 |
| Domain-inspired synthetic agricultural benchmark | Year transfer | multimodal_ssl | 1.00 | 1.00 | 0.01 | 1 |
| Domain-inspired synthetic agricultural benchmark | Year transfer | physics_aware_fm | 1.00 | 1.00 | 0.01 | 1 |
| Domain-inspired synthetic agricultural benchmark | Year transfer | baseline | 0.74 | 0.73 | 0.24 | 1 |
