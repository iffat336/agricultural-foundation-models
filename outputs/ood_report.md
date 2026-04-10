# OOD Benchmark Summary

Source log: `data\experiment_runs.csv`

## Best Variant Per Split

- `Monitoring crops and water from space` / `Region transfer`: `physics_aware_fm` (score=71.10, f1=0.69, ece=0.10)
- `Monitoring crops and water from space` / `Year transfer`: `physics_aware_fm` (score=68.30, f1=0.66, ece=0.11)
- `Phenotyping and breeding resilient crops` / `Missing modality`: `physics_aware_fm` (score=60.40, f1=0.58, ece=0.13)
- `Precision farming for pest and disease` / `Sensor transfer`: `physics_aware_fm` (score=64.70, f1=0.62, ece=0.12)

## Variant Gains Vs Baseline

### Monitoring crops and water from space
- `Region transfer` / `physics_aware_fm`: score=71.10, gain_vs_baseline=+9.90
- `Region transfer` / `multimodal_ssl`: score=67.80, gain_vs_baseline=+6.60
- `Region transfer` / `baseline`: score=61.20, gain_vs_baseline=+0.00
- `Year transfer` / `physics_aware_fm`: score=68.30, gain_vs_baseline=+9.90
- `Year transfer` / `multimodal_ssl`: score=64.20, gain_vs_baseline=+5.80
- `Year transfer` / `baseline`: score=58.40, gain_vs_baseline=+0.00

### Phenotyping and breeding resilient crops
- `Missing modality` / `physics_aware_fm`: score=60.40, gain_vs_baseline=+10.90
- `Missing modality` / `multimodal_ssl`: score=56.10, gain_vs_baseline=+6.60
- `Missing modality` / `baseline`: score=49.50, gain_vs_baseline=+0.00

### Precision farming for pest and disease
- `Sensor transfer` / `physics_aware_fm`: score=64.70, gain_vs_baseline=+10.10
- `Sensor transfer` / `multimodal_ssl`: score=60.50, gain_vs_baseline=+5.90
- `Sensor transfer` / `baseline`: score=54.60, gain_vs_baseline=+0.00

## Table Snapshot

| use_case | split | variant | score | f1 | ece | runs |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Monitoring crops and water from space | Region transfer | physics_aware_fm | 71.10 | 0.69 | 0.10 | 1 |
| Monitoring crops and water from space | Region transfer | multimodal_ssl | 67.80 | 0.65 | 0.14 | 1 |
| Monitoring crops and water from space | Region transfer | baseline | 61.20 | 0.59 | 0.18 | 1 |
| Monitoring crops and water from space | Year transfer | physics_aware_fm | 68.30 | 0.66 | 0.11 | 1 |
| Monitoring crops and water from space | Year transfer | multimodal_ssl | 64.20 | 0.62 | 0.15 | 1 |
| Monitoring crops and water from space | Year transfer | baseline | 58.40 | 0.56 | 0.20 | 1 |
| Phenotyping and breeding resilient crops | Missing modality | physics_aware_fm | 60.40 | 0.58 | 0.13 | 1 |
| Phenotyping and breeding resilient crops | Missing modality | multimodal_ssl | 56.10 | 0.54 | 0.17 | 1 |
| Phenotyping and breeding resilient crops | Missing modality | baseline | 49.50 | 0.47 | 0.22 | 1 |
| Precision farming for pest and disease | Sensor transfer | physics_aware_fm | 64.70 | 0.62 | 0.12 | 1 |
| Precision farming for pest and disease | Sensor transfer | multimodal_ssl | 60.50 | 0.58 | 0.16 | 1 |
| Precision farming for pest and disease | Sensor transfer | baseline | 54.60 | 0.52 | 0.21 | 1 |
