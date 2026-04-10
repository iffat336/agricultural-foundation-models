# Technical Report

## Title

Physics-Aware Agricultural Foundation Models for Robust Multimodal Learning

## Purpose

This report documents the current evidence-bearing components of the project beyond the Streamlit presentation app. It is meant to show that the repository is moving toward a real research workflow that fits the Wageningen University & Research PhD position on foundation models for agricultural sciences.

## Research Problem

Agricultural machine learning systems often perform well in narrow in-domain settings but degrade when evaluated across geographic, temporal, sensor, or modality shifts. This project treats out-of-distribution robustness as a central objective rather than a secondary evaluation detail.

## Research Direction

The project studies agriculture-specific foundation models that align:

- biological signals,
- environmental signals,
- and management signals,

through a `G-E-M` framing inspired by AgriscienceFM.

The core technical agenda combines:

- multimodal self-supervised learning,
- geospatial and time-series modeling,
- physics-aware adaptation,
- and OOD benchmark design.

## Current Evidence In The Repo

### 1. Experiment Log Schema

The repository now includes a structured experiment log at:

- `data/experiment_runs.csv`

The current sample log records:

- use case,
- OOD split,
- model variant,
- score,
- F1,
- calibration error,
- and notes.

This structure is designed so real experiments can be dropped in later without changing the reporting pipeline.

### 2. OOD Benchmark Pipeline

The repository now includes an executable benchmark summarizer:

- `eval/ood_benchmark.py`
- `experiments/run_ood_benchmark.py`

These scripts:

- validate experiment logs,
- summarize results by use case, split, and model variant,
- identify the best-performing variant per split,
- compute gains versus a baseline,
- and export report artifacts for later inclusion in application materials.

### 3. Streamlit Integration

The Streamlit app can now:

- read local evidence from `data/experiment_runs.csv`,
- accept uploaded experiment logs,
- personalize proposal outputs using applicant information,
- and render interview-ready summaries of the research workflow.

## Current Model Variants

The benchmark scaffold assumes the following comparison style:

- `baseline`
- `multimodal_ssl`
- `physics_aware_fm`

This is intentionally simple and is designed to support the exact scientific questions highlighted in the vacancy:

- does multimodal self-supervision beat simpler baselines,
- do scientific priors improve robustness,
- and what happens under realistic agricultural shift.

## Why This Matters For The PhD Position

This repository now demonstrates progress on several duties and qualities named in the vacancy:

- familiarity with self-supervised learning concepts,
- multimodal model design thinking,
- benchmark development,
- Python and machine-learning workflow readiness,
- and the ability to communicate a research agenda clearly.

It also begins to connect the project more credibly to:

- Prof. Ioannis Athanasiadis through agricultural impact and benchmark realism,
- Prof. Ricardo da Silva Torres through multimodal self-supervised remote sensing,
- and Dr. Taniya Kapoor through physics-aware scientific ML.

## Current Limitations

The project still remains partly proposal-driven. The main limitations are:

- the included experiment log is currently a sample scaffold,
- there is not yet a trained multimodal model implementation in the repository,
- and the benchmark evidence is not yet backed by a full training pipeline.

These limitations are acceptable for a developing portfolio project, but they should be addressed before treating the repository as proof of completed research.

## Best Next Technical Steps

1. Replace the sample experiment log with actual runs from a baseline and one improved model.
2. Add a real OOD split generator for field-level train/test partitioning by geography or year.
3. Implement a minimal multimodal baseline with image and weather branches.
4. Add a short reproducibility note describing data assumptions and metrics.

## Summary

The repository now goes beyond a polished concept app. It contains a minimal but concrete evaluation scaffold, evidence schema, and research reporting structure aligned with the WUR PhD vacancy. The next milestone is to replace the sample evidence with real experiments and attach at least one implemented baseline model.
