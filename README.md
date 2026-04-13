# AgriFusion

**AgriFusion: Image + Weather Learning for Crop Stress and Yield Risk**

A compact Streamlit demo for the Wageningen University & Research PhD position on **Foundation Models for Agricultural Sciences**.

This project is framed as a **foundation-model-inspired multimodal agricultural prototype**. It focuses on one question instead of many scattered demos:

**Given field imagery and basic field conditions, can multimodal modeling estimate crop stress risk and yield risk more robustly than a single-modality model?**

## Why This Fits The Vacancy

The WUR post is explicitly about:

- multimodal agricultural data,
- weak generalization in agriculture,
- image plus time-series or heterogeneous tabular inputs,
- domain-specific and self-supervised representation learning,
- and downstream tasks such as yield forecasting and crop failure detection.

AgriFusion is built directly around that space.

It combines:

- `field image input`
- `weather / seasonal signals`
- `field metadata`
- `single-modality vs multimodal comparison`
- `generalization testing under agricultural shift`
- `an honest explanation of where the model fails`

## App Structure

The Streamlit app has seven focused sections:

1. `Home`
   Problem framing, vacancy fit, and the central research question.
2. `Data Explorer`
   Synthetic canopy samples, weather plots, metadata, class distribution, missingness, and split logic.
3. `Model Lab`
   Comparison of image-only, tabular-only proxy, multimodal, and physics-aware variants using benchmark outputs, confusion matrices, and error views.
4. `Prediction Demo`
   Interactive prototype for stress-risk and yield-risk prediction from image plus field conditions, with confidence and top influencing factors.
5. `Explainability`
   Visual risk heatmap and feature contribution analysis for the prototype scorer.
6. `Generalization Test`
   Region, year, climate-stress, sensor, and missing-modality performance comparison.
7. `Research Reflection`
   What worked, what failed, and how to pitch the same project differently to each professor.

## Best Technical Framing

This repo does **not** claim to be a full agricultural foundation model.

The right framing is:

**A foundation-model-inspired multimodal agricultural prototype**

That wording is stronger and more credible because it shows ambition without exaggeration.

## Current Evidence

The benchmark charts are based on a **reproducible synthetic pilot** stored in:

- `data/experiment_runs.csv`
- `outputs/synthetic_study/`
- `outputs/ood_summary.csv`

The app also includes a live prototype prediction page. That page is intentionally presented as an **interactive research demo**, not as a deployed agronomic decision system.

## How It Maps To The Three Professors

- `Ioannis Athanasiadis`: agriculture, food security, yield forecasting, and use-inspired AI for decision support
- `Ricardo da Silva Torres`: multimodal learning, machine vision, representation learning, and data science
- `Taniya Kapoor`: physics-informed, knowledge-guided, and scientifically grounded machine learning

## Quick Start

```bash
git clone https://github.com/iffat336/agricultural-foundation-models.git
cd agricultural-foundation-models
pip install -r requirements.txt
streamlit run app.py
```

For the heavier local ML prototype stack:

```bash
pip install -r requirements-ml.txt
```

## Reproduce The Synthetic Benchmark

```bash
python experiments/train_dummy_multimodal.py --epochs 4 --output-dir outputs/synthetic_study
python experiments/run_ood_benchmark.py --input data/experiment_runs.csv --output-dir outputs
```

## Honest Limitations

- the benchmark evidence is synthetic, not real agricultural data
- the live prediction page is a prototype scorer, not saved trained inference
- tabular-only benchmarking should still be strengthened with a learned baseline
- true Grad-CAM or SHAP would require a saved trained model pipeline

## Best Next Step

Replace the synthetic pilot with one real agricultural benchmark using:

- image tiles,
- aligned weather time series,
- field or parcel metadata,
- and the same generalization-test structure already used in the app

## License

MIT
