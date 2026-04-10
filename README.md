# Physics-Aware Agricultural Foundation Models

An interactive Streamlit project concept for the Wageningen University & Research PhD position on **Foundation Models for Agricultural Sciences**.

This repo is intentionally framed as a **research proposal interface**, not a fake benchmark dashboard. It combines:

- multimodal self-supervised learning,
- out-of-distribution robustness,
- physics-aware scientific priors,
- the `G-E-M` framing from AgriscienceFM,
- and a clear mapping to the interests of the three listed supervisors.

## Why This Project Fits The Vacancy

The WUR position asks for research on:

- domain-specific foundation models for agriculture,
- multimodal heterogeneous data,
- image and time-series modeling,
- robustness under data shift,
- and self-supervised, contrastive, physics-informed, and knowledge-guided learning.

This app turns those ideas into one coherent project narrative:

- **Athanasiadis fit**: agriculture, food security, benchmark relevance, real impact
- **Torres fit**: self-supervised learning, multimodal remote sensing, representation learning
- **Kapoor fit**: physics-aware ML, scientific constraints, trustworthy adaptation

## App Sections

- `Overview`: vacancy pain points, project thesis, and honest benchmark targets
- `Research Fit`: professor-aware framing and the AgriscienceFM `G-E-M` lens
- `Method Blueprint`: multimodal pipeline, SSL objectives, and physics-aware layer
- `Poster Architecture`: a visual interview-ready summary of the full research pipeline
- `Experimental Plan`: OOD splits, ablations, and the technical stack
- `Dataset Strategy`: data sources, G-E-M signal mapping, and supervisor-linked rationale
- `Candidate Fit`: how the project demonstrates motivation, collaboration, writing, Python, PyTorch, and scikit-learn readiness
- `Supervisor Alignment`: how the project combines all three supervision profiles
- `Proposal Builder`: a configurable mini abstract generator for your application pitch
- `One-Page Proposal`: a downloadable research proposal draft tailored to the selected use case and emphasis
- `Motivation Letter`: a downloadable WUR-specific motivation letter draft
- `Application Pitch`: CV, motivation-letter, and interview wording you can reuse
- `References`: source links that ground the pitch in current public research context

## Suggested ML Stack

- `PyTorch`
- `PyTorch Lightning`
- `timm`
- `transformers`
- `TorchGeo`
- `xarray`
- `rasterio`
- `geopandas`
- `scikit-learn`
- `wandb`

## Quick Start

```bash
git clone https://github.com/iffat336/agricultural-foundation-models.git
cd agricultural-foundation-models
pip install -r requirements.txt
streamlit run app.py
```

For the full local ML prototype stack, use:

```bash
pip install -r requirements-ml.txt
```

## Why The New Version Is Stronger

- it now reflects the supervisors' current research directions more explicitly
- it uses the AgriscienceFM `G-E-M` concept instead of a generic multimodal story
- it presents publishable research questions, OOD evaluation logic, and ablation design
- it now includes dataset planning, hypotheses, risks, and a reusable proposal builder
- it now shows how the project reflects the vacancy's required qualities, duties, and ML capabilities
- it now generates a one-page proposal and a WUR-specific motivation letter draft
- it now supports editable applicant profile fields and local or uploaded experiment logs
- it reads like a serious PhD pitch rather than a startup-style dashboard

## Personalization And Evidence

- fill in your name, degree, thesis focus, email, and GitHub from the sidebar to personalize the outputs
- the app will automatically read local logs from `data/experiment_runs.csv`
- you can also upload a CSV from the sidebar to replace the local evidence during a session

## Concrete Research Artifacts

- OOD benchmark utility: `eval/ood_benchmark.py`
- CLI runner: `experiments/run_ood_benchmark.py`
- Synthetic multimodal dataset: `data/synthetic_multimodal.py`
- Dummy multimodal training script: `experiments/train_dummy_multimodal.py`
- Minimal PyTorch models: `models/multimodal_baseline.py`
- Technical report: `docs/technical_report.md`
- Generated sample outputs:
  - `outputs/ood_summary.csv`
  - `outputs/ood_best_variants.csv`
  - `outputs/ood_improvements.csv`
  - `outputs/ood_report.md`

## Project Status

- `Implemented`: Streamlit research portfolio, OOD benchmark summarizer, synthetic experiment schema, dummy multimodal prototype
- `Prototype`: image-only and multimodal PyTorch baselines on synthetic data with scikit-learn metrics
- `Sample evidence`: CSV logs and benchmark outputs generated from dummy data for workflow demonstration
- `Not yet real evidence`: agricultural dataset training runs, real remote-sensing experiments, and thesis-backed results

### Run The OOD Benchmark Summarizer

```bash
python experiments/run_ood_benchmark.py --input data/experiment_runs.csv --output-dir outputs
```

### Run The Dummy Multimodal Prototype

```bash
python experiments/train_dummy_multimodal.py --epochs 8 --output-dir outputs/dummy_training
```

This creates:

- `outputs/dummy_training/dummy_training_history.csv`
- `outputs/dummy_training/dummy_training_runs.csv`
- `outputs/dummy_training/dummy_training_summary.md`

The generated `dummy_training_runs.csv` follows the same schema as the app and benchmark pipeline, so you can upload it directly in the sidebar.

## Streamlit Cloud Note

- `requirements.txt` is intentionally minimal for app deployment
- `requirements-ml.txt` contains the heavier local ML stack used for synthetic training and benchmark prototyping
- if you deploy on Streamlit Cloud, it should install only `requirements.txt`

### Experiment Log Schema

Use these columns in your CSV:

- `run_name`
- `use_case`
- `split`
- `model_variant`
- `score`
- `f1`
- `ece`
- `notes`

## Important Note

The charts in the app are presented as **illustrative targets and experiment plans** unless you have actually run the experiments and can support the numbers. That framing is more credible for a PhD application than hard-coded result claims.

## Next Improvements

- replace the sample experiment log with your own real training outputs
- add richer dataset cards with source links, licenses, and preprocessing notes
- connect proposal exports to PDF or Markdown templates
- refine the motivation letter with your exact MSc thesis, publications, and personal background

## License

MIT
