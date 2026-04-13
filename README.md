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
- `Pain Point Engine`: maps the exact problems in the vacancy to technical research responses
- `Athanasiadis Upgrade`: responsible AI, benchmark realism, knowledge-guided ML, and crop-model-aware project framing
- `Torres Upgrade`: multimedia retrieval, visual analytics, data science, and eScience-oriented project framing
- `Three-Professor Blueprint`: one unified agenda connecting Athanasiadis, Torres, Kapoor, and the vacancy
- `Advanced Agenda`: modular research architecture, advanced questions, and project distinctiveness
- `Method Blueprint`: multimodal pipeline, SSL objectives, and physics-aware layer
- `Poster Architecture`: a visual interview-ready summary of the full research pipeline
- `Experimental Plan`: OOD splits, ablations, and the technical stack
- `Vacancy Match`: direct mapping to the actual WUR position duties, requirements, application pack, and dates
- `Dataset Strategy`: data sources, G-E-M signal mapping, and supervisor-linked rationale
- `Candidate Fit`: how the project demonstrates motivation, collaboration, writing, Python, PyTorch, and scikit-learn readiness
- `CV Value Pack`: copy-ready CV bullets, recruiter summary, repo-proof mapping, and interview talking points
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
- it now aligns more directly with Athanasiadis through responsible AI, benchmark realism, knowledge-guided ML, and crop-yield-oriented framing
- it now aligns more directly with Torres through multimedia retrieval, visual analytics, and data-centric multimodal research workflows
- it now combines all three supervisors into one more advanced and distinctive research agenda
- it uses the AgriscienceFM `G-E-M` concept instead of a generic multimodal story
- it now maps the vacancy pain points directly to technical modules, benchmark design, and supervisor-specific methods
- it presents publishable research questions, OOD evaluation logic, and ablation design
- it now maps the project directly to the vacancy duties, requested qualities, and required application documents
- it now includes dataset planning, hypotheses, risks, and a reusable proposal builder
- it now shows how the project reflects the vacancy's required qualities, duties, and ML capabilities
- it now includes a CV-focused value pack so the repo can be presented as evidence of research readiness
- it now generates a one-page proposal and a WUR-specific motivation letter draft
- it now supports editable applicant profile fields and local or uploaded experiment logs
- it reads like a serious PhD pitch rather than a startup-style dashboard

## Why This Adds Value To A CV

- it shows research ownership: problem framing, hypotheses, risks, and evaluation design
- it shows technical execution: benchmark utilities, prototype training flow, and experiment schema
- it shows scientific communication: proposal, motivation-letter, CV, and interview-ready exports
- it shows honest judgment: reproducible synthetic pilot evidence is labeled honestly, not inflated as finished results

## Athanasiadis-Specific Upgrade

- the project now emphasizes responsible AI for global agricultural challenges rather than only model novelty
- it frames benchmark design as a central contribution, especially for crop-yield and climate-stress-relevant tasks
- it positions physics-aware learning as a bridge between process-based crop modeling and modern ML
- it adds a roadmap toward plant phenomics, crop-yield benchmarking, and digital-twin or management extensions

## Three-Professor Upgrade

- Athanasiadis gives the benchmarked agricultural-impact and food-systems framing
- Torres adds multimedia retrieval, information visualisation, and data-centric multimodal analysis
- Kapoor adds physics-informed scientific machine learning and trustworthy adaptation
- together they turn the repo into a more original research platform rather than a standard remote-sensing project

## Advanced Pain-Point Upgrade

- the project now answers the position's core pain points explicitly: subtle shift, multimodal heterogeneity, weak agricultural latent structure, fragmented benchmarks, and lack of scientific grounding
- it adds a stronger technical identity through retrieval heads, visual analytics, scientific consistency checks, and benchmark realism
- it reads more like a science-facing agricultural foundation-model platform than a single-task app

## Vacancy-Specific Upgrade

- the app now includes a direct vacancy-matching section for the WUR PhD position
- it maps the repo to the duties around self-supervised foundation models, multimodal architectures, HPC training, and benchmark development
- it includes an application checklist for the CV, motivation letter, and scientific-writing sample, all capped at 3 pages each
- it highlights the interview date of May 15, 2026 and flags the deadline inconsistency in the vacancy text: May 4, 2026 in the body versus May 5, 2026 in the footer

## Personalization And Evidence

- fill in your name, degree, thesis focus, email, and GitHub from the sidebar to personalize the outputs
- the app will automatically read local logs from `data/experiment_runs.csv`
- you can also upload a CSV from the sidebar to replace the local evidence during a session

## Concrete Research Artifacts

- OOD benchmark utility: `eval/ood_benchmark.py`
- CLI runner: `experiments/run_ood_benchmark.py`
- Synthetic multimodal dataset: `data/synthetic_multimodal.py`
- Domain-inspired synthetic study trainer: `experiments/train_dummy_multimodal.py`
- Minimal PyTorch models: `models/multimodal_baseline.py`
- Technical report: `docs/technical_report.md`
- CV value note: `docs/cv_value_pack.md`
- Generated sample outputs:
  - `outputs/ood_summary.csv`
  - `outputs/ood_best_variants.csv`
  - `outputs/ood_improvements.csv`
  - `outputs/ood_report.md`

## Project Status

- `Implemented`: Streamlit research portfolio, OOD benchmark summarizer, synthetic experiment schema, and a domain-inspired synthetic study pipeline
- `Prototype`: image-only, multimodal, and physics-aware PyTorch baselines on synthetic data with benchmark-compatible metrics
- `Synthetic pilot evidence`: CSV logs and benchmark outputs generated from reproducible domain-inspired synthetic data
- `Not yet real evidence`: agricultural dataset training runs, real remote-sensing experiments, and thesis-backed results

### Run The OOD Benchmark Summarizer

```bash
python experiments/run_ood_benchmark.py --input data/experiment_runs.csv --output-dir outputs
```

### Run The Domain-Inspired Synthetic Study

```bash
python experiments/train_dummy_multimodal.py --epochs 4 --output-dir outputs/synthetic_study
```

This creates:

- `outputs/synthetic_study/synthetic_study_history.csv`
- `outputs/synthetic_study/synthetic_study_runs.csv`
- `outputs/synthetic_study/synthetic_study_summary.md`

The generated `synthetic_study_runs.csv` follows the same schema as the app and benchmark pipeline, and the trainer can also write directly to `data/experiment_runs.csv` for the app to load by default.

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

The charts and benchmark outputs in the app should be framed as **reproducible synthetic pilot evidence** unless you have actually run the experiments on real agricultural data. That framing is far more credible than pretending synthetic or scaffolded evidence is a finished scientific result.

## Next Improvements

- replace the synthetic pilot experiment log with your own real training outputs
- add richer dataset cards with source links, licenses, and preprocessing notes
- connect proposal exports to PDF or Markdown templates
- refine the motivation letter with your exact MSc thesis, publications, and personal background
- replace the synthetic pilot evidence with real runs that support prediction, retrieval, and robustness analysis together

## License

MIT
