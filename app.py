from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Agricultural Foundation Models",
    layout="wide",
    initial_sidebar_state="expanded",
)


PAIN_POINTS = [
    ("Generalization under shift", "Agricultural models often fail across regions, years, sensors, and management settings."),
    ("Fragmented modalities", "Satellite imagery, weather, field boundaries, and metadata are usually modeled separately."),
    ("Weak scientific grounding", "Purely data-driven systems may ignore crop dynamics and become less trustworthy."),
    ("Benchmark realism", "Applied agricultural AI should be tested under climate stress, operational constraints, and stakeholder-relevant tasks."),
]

SUPERVISORS = [
    {
        "name": "Prof. Ioannis Athanasiadis",
        "focus": "Responsible agricultural AI, benchmark realism, knowledge-guided ML, crop modeling, and use-inspired research",
        "fit": "Frames the project around real-world agricultural impact, climate-stress-aware benchmarking, process-informed learning, and collaboration with domain experts.",
        "line": "I want to build responsible agricultural AI that is benchmarked under realistic shift, informed by agricultural knowledge, and useful for food-system decisions.",
    },
    {
        "name": "Prof. Ricardo da Silva Torres",
        "focus": "Data science, visual computing, multimedia analysis and retrieval, information visualisation, and multimodal remote sensing AI",
        "fit": "Anchors the representation-learning and data-centric side through multimodal retrieval, visual analytics, scalable representation learning, and benchmark-oriented eScience workflows.",
        "line": "I want to study how agriculture-specific multimodal representations can support not only prediction, but also retrieval, exploration, and visual understanding across agricultural datasets.",
    },
    {
        "name": "Dr. Taniya Kapoor",
        "focus": "Physics-informed ML, scientific machine learning, trustworthy adaptation, and foundation models for science",
        "fit": "Drives the scientific-ML layer through constraints, priors, plausibility checks, and physically grounded adaptation under shift.",
        "line": "I am interested in combining representation learning with scientific priors so agricultural foundation models become more trustworthy, stable, and scientifically meaningful under shift.",
    },
]

GEM = pd.DataFrame(
    [
        ["G", "Biological material", "Crop type, disease, phenotyping", "Vision SSL and phenotype-aware embeddings"],
        ["E", "Environment", "Weather, soil, climate, radiation", "Temporal encoders and climate-aware conditioning"],
        ["M", "Management", "Field boundaries, geolocation, metadata", "Geo encoders and metadata fusion"],
        ["Alignment", "Cross-driver fusion", "Joint latent space across G-E-M", "Cross-modal contrastive learning"],
        ["Physics", "Scientific grounding", "Crop dynamics priors", "Constraint losses and plausibility checks"],
    ],
    columns=["Module", "Meaning", "Signals", "Contribution"],
)

OOD = pd.DataFrame(
    [
        ["Region transfer", "Train on one geography, test on another"],
        ["Year transfer", "Train on one season or year, test on another"],
        ["Sensor transfer", "Vary acquisition conditions or imagery source"],
        ["Missing modality", "Drop weather, metadata, or geo context at test time"],
        ["Sparse labels", "Low-label downstream finetuning evaluation"],
    ],
    columns=["Split", "Design"],
)

ABLATIONS = pd.DataFrame(
    [
        ["Image only", "Is multimodal fusion actually needed?"],
        ["Image + weather", "How much temporal context helps?"],
        ["Full multimodal", "Upper-bound representation quality"],
        ["Full multimodal, no physics", "Isolate scientific-prior contribution"],
        ["Full multimodal + physics", "Target trustworthy model"],
    ],
    columns=["Variant", "Question"],
)

STACK = pd.DataFrame(
    [
        ["PyTorch", "Core modeling"],
        ["PyTorch Lightning", "HPC experiment management"],
        ["timm", "Vision backbones"],
        ["transformers", "Cross-modal fusion"],
        ["TorchGeo", "Remote sensing tooling"],
        ["xarray", "Weather and scientific arrays"],
        ["rasterio", "Raster ingestion"],
        ["geopandas", "Field polygons and joins"],
        ["scikit-learn", "Baselines and evaluation"],
        ["wandb", "Tracking and ablations"],
    ],
    columns=["Library", "Role"],
)

STACK_DETAILS = pd.DataFrame(
    [
        ["PyTorch", "Model development", "Custom encoders, multimodal fusion, SSL objectives, physics-aware losses"],
        ["scikit-learn", "Baselines and metrics", "Classical baselines, calibration, metrics, and structured evaluation"],
        ["PyTorch Lightning", "Large-scale training", "Cleaner experiment loops, logging, checkpointing, multi-GPU/HPC workflows"],
        ["TorchGeo", "Remote sensing ML", "Geospatial samplers, remote-sensing datasets, and domain-aware transforms"],
        ["xarray", "Scientific time series", "Weather cubes, aligned temporal windows, labeled arrays"],
        ["rasterio + geopandas", "Geo data engineering", "Raster ingestion, field polygons, parcel-level joins"],
        ["wandb", "Experiment tracking", "Ablations, comparisons, and report-ready experiment metadata"],
    ],
    columns=["Library", "Capability", "How it supports the project"],
)

POSITION_QUALITIES = pd.DataFrame(
    [
        ["Highly motivated", "Long-horizon benchmark plan, paper roadmap, and explicit research hypotheses"],
        ["Self-driven", "Proposal builder, independent project framing, and reproducible workflow thinking"],
        ["Curious", "Multiple use cases, ablation questions, and vacancy-aware novelty framing"],
        ["Dynamic international team fit", "Supervisor-aware collaboration story, benchmark construction, and interdisciplinary agricultural framing"],
        ["Applied ML background", "Remote sensing, multimodal SSL, OOD evaluation, and scientific ML positioning"],
        ["Scientific writing", "Built-in CV, motivation-letter, proposal, technical report, and interview phrasing"],
        ["English communication readiness", "Clear written exports and concise research summaries appropriate for an international PhD setting"],
    ],
    columns=["Quality from vacancy", "How the project demonstrates it"],
)

DUTY_ALIGNMENT = pd.DataFrame(
    [
        ["Familiarize with state-of-the-art self-supervised methods", "Research Fit, Method Blueprint, and literature-grounded SSL objectives"],
        ["Design, develop, and evaluate self-supervised architectures", "Multimodal G-E-M pipeline, model sketch, OOD plan, and benchmark framing"],
        ["Perform large-scale training on HPC", "PyTorch Lightning, reproducible experiment workflow, and HPC-oriented stack choices"],
        ["Disseminate research results through papers and conferences", "Hypotheses, proposal exports, technical report, and interview-ready summaries"],
        ["Collaborate on datasets and downstream benchmarks", "Dataset Strategy page, benchmark suite framing, and supervisor-aware use-case design"],
    ],
    columns=["Vacancy duty", "Where the project shows it"],
)

PHD_CAPABILITIES = pd.DataFrame(
    [
        ["Python proficiency", "PyTorch-centered modeling and reproducible experiment structure"],
        ["PyTorch experience", "SSL losses, multimodal encoders, adaptation modules, and HPC-aware training"],
        ["scikit-learn experience", "Baselines, metrics, calibration, and benchmarking"],
        ["Remote sensing familiarity", "Sentinel-2, geospatial joins, field boundaries, and TorchGeo tooling"],
        ["Time-series modeling", "Weather streams, temporal encoders, and year-shift evaluation"],
        ["Research communication", "Abstract generator, application pitch, and publication-oriented ablations"],
        ["International-team readiness", "Supervisor-aware framing, collaborative benchmark design, and clear written communication"],
        ["English writing signal", "One-page proposal, motivation letter, technical report, and CV exports"],
    ],
    columns=["Capability", "How this project signals it"],
)

VACANCY_SCOPE = pd.DataFrame(
    [
        ["Research target", "Domain-specific foundation models for agricultural sciences"],
        ["Core ML theme", "Modern self-supervised, contrastive, physics-informed, and knowledge-guided learning"],
        ["Data modalities", "Text, location, images, and time-series agricultural signals"],
        ["Priority tasks", "Crop type classification, crop yield forecasting, field boundary delineation, crop disease, and crop failure detection"],
        ["Team setting", "Interdisciplinary AgriscienceFM collaboration across AI, remote sensing, crop modelling, and food security"],
        ["Infrastructure expectation", "Large-scale training on HPC servers"],
    ],
    columns=["Vacancy dimension", "What the position asks for"],
)

APPLICATION_REQUIREMENTS = pd.DataFrame(
    [
        ["Curriculum vitae", "Maximum 3 pages", "Use the CV Value Pack bullets and repo assets from this app"],
        ["Motivation letter", "Maximum 3 pages", "Use the WUR-specific letter export and keep it tightly matched to the vacancy"],
        ["Scientific writing sample", "Maximum 3 pages", "Select your thesis/report section that best shows ML methods, evaluation, and writing clarity"],
        ["Submission route", "Apply through the WUR vacancy website only", "Prepare documents locally, then upload through the official portal"],
        ["Not required now", "Grades and transcripts", "Do not spend time adding extra files for this stage"],
    ],
    columns=["Application item", "Constraint", "Best project response"],
)

VACANCY_TIMELINE = pd.DataFrame(
    [
        ["Current reference date", "April 11, 2026", "This workspace date shows you still have time to tailor the application materials."],
        ["Application deadline in vacancy body", "May 4, 2026", "Treat this as the safer target date for submission."],
        ["Closing date shown in vacancy footer", "May 5, 2026", "The page text is inconsistent, so verify on the live submission form before final submission."],
        ["First interviews", "May 15, 2026", "Prepare a short oral pitch on multimodal SSL, OOD benchmarking, and agricultural impact."],
    ],
    columns=["Milestone", "Date", "What to do"],
)

REFERENCES = [
    ("AgriscienceFM project", "https://www.agriscience.fm/"),
    ("Ioannis Athanasiadis WUR profile", "https://www.wur.nl/nl/personen/ioannis-athanasiadis.htm"),
    ("Need for foundational models in agriculture", "https://research.wur.nl/en/publications/from-general-to-specialized-the-need-for-foundational-models-in-a/"),
    ("Knowledge-guided crop growth modeling", "https://research.wur.nl/en/publications/knowledge-guided-machine-learning-with-multivariate-sparse-data-f-2/"),
    ("Ricardo da Silva Torres research profile", "https://research.wur.nl/en/persons/ricardo-da-silva-torres/"),
    ("Semantically aware contrastive learning for multispectral remote sensing", "https://research.wur.nl/en/publications/semantically-aware-contrastive-learning-for-multispectral-remote-/"),
    ("Taniya Kapoor WUR profile", "https://www.wur.nl/en/persons/dr-t-taniya-kapoor-phd"),
    ("Oxford Schmidt AI in Science fellows", "https://www.mpls.ox.ac.uk/latest/news/oxford-welcomes-new-schmidt-ai-in-science-fellows"),
]

USE_CASES = {
    "Monitoring crops and water from space": {
        "tasks": "Crop type classification, field boundary delineation, water stress indicators",
        "modalities": "Sentinel-2 imagery, field polygons, weather summaries",
        "pitch": "Best showcase for Torres-style SSL and WUR remote sensing relevance.",
    },
    "Phenotyping and breeding resilient crops": {
        "tasks": "Trait proxy learning, resilience clustering, cross-environment transfer",
        "modalities": "Image data, environmental streams, genotype-related metadata",
        "pitch": "Strong G-E-M story with a science-facing foundation-model angle.",
    },
    "Precision farming for pest and disease": {
        "tasks": "Disease risk scoring, crop failure detection, early warning",
        "modalities": "Multispectral imagery, weather sequences, management context",
        "pitch": "Combines all three supervision strengths with practical value.",
    },
    "Benchmarking crop yield under climate stress": {
        "tasks": "Cross-region yield forecasting, climate-stress robustness, calibration under year shift",
        "modalities": "Satellite imagery, weather time series, crop-growth priors, management metadata",
        "pitch": "Direct Athanasiadis fit through benchmark realism, food-system relevance, and knowledge-guided agricultural AI.",
    },
    "Adaptive fertilizer management and digital twins": {
        "tasks": "Nitrogen-efficiency support, management simulation, constrained decision recommendations",
        "modalities": "Field observations, weather sequences, soil context, management actions",
        "pitch": "Extends the project toward digital twins, reinforcement learning, and actionable farm decisions.",
    },
    "Multimodal agricultural retrieval and visual analytics": {
        "tasks": "Cross-modal field retrieval, failure-case exploration, benchmark visual analytics, and evidence search across agricultural datasets",
        "modalities": "Satellite imagery, weather streams, geolocation, metadata, and text annotations",
        "pitch": "Direct Torres fit through multimedia retrieval, visual computing, data-centric eScience, and interpretable benchmark exploration.",
    },
}

DATASETS = pd.DataFrame(
    [
        ["Sentinel-2", "Multispectral satellite imagery", "Crop type, field boundaries, vegetation dynamics", "Torres"],
        ["ERA5 or weather station data", "Temperature, rainfall, radiation, humidity", "Temporal context and environment module", "Athanasiadis + Kapoor"],
        ["Field boundaries", "Parcel polygons and geospatial joins", "Management context and spatial grounding", "Athanasiadis"],
        ["Agronomic metadata", "Region, season, management notes, task labels", "Cross-modal alignment and retrieval", "Torres + Athanasiadis"],
        ["Field notes and text reports", "Text annotations, descriptions, and weak supervision cues", "Supports text-image-location learning and agricultural retrieval workflows", "Torres + Athanasiadis"],
        ["Crop-growth priors", "GDD, radiation-use signals, seasonal plausibility", "Physics-aware regularization", "Kapoor"],
        ["Crop-model outputs", "Process-based simulation traces or seasonal indicators", "Hybrid knowledge-guided learning and crop-yield forecasting", "Athanasiadis + Kapoor"],
        ["Phenomics signals", "Trait proxies, stress indicators, plant observations", "Plant phenomics and resilient phenotype representation learning", "Athanasiadis + Torres"],
    ],
    columns=["Dataset or signal", "Type", "Why include it", "Strongest supervisor link"],
)

HYPOTHESES = pd.DataFrame(
    [
        ["H1", "G-E-M multimodal SSL outperforms image-only pretraining under region and year shift."],
        ["H2", "Physics-aware regularization improves calibration and robustness under sparse-label finetuning."],
        ["H3", "Agriculture-specific latent spaces transfer better across crop tasks than generic pretrained backbones."],
    ],
    columns=["Hypothesis", "Claim"],
)

RISKS = pd.DataFrame(
    [
        ["Modality mismatch", "Align all inputs at field-season level and benchmark missing-modality robustness."],
        ["Weak labels", "Use SSL pretraining and low-label downstream evaluation."],
        ["Physics priors too rigid", "Use soft regularization rather than hard constraints."],
        ["Benchmark too broad", "Start with one flagship use case and expand incrementally."],
    ],
    columns=["Research risk", "Mitigation"],
)

ATHANASIADIS_SIGNALS = pd.DataFrame(
    [
        ["Responsible AI for global challenges", "His chair group explicitly advances AI methods for responsible, use-inspired impact.", "Frame the project around food systems, climate stress, and decision-relevant agricultural outcomes."],
        ["Use-inspired applied research", "He works in close cooperation with domain experts in environmental, social, and life sciences.", "Present the project as domain-aware benchmark and model design rather than generic ML experimentation."],
        ["AgMIP AgML benchmark culture", "He co-founded and coordinates the AgMIP machine-learning effort.", "Emphasize reproducible crop-yield and climate-stress benchmarking as a central project contribution."],
        ["Knowledge-guided ML in plant phenomics", "His talks and publications highlight knowledge-guided machine learning in agriculture and phenomics.", "Strengthen process-informed priors, phenotyping links, and scientifically grounded representation learning."],
        ["Process-based models plus ML", "He explicitly frames process-based crop models and ML as complementary.", "Position physics-aware learning as a hybrid bridge between crop modeling and foundation models."],
        ["Digital twins and reinforcement learning", "His recent work includes agricultural digital twins with reinforcement-learning intelligence.", "Add a roadmap from foundation models toward digital-twin and decision-support extensions."],
        ["Constrained agronomic decision support", "His publication list includes constrained RL for adaptive fertilizer management.", "Show how the project could later support trustworthy, management-aware recommendations."],
    ],
    columns=["Athanasiadis signal", "What it says about his research", "How this project should respond"],
)

BENCHMARK_PRINCIPLES = pd.DataFrame(
    [
        ["Real-world task choice", "Benchmark crop-relevant problems such as yield forecasting, stress detection, and resilient phenotyping rather than toy image tasks."],
        ["Stress-aware evaluation", "Include climate extremes, year shift, geography shift, and missing-modality settings as first-class benchmark splits."],
        ["Knowledge-guided baselines", "Compare generic deep models, multimodal SSL, and knowledge-guided or process-aware variants."],
        ["Operational usefulness", "Report calibration, transfer, robustness, and decision-relevant behavior instead of accuracy alone."],
        ["Domain-expert fit", "Explain why each task matters to agronomy, food systems, breeders, or farm management."],
        ["Transparent claims", "Separate sample workflow evidence from real-data findings to keep the project credible."],
    ],
    columns=["Benchmark principle", "Project implication"],
)

RESPONSIBLE_AI = pd.DataFrame(
    [
        ["Transparency", "Label sample results and illustrative targets honestly rather than overstating evidence."],
        ["Use-inspired evaluation", "Focus on crop stress, yield, resilience, and management tasks that matter in practice."],
        ["Robustness under shift", "Treat geographic, seasonal, and climatic variation as core evaluation settings."],
        ["Scientific grounding", "Use crop-growth priors, agronomic assumptions, and process knowledge where useful."],
        ["Human collaboration", "Frame the workflow as support for agronomists, breeders, and benchmark builders."],
    ],
    columns=["Responsible AI principle", "How the project demonstrates it"],
)

HYBRID_ROADMAP = pd.DataFrame(
    [
        ["Now", "Multimodal SSL + OOD benchmarking", "Strong fit for agricultural foundation models and benchmark design."],
        ["Next", "Knowledge-guided crop-yield forecasting under climate stress", "Directly reflects Athanasiadis' yield and benchmarking interests."],
        ["Then", "Phenotyping-aware and process-informed adaptation", "Links plant phenomics, scientific priors, and domain knowledge."],
        ["Future", "Agricultural digital twins and constrained decision support", "Builds a credible bridge to RL, management, and interoperable digital twins."],
    ],
    columns=["Stage", "Upgrade", "Why it adds value"],
)

TORRES_SIGNALS = pd.DataFrame(
    [
        ["Multimedia analysis", "His work spans multimedia analysis across heterogeneous data.", "Treat agriculture as a multimodal data science problem, not only an image-classification problem."],
        ["Multimedia retrieval", "He has worked extensively on retrieval-oriented systems.", "Add image-text-location retrieval, nearest-neighbour exploration, and benchmark-search capabilities to the project story."],
        ["Visual computing", "His background includes visual computing and advanced representation learning.", "Strengthen computer-vision framing for remote sensing, phenotyping, and cross-view agricultural understanding."],
        ["Information visualisation", "He works on making data and model behaviour explorable.", "Include visual analytics for benchmark failures, OOD drift, and cross-modal embeddings."],
        ["Databases and scalable data science", "His research connects ML with data infrastructure and eScience workflows.", "Present the repo as an extensible data-and-model platform, not just one model architecture."],
        ["Multidisciplinary eScience", "He develops multidisciplinary research systems rather than isolated methods.", "Emphasize reusable pipelines, benchmark tooling, and collaborative scientific workflows."],
    ],
    columns=["Torres signal", "What it says about his research", "How this project should respond"],
)

KAPOOR_SIGNALS = pd.DataFrame(
    [
        ["Scientific machine learning", "She works at the intersection of ML and scientific structure.", "Keep the project grounded in crop processes, seasonality, and scientific plausibility."],
        ["Physics-informed modeling", "Her profile emphasizes physics-aware learning.", "Use soft constraints, priors, and mechanism-aware regularization in adaptation and evaluation."],
        ["Trustworthy adaptation", "Scientific priors should improve stability under shift.", "Highlight calibration, robustness, and reliable transfer rather than raw performance alone."],
        ["Foundation models for science", "She connects modern foundation models to scientific domains.", "Position the repo as a science-facing foundation-model portfolio rather than a generic AI demo."],
    ],
    columns=["Kapoor signal", "What it says about her research", "How this project should respond"],
)

SUPERVISOR_SYNERGY = pd.DataFrame(
    [
        ["Athanasiadis", "Responsible agricultural AI, benchmarks, food-system impact", "Defines the real agricultural problems, benchmark realism, and use-inspired research agenda"],
        ["Torres", "Multimodal representations, retrieval, visual analytics, scalable data science", "Makes the project more original through retrieval, visual exploration, and data-centric model understanding"],
        ["Kapoor", "Physics-informed ML, scientific constraints, trustworthy adaptation", "Adds scientific credibility, plausibility, and reliability under real-world shift"],
        ["Vacancy requirements", "SSL, multimodal data, HPC, papers, collaboration", "Turns the three-supervisor concept into a position-specific PhD portfolio"],
    ],
    columns=["Layer", "Core contribution", "Why it matters in your project"],
)

UNIQUE_ADVANTAGES = pd.DataFrame(
    [
        ["Not just prediction", "The project now includes retrieval, benchmark visual analytics, and scientific explanation paths."],
        ["Not just deep learning", "It combines SSL, knowledge-guided learning, and physics-aware adaptation."],
        ["Not just one professor fit", "It has a clear role for Athanasiadis, Torres, and Kapoor in one coherent agenda."],
        ["Not just a vacancy summary", "It includes reusable exports, benchmark artifacts, and research-portfolio framing for your CV."],
        ["Not just a dashboard", "It reads like a PhD-ready research platform with extensible workflows and scientific positioning."],
    ],
    columns=["Why the project is distinctive", "Project upgrade"],
)

POSITION_PAINPOINT_ENGINE = pd.DataFrame(
    [
        ["Subtle distribution shift", "Agricultural models fail when geography, year, climate, or sensor conditions change slightly.", "OOD benchmark suite, climate-stress splits, calibration tracking, and robust adaptation.", "Athanasiadis + Kapoor"],
        ["Heterogeneous multimodal data", "The vacancy stresses text, location, images, and time-series data.", "Text-image-location-time fusion with cross-modal contrastive learning and retrieval-aware latent spaces.", "Torres + Kapoor + Athanasiadis"],
        ["Weak agriculture-specific latent structure", "Generic pretrained models may not capture agricultural semantics well.", "G-E-M-aligned self-supervised learning with task-aware latent diagnostics and retrieval evaluation.", "Torres + Athanasiadis"],
        ["Lack of scientific grounding", "Purely data-driven systems can be brittle or hard to trust.", "Crop-growth priors, plausibility constraints, and knowledge-guided adaptation.", "Kapoor + Athanasiadis"],
        ["Fragmented evaluation culture", "Benchmarks and downstream tasks are often disconnected from real agricultural use.", "Benchmark matrix covering crop type, yield, disease, failure detection, and field boundaries.", "Athanasiadis + Torres"],
        ["Need for dynamic intelligent systems", "The vacancy aims at systems that provide further insight into food-security-relevant applications.", "Retrieval, visual analytics, digital-twin extensions, and actionable model inspection workflows.", "All three"],
    ],
    columns=["Pain point from position", "Why it matters", "Advanced project response", "Strongest supervision link"],
)

ADVANCED_RESEARCH_MODULES = pd.DataFrame(
    [
        ["Multimodal foundation encoder", "Learns unified agricultural representations from imagery, weather, location, metadata, and text."],
        ["Cross-modal retrieval head", "Supports field search, similar-case retrieval, and failure-case exploration across modalities."],
        ["Scientific consistency head", "Uses crop-growth priors and plausibility checks to regularize adaptation under shift."],
        ["OOD benchmark engine", "Evaluates region, year, climate-stress, sensor, and missing-modality generalization."],
        ["Visual analytics layer", "Turns benchmark outputs into explorable views for scientific analysis and communication."],
        ["Decision-support extension path", "Connects foundation models to digital twins, constrained management, and food-security use cases."],
    ],
    columns=["Advanced module", "Role in the upgraded project"],
)

BENCHMARK_MATRIX = pd.DataFrame(
    [
        ["Crop type classification", "Imagery + location + field boundaries", "Macro-F1, calibration, region transfer", "Remote sensing and domain-specific representation quality"],
        ["Crop yield forecasting", "Imagery + weather + crop priors + metadata", "RMSE, calibration, year shift, climate-stress robustness", "Food security and benchmark realism"],
        ["Field boundary delineation", "Imagery + geospatial context", "IoU, boundary F1, sensor transfer", "Spatial precision and operational relevance"],
        ["Crop disease detection", "Imagery + weather + text notes", "AUC, ECE, low-label robustness", "Multimodal risk detection and retrieval support"],
        ["Crop failure detection", "Imagery + weather + management context", "F1, recall, early-warning lead time", "Actionable failure analysis under shift"],
        ["Multimodal retrieval and analytics", "Image + text + location + metadata", "Recall@K, cluster purity, failure-case coverage", "Torres-style retrieval and visual exploration"],
    ],
    columns=["Task", "Modalities", "Key metrics", "Why it belongs"],
)

PROFESSOR_METHOD_MATRIX = pd.DataFrame(
    [
        ["Athanasiadis", "Responsible AI, benchmark realism, crop modelling, food security", "Climate-stress benchmarks, use-inspired task design, agronomic context, food-system framing"],
        ["Torres", "Multimedia retrieval, visual computing, data science, information visualisation", "Cross-modal retrieval, visual analytics, scalable representation learning, data-centric benchmark exploration"],
        ["Kapoor", "Physics-informed ML, scientific machine learning, trustworthy adaptation", "Priors, constraints, scientific consistency, calibration-aware adaptation under shift"],
        ["Vacancy", "SSL, multimodal data, HPC, papers, collaboration", "Turns the three methodological layers into a concrete PhD-ready execution plan"],
    ],
    columns=["Source", "Research identity", "What the project should implement or signal"],
)

ADVANCED_RESEARCH_QUESTIONS = pd.DataFrame(
    [
        ["RQ1", "How can agriculture-specific multimodal SSL learn latent spaces that support both prediction and retrieval across tasks and modalities?"],
        ["RQ2", "When do scientific priors improve robustness, calibration, and failure detection under geographic, climatic, and temporal shift?"],
        ["RQ3", "How should agricultural benchmarks be designed so they remain useful for real food-security and management problems rather than narrow in-domain scoring?"],
        ["RQ4", "Can visual analytics and retrieval make foundation-model behavior more interpretable and reusable for interdisciplinary agricultural research teams?"],
    ],
    columns=["Research question", "Advanced framing"],
)

CV_SIGNAL_MAP = pd.DataFrame(
    [
        ["Research framing", "Professor-aware project arc, hypotheses, risks, and one-page proposal export", "Shows you can define a coherent PhD agenda instead of only building a demo"],
        ["Technical execution", "OOD benchmark utility, synthetic study trainer, and experiment-log schema", "Signals hands-on Python, PyTorch, evaluation, and workflow design"],
        ["Scientific communication", "Motivation letter, interview pitch, and CV-oriented exports inside the app", "Shows writing clarity and the ability to translate research into application materials"],
        ["Portfolio maturity", "README, technical report, repo artifacts, and downloadable summaries", "Makes the repository easier for supervisors and recruiters to scan quickly"],
    ],
    columns=["CV value area", "Project evidence", "Why it matters"],
)


def style() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
            .stApp {
                background:
                    radial-gradient(circle at 12% 12%, rgba(0,194,255,0.12), transparent 24%),
                    radial-gradient(circle at 86% 10%, rgba(88,255,182,0.10), transparent 20%),
                    radial-gradient(circle at 50% 100%, rgba(255,184,77,0.08), transparent 26%),
                    linear-gradient(180deg, #050505 0%, #0b0b0d 48%, #111216 100%);
            }
            html, body, [class*="css"] {
                font-family: "IBM Plex Sans", sans-serif;
                color: #f4f7fb;
            }
            h1, h2, h3, h4 { font-family: "Space Grotesk", sans-serif; color: #f8fbff; }
            p, li, label, .stMarkdown, .stCaption { color: #d2d9e3; }
            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, rgba(12,14,18,0.98) 0%, rgba(7,8,11,0.98) 100%);
                border-right: 1px solid rgba(255,255,255,0.08);
            }
            .hero, .card, .metric {
                background: linear-gradient(180deg, rgba(20,22,28,0.92) 0%, rgba(11,13,17,0.94) 100%);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 24px;
                box-shadow: 0 18px 40px rgba(0,0,0,0.35);
            }
            .hero { padding: 2rem 2.2rem; }
            .card { padding: 1rem; height: 100%; }
            .metric { padding: 1rem; }
            .kicker { color: #54f0c4; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.88rem; font-weight: 700; }
            .hero-title { font-size: 3rem; line-height: 1.0; margin: 0.35rem 0 1rem 0; }
            .muted { color: #a8b3c2; }
            .metric-label { color: #8d98a8; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.06em; }
            .metric-value { font-family: "Space Grotesk", sans-serif; font-size: 1.95rem; margin: 0.2rem 0; }
            .metric-value, .card h4, .hero, .poster-panel h4 { color: #f8fbff; }
            .metric-note { color: #54f0c4; font-weight: 600; }
            .poster-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            .poster-panel {
                background: linear-gradient(160deg, rgba(18,21,27,0.95), rgba(7,10,14,0.90));
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 26px;
                padding: 1rem 1.1rem;
                min-height: 190px;
                box-shadow: 0 18px 36px rgba(0,0,0,0.32);
            }
            .poster-panel h4 {
                margin-bottom: 0.55rem;
            }
            .poster-tag {
                display: inline-block;
                padding: 0.2rem 0.55rem;
                border-radius: 999px;
                background: rgba(84,240,196,0.12);
                color: #54f0c4;
                font-size: 0.78rem;
                font-weight: 700;
                margin-bottom: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            [data-testid="stDataFrame"], div[data-baseweb="select"], div[data-baseweb="input"] > div {
                border-radius: 18px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, text: str) -> None:
    st.markdown(f'<div class="card"><h4>{title}</h4><div>{text}</div></div>', unsafe_allow_html=True)


def metric(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def bar_fig() -> go.Figure:
    x = ["Region", "Year", "Sensor", "Missing modality", "Sparse labels"]
    base = [61, 58, 54, 49, 45]
    target = [71, 68, 64, 60, 57]
    fig = go.Figure()
    fig.add_bar(name="Typical baseline", x=x, y=base, marker_color="#ff9f43")
    fig.add_bar(name="Project target", x=x, y=target, marker_color="#54f0c4")
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(16,18,24,0.88)",
        margin=dict(l=20, r=20, t=30, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
        yaxis_title="Illustrative OOD score",
        font=dict(color="#ecf2ff"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def radar_fig() -> go.Figure:
    axes = ["Agri impact", "SSL", "Multimodal RS", "Scientific ML", "Benchmarks"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[5, 3, 3, 4, 5], theta=axes, fill="toself", name="Athanasiadis", line_color="#54f0c4", fillcolor="rgba(84,240,196,0.14)"))
    fig.add_trace(go.Scatterpolar(r=[3, 5, 5, 2, 4], theta=axes, fill="toself", name="Torres", line_color="#4fc3ff", fillcolor="rgba(79,195,255,0.12)"))
    fig.add_trace(go.Scatterpolar(r=[3, 3, 2, 5, 3], theta=axes, fill="toself", name="Kapoor", line_color="#b18cff", fillcolor="rgba(177,140,255,0.12)"))
    fig.add_trace(go.Scatterpolar(r=[5, 5, 5, 5, 5], theta=axes, fill="toself", name="Project fit", line_color="#f8fbff", fillcolor="rgba(248,251,255,0.07)"))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(16,18,24,0.88)",
            radialaxis=dict(visible=True, range=[0, 5], gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.08)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=25, b=20),
        legend=dict(orientation="h", y=1.13, x=0),
        font=dict(color="#ecf2ff"),
    )
    return fig


def sankey_fig() -> go.Figure:
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=18,
            label=["G inputs", "E inputs", "M inputs", "Vision SSL", "Temporal SSL", "Geo fusion", "Cross-modal alignment", "Physics layer", "OOD suite", "Crop tasks"],
            color=["#ff9f43", "#ff9f43", "#ff9f43", "#54f0c4", "#54f0c4", "#54f0c4", "#4fc3ff", "#f8fbff", "#b18cff", "#54f0c4"],
        ),
        link=dict(
            source=[0, 1, 2, 3, 4, 5, 6, 7, 7],
            target=[3, 4, 5, 6, 6, 6, 7, 8, 9],
            value=[6, 6, 4, 7, 6, 4, 9, 6, 7],
        ),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=20, b=10), font=dict(color="#ecf2ff"))
    return fig


def heatmap_fig() -> go.Figure:
    rows = ["Crop monitoring", "Yield-related prediction", "Disease/failure risk"]
    cols = ["G", "E", "M", "Physics", "OOD need"]
    values = [
        [4, 3, 4, 2, 5],
        [3, 5, 3, 5, 5],
        [4, 5, 4, 4, 5],
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=cols,
            y=rows,
            colorscale=[[0, "#111317"], [0.45, "#214154"], [0.7, "#2d8f74"], [1, "#54f0c4"]],
            showscale=False,
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(16,18,24,0.88)",
        margin=dict(l=20, r=20, t=20, b=10),
        font=dict(color="#ecf2ff"),
    )
    return fig


def load_experiment_logs(uploaded_file=None):
    columns = ["run_name", "use_case", "split", "model_variant", "score", "f1", "ece", "notes"]
    source = None

    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            source = "uploaded CSV"
        else:
            local_path = Path("data") / "experiment_runs.csv"
            if local_path.exists():
                df = pd.read_csv(local_path)
                source = str(local_path)
            else:
                return pd.DataFrame(columns=columns), None
    except Exception:
        return pd.DataFrame(columns=columns), None

    for column in columns:
        if column not in df.columns:
            df[column] = ""

    for numeric_col in ["score", "f1", "ece"]:
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    return df[columns], source


def evidence_fig(df: pd.DataFrame) -> go.Figure:
    summary = (
        df.dropna(subset=["score"])
        .groupby(["split", "model_variant"], as_index=False)["score"]
        .mean()
    )
    fig = go.Figure()
    palette = {
        "baseline": "#ff9f43",
        "multimodal_ssl": "#4fc3ff",
        "physics_aware_fm": "#54f0c4",
    }
    for variant in summary["model_variant"].unique():
        subset = summary[summary["model_variant"] == variant]
        fig.add_bar(
            name=variant,
            x=subset["split"],
            y=subset["score"],
            marker_color=palette.get(variant, "#f8fbff"),
        )
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(16,18,24,0.88)",
        margin=dict(l=20, r=20, t=25, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
        yaxis_title="Observed score from experiment logs",
        font=dict(color="#ecf2ff"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def summarize_experiment_evidence(df: pd.DataFrame) -> dict:
    observed = df.dropna(subset=["score"]).copy()
    summary = {
        "runs": 0,
        "use_cases": 0,
        "splits": 0,
        "variants": 0,
        "best_variant": "planned",
        "best_score": None,
        "avg_gain": None,
    }
    if observed.empty:
        return summary

    summary["runs"] = int(len(observed))
    summary["use_cases"] = int(observed["use_case"].nunique())
    summary["splits"] = int(observed["split"].nunique())
    summary["variants"] = int(observed["model_variant"].nunique())

    best_row = observed.sort_values("score", ascending=False).iloc[0]
    summary["best_variant"] = str(best_row["model_variant"])
    summary["best_score"] = float(best_row["score"])

    baseline = (
        observed[observed["model_variant"] == "baseline"]
        .groupby("split")["score"]
        .mean()
        .rename("baseline")
    )
    physics = (
        observed[observed["model_variant"] == "physics_aware_fm"]
        .groupby("split")["score"]
        .mean()
        .rename("physics_aware_fm")
    )
    aligned = pd.concat([baseline, physics], axis=1).dropna()
    if not aligned.empty:
        summary["avg_gain"] = float((aligned["physics_aware_fm"] - aligned["baseline"]).mean())

    return summary


def build_pilot_findings(df: pd.DataFrame) -> list[str]:
    observed = df.dropna(subset=["score"]).copy()
    if observed.empty:
        return []

    findings: list[str] = []
    baseline_rows = observed[observed["model_variant"] == "baseline"]
    if not baseline_rows.empty:
        weakest = baseline_rows.sort_values("score").iloc[0]
        findings.append(
            f"The image-only baseline is weakest on {weakest['split'].lower()} with score {weakest['score']:.2f}, which supports the argument that agricultural shift is not solved by vision-only modeling."
        )

    comparison = observed.pivot_table(index="split", columns="model_variant", values="score", aggfunc="mean")
    if {"baseline", "multimodal_ssl"}.issubset(comparison.columns):
        gains = (comparison["multimodal_ssl"] - comparison["baseline"]).dropna()
        if not gains.empty:
            top_split = gains.idxmax()
            findings.append(
                f"The multimodal model shows its clearest advantage on {top_split.lower()}, improving over the baseline by about {gains.loc[top_split]:.2f} score points."
            )

    missing_rows = observed[observed["split"].astype(str).str.lower() == "missing modality"]
    if not missing_rows.empty:
        weakest_missing = missing_rows.sort_values("score").iloc[0]
        findings.append(
            f"The missing-modality setting remains a meaningful failure case, which gives you an honest limitation to discuss instead of pretending the prototype solves everything."
        )

    physics_rows = observed[observed["model_variant"] == "physics_aware_fm"]
    if not physics_rows.empty:
        climate_rows = physics_rows[physics_rows["split"].astype(str).str.lower() == "climate-stress transfer"]
        if not climate_rows.empty:
            note = str(climate_rows.iloc[0]["notes"])
            if "physics_mae=" in note:
                physics_mae = note.split("physics_mae=")[-1]
                findings.append(
                    f"The physics-aware model also reports an auxiliary scientific-consistency target, which makes the pilot feel closer to a real scientific-ML workflow than a plain classifier benchmark."
                )

    return findings[:4]


def build_repo_assets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Interactive portfolio app", "app.py", "Shows project framing, benchmark logic, export tooling, and application-ready storytelling."],
            ["Benchmark utility", "eval/ood_benchmark.py", "Demonstrates that the repo includes reusable evaluation code rather than slides only."],
            ["CLI experiment runner", "experiments/run_ood_benchmark.py", "Signals reproducibility and command-line workflow readiness."],
            ["Synthetic study trainer", "experiments/train_dummy_multimodal.py", "Runs a domain-inspired synthetic agricultural benchmark with baseline, multimodal, and physics-aware variants."],
            ["Baseline models", "models/multimodal_baseline.py", "Adds evidence of PyTorch-oriented architectural thinking and physics-aware model design."],
            ["Technical report", "docs/technical_report.md", "Provides a supervisor-friendly written explanation of the repo's evidence, limits, and benchmark workflow."],
        ],
        columns=["Asset", "Path", "CV proof"],
    )


def build_recruiter_summary(case_name: str, novelty: str, summary: dict) -> str:
    evidence_line = (
        f"The current repo includes a reproducible synthetic pilot with {summary['runs']} structured experiment rows across {summary['splits']} OOD settings "
        f"and {summary['variants']} model variants to demonstrate the evaluation workflow honestly."
        if summary["runs"]
        else "The repo currently focuses on research framing, benchmark design, and implementation scaffolding rather than final real-data results."
    )
    return (
        "This project adds CV value because it looks like a research portfolio, not only a coursework app. "
        f"It frames {case_name.lower()} through multimodal agricultural foundation models with a {novelty} emphasis, "
        "connects the idea to named WUR supervisors, including responsible AI, retrieval-oriented multimodal learning, and scientific adaptation, and backs the narrative with benchmark scaffolding, prototype code, and reusable application materials. "
        f"{evidence_line}"
    )


def build_cv_bullets(case_name: str, novelty: str, summary: dict) -> list[str]:
    evidence_clause = (
        f"Created a reusable OOD evaluation scaffold covering {summary['splits']} shift settings, {summary['variants']} model variants, and exportable benchmark reports, supported by a reproducible synthetic pilot study."
        if summary["runs"]
        else "Created a reusable OOD evaluation scaffold for region, year, sensor, and missing-modality robustness studies in agricultural ML."
    )
    return [
        (
            f"Designed a research-oriented portfolio project on agriculture-specific foundation models for {case_name.lower()}, "
            f"combining G-E-M multimodal alignment, {novelty}-driven modeling, retrieval-aware representation learning, and WUR-targeted PhD positioning."
        ),
        (
            "Built a Streamlit-based research interface that generates CV-ready abstracts, a one-page proposal, a motivation letter, "
            "and interview wording from the same technical project narrative."
        ),
        evidence_clause,
        (
            "Structured the repository with benchmark utilities, prototype training scripts, visual research framing, technical documentation, and evidence artifacts "
            "to present Python, PyTorch, scikit-learn, multimodal analysis, and scientific communication readiness in one place."
        ),
    ]


def build_interview_points(case_name: str, summary: dict) -> list[str]:
    points = [
        "I built this project to show that I can define a research question, not just implement isolated models.",
        f"The flagship use case is {case_name.lower()}, but the pipeline is intentionally designed so the evaluation logic can transfer across agricultural tasks.",
        "The repo is honest about current limitations: it distinguishes proposal framing, scaffolded evidence, and not-yet-real experimental claims.",
        "What makes the project strong is the combination of multimodal SSL, physics-aware reasoning, and OOD robustness rather than any single component in isolation.",
    ]
    if summary["avg_gain"] is not None:
        points.append(
            f"In the current reproducible synthetic pilot, the physics-aware variant is about {summary['avg_gain']:.1f} points better than the baseline on average across shared OOD splits."
        )
    return points


def build_cv_value_pack(case_name: str, priority: str, novelty: str, profile: dict, summary: dict) -> str:
    bullets = build_cv_bullets(case_name, novelty, summary)
    interview_points = build_interview_points(case_name, summary)
    assets = build_repo_assets()
    assets_text = "\n".join(
        f"- {row.Asset}: {row.Path} - {row['CV proof']}"
        for _, row in assets.iterrows()
    )
    bullet_text = "\n".join(f"- {bullet}" for bullet in bullets)
    interview_text = "\n".join(f"- {point}" for point in interview_points)
    return (
        "CV Value Pack\n"
        f"Applicant: {profile['name']}\n"
        f"Target emphasis: {priority}\n"
        f"Flagship use case: {case_name}\n"
        f"Main novelty: {novelty}\n\n"
        "Recruiter Summary\n"
        f"{build_recruiter_summary(case_name, novelty, summary)}\n\n"
        "CV Bullets\n"
        f"{bullet_text}\n\n"
        "Interview Proof Points\n"
        f"{interview_text}\n\n"
        "Repo Assets To Mention\n"
        f"{assets_text}\n"
    )


def render_poster(case_name: str) -> None:
    case = USE_CASES[case_name]
    st.markdown(
        f"""
        <div class="poster-grid">
            <div class="poster-panel">
                <div class="poster-tag">Problem</div>
                <h4>Agricultural Generalization Failure</h4>
                <div>Models often break under subtle shift across region, year, sensor, and management context. This project targets robust transfer instead of narrow in-domain gains.</div>
            </div>
            <div class="poster-panel">
                <div class="poster-tag">Inputs</div>
                <h4>G-E-M Signal Stack</h4>
                <div>Biological, environmental, and management signals are aligned using imagery, weather time series, field polygons, and metadata.</div>
            </div>
            <div class="poster-panel">
                <div class="poster-tag">Method</div>
                <h4>Multimodal SSL + Physics</h4>
                <div>Contrastive and masked self-supervision learn agriculture-specific representations, while crop-dynamics-inspired priors regularize adaptation.</div>
            </div>
            <div class="poster-panel">
                <div class="poster-tag">Flagship Use Case</div>
                <h4>{case_name}</h4>
                <div><strong>Tasks:</strong> {case["tasks"]}<br><br><strong>Modalities:</strong> {case["modalities"]}</div>
            </div>
            <div class="poster-panel">
                <div class="poster-tag">Evaluation</div>
                <h4>OOD Benchmark Design</h4>
                <div>Region shift, year shift, sensor shift, missing modality, and sparse-label finetuning are treated as first-class evaluation settings.</div>
            </div>
            <div class="poster-panel">
                <div class="poster-tag">Impact</div>
                <h4>WUR + AgriscienceFM Fit</h4>
                <div>Designed to contribute to food-security-relevant agricultural AI, benchmark construction, and foundation-model research grounded in scientific reasoning.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_pitch(case_name: str, priority: str, novelty: str) -> str:
    case = USE_CASES[case_name]
    novelty_map = {
        "self-supervision": "multimodal self-supervised pretraining and agriculture-specific latent representations",
        "physics": "physics-aware adaptation with crop-dynamics-inspired priors",
        "benchmarking": "out-of-distribution benchmarking across region, year, and modality shift",
    }
    priority_map = {
        "Athanasiadis": "responsible agricultural AI, food-system relevance, benchmark realism, and knowledge-guided learning",
        "Torres": "multimodal representation learning, retrieval, and visual analytics for agricultural remote sensing",
        "Kapoor": "scientific priors, physics-aware modeling, and trustworthy adaptation",
    }
    return (
        f"My proposed project focuses on {case_name.lower()}, where the core tasks are {case['tasks'].lower()}. "
        f"I want to build an agriculture-specific foundation-model pipeline that aligns biological, environmental, "
        f"and management signals through {novelty_map[novelty]}. The main scientific emphasis would be "
        f"{priority_map[priority]}, while the evaluation would center on robustness under real-world agricultural shift. "
        f"This makes the project a strong fit for the WUR PhD because it combines multimodal learning, domain-specific "
        f"foundation models, and scientifically grounded evaluation in one coherent agenda."
    )


def build_fit_statement(case_name: str, profile: dict) -> str:
    return (
        f"This project also demonstrates readiness for the role beyond the topic itself. Around the flagship use case "
        f"of {case_name.lower()}, it highlights Python-based ML development, PyTorch-centered modeling, scikit-learn "
        f"evaluation, remote sensing data handling, multimodal experiment design, and research communication. In other "
        f"words, it helps present {profile['name']} as motivated, self-driven, technically capable, and prepared to "
        f"contribute to both model building and collaborative benchmark development from the perspective of {profile['degree']} "
        f"work centered on {profile['thesis_topic']}."
    )


def build_athanasiadis_upgrade(case_name: str) -> str:
    case = USE_CASES[case_name]
    return (
        f"To align the project more directly with Prof. Athanasiadis, the strongest framing is to treat {case_name.lower()} "
        "as a benchmarked, knowledge-guided agricultural AI problem rather than only a multimodal modeling problem. "
        f"That means centering tasks such as {case['tasks'].lower()}, evaluating them under realistic climatic and geographic shift, "
        "and showing how process-based signals, agronomic priors, and responsible evaluation make the work more useful for real agricultural decisions."
    )


def build_athanasiadis_talking_points(case_name: str) -> list[str]:
    return [
        "This project treats agricultural AI as a responsible, use-inspired research problem connected to food systems and environmental stress.",
        f"The flagship use case of {case_name.lower()} is framed as a benchmark problem with realistic out-of-distribution evaluation, not just a single-model demo.",
        "The knowledge-guided layer is intentionally positioned as a bridge between crop modeling and modern representation learning.",
        "The repo is structured so it can grow toward crop-yield forecasting, plant phenomics, and digital-twin decision support without losing credibility.",
        "That makes the project feel closer to Athanasiadis' style of agricultural AI: benchmarked, domain-aware, collaborative, and practically grounded.",
    ]


def build_torres_upgrade(case_name: str) -> str:
    case = USE_CASES[case_name]
    return (
        f"To align the project more directly with Prof. Ricardo da Silva Torres, {case_name.lower()} should be framed not only as a prediction task, "
        "but also as a multimodal data-science problem involving representation learning, retrieval, visual exploration, and scalable benchmark analysis. "
        f"That means using modalities such as {case['modalities'].lower()} to learn latent spaces that support both downstream prediction and cross-modal search, explanation, and failure-case analysis."
    )


def build_three_professor_summary(case_name: str) -> str:
    return (
        f"The most advanced version of this project treats {case_name.lower()} as a three-layer research agenda: "
        "Athanasiadis gives the responsible agricultural benchmark and food-system framing, Torres gives the multimodal representation, retrieval, and visual analytics layer, "
        "and Kapoor gives the scientific-machine-learning layer for physics-aware, trustworthy adaptation. Together, that makes the portfolio much more distinctive than a standard remote-sensing project."
    )


def build_three_professor_cv_line(case_name: str) -> str:
    return (
        f"Built a three-supervisor-aligned research portfolio around {case_name.lower()}, combining responsible agricultural AI, multimodal retrieval and representation learning, "
        "and physics-aware scientific adaptation for robust foundation models."
    )


def build_pain_point_summary(case_name: str) -> str:
    return (
        f"The strongest version of this project treats {case_name.lower()} as a response to the exact pain points in the vacancy: "
        "subtle distribution shift, fragmented multimodal data, weak agriculture-specific latent structure, lack of scientific grounding, and the need for dynamic intelligent systems that matter for food security. "
        "Each of those pain points is answered through one integrated agenda: benchmark realism from Athanasiadis, retrieval-aware multimodal learning from Torres, and trustworthy scientific adaptation from Kapoor."
    )


def build_advanced_project_line(case_name: str) -> str:
    return (
        f"This is not only a project about {case_name.lower()}; it is an advanced agricultural AI platform concept that combines multimodal foundation modeling, cross-modal retrieval, visual benchmark analytics, "
        "and physics-aware scientific reliability in one coherent PhD narrative."
    )


def build_vacancy_fit(case_name: str) -> str:
    case = USE_CASES[case_name]
    return (
        "This project now matches the vacancy more directly by framing the work around domain-specific agricultural foundation models, "
        "multimodal heterogeneous data, image and time-series architectures, and rigorous out-of-distribution evaluation. "
        f"The selected flagship use case, {case_name.lower()}, covers tasks such as {case['tasks'].lower()}, which are explicitly named or strongly implied by the vacancy."
    )


def build_application_checklist(case_name: str, profile: dict) -> str:
    return (
        "WUR Application Checklist\n"
        f"Applicant: {profile['name']}\n"
        f"Flagship use case: {case_name}\n\n"
        "Documents to prepare\n"
        "- CV: maximum 3 pages\n"
        "- Motivation letter: maximum 3 pages\n"
        "- Scientific writing sample written by you: maximum 3 pages\n\n"
        "Do not include now\n"
        "- Grades or transcripts are not required at this stage\n"
        "- Extra files outside the requested set may be ignored\n\n"
        "Project points to emphasize\n"
        "- Self-supervised agricultural foundation models\n"
        "- Multimodal data integration across images, location, text, and time series\n"
        "- OOD robustness across region, year, sensor, and missing-modality shift\n"
        "- Python, PyTorch, scikit-learn, and reproducible benchmark workflow\n"
        "- Research communication through proposal, technical report, and application-ready exports\n\n"
        "Dates to plan around\n"
        "- Safer deadline target: May 4, 2026\n"
        "- Vacancy footer also shows: May 5, 2026\n"
        "- First interviews scheduled: May 15, 2026\n"
    )


def build_one_page_proposal(case_name: str, priority: str, novelty: str, profile: dict) -> str:
    case = USE_CASES[case_name]
    pitch = build_pitch(case_name, priority, novelty)
    fit = build_fit_statement(case_name, profile)
    return (
        "One-Page Research Proposal\n"
        "Title: Physics-Aware Agricultural Foundation Models for Robust Multimodal Learning\n\n"
        f"Applicant: {profile['name']}\n"
        f"Background: {profile['degree']}\n"
        f"Current focus: {profile['thesis_topic']}\n"
        f"Profile link: {profile['github']}\n\n"
        "Target Position\n"
        "PhD Position - Foundation Models for Agricultural Sciences, Wageningen University & Research\n\n"
        "Motivation\n"
        "Agricultural machine learning models often fail under subtle distribution shifts across regions, seasons, "
        "sensors, and management settings. This makes it difficult to deploy reliable AI systems for food-security-"
        "relevant agricultural applications. The proposed project addresses this challenge by studying domain-specific "
        "foundation models for agriculture that are explicitly designed for multimodal heterogeneous data and robust evaluation.\n\n"
        "Project Focus\n"
        f"{pitch}\n\n"
        "Methodological Plan\n"
        "- Align biological, environmental, and management signals through a G-E-M formulation inspired by AgriscienceFM.\n"
        "- Use multimodal self-supervised learning over imagery, weather time series, geospatial context, and metadata.\n"
        "- Introduce physics-aware regularization through crop-dynamics-inspired priors during adaptation.\n"
        "- Evaluate robustness under region shift, year shift, sensor shift, missing-modality settings, and sparse labels.\n\n"
        "Flagship Use Case\n"
        f"- Use case: {case_name}\n"
        f"- Tasks: {case['tasks']}\n"
        f"- Modalities: {case['modalities']}\n\n"
        "Expected Research Outputs\n"
        "- A multimodal pretraining pipeline for agricultural data.\n"
        "- A physics-aware adaptation framework for robust transfer.\n"
        "- An OOD benchmark suite for agricultural foundation models.\n"
        "- Publishable insights on when multimodal and scientific priors improve generalization.\n\n"
        "Why I Fit This Position\n"
        f"{fit}\n"
    )


def build_motivation_letter(case_name: str, priority: str, novelty: str, profile: dict) -> str:
    case = USE_CASES[case_name]
    priority_map = {
        "Athanasiadis": "responsible agricultural AI, benchmark realism, and knowledge-guided learning",
        "Torres": "multimodal self-supervised representation learning, retrieval, and visual analytics for remote sensing",
        "Kapoor": "physics-informed scientific machine learning for robust and trustworthy adaptation",
    }
    novelty_map = {
        "self-supervision": "multimodal self-supervised learning",
        "physics": "physics-aware adaptation",
        "benchmarking": "out-of-distribution benchmarking",
    }
    return (
        "Dear Prof. Athanasiadis and selection committee,\n\n"
        "I am writing to apply for the PhD Position on Foundation Models for Agricultural Sciences at Wageningen "
        f"University & Research. I recently completed {profile['degree']}, with work centered on "
        f"{profile['thesis_topic']}. The position strongly matches my research interests in multimodal machine learning, "
        "robust generalization, and scientific AI for real-world applications.\n\n"
        "What attracts me most to this PhD is the opportunity to study domain-specific foundation models for "
        "agriculture in a setting where methodological depth and practical relevance clearly go together. I am "
        "particularly interested in the challenge highlighted in the vacancy: current AI models often fail to "
        "generalize when agricultural data shifts across regions, seasons, sensing conditions, and management "
        "contexts. I would be excited to investigate how agriculture-specific latent representations can be learned "
        f"from multimodal heterogeneous data through {novelty_map[novelty]}, and how such models can be evaluated "
        "more rigorously under realistic distribution shifts.\n\n"
        f"Within this broader direction, I am especially drawn to {case_name.lower()}, including tasks such as "
        f"{case['tasks'].lower()}. I find this use case compelling because it naturally combines image data, time-series "
        "signals, geospatial context, and application relevance for food security and agricultural decision support. "
        "The AgriscienceFM vision is especially motivating to me because the G-E-M framing offers a clear way to align "
        "biological, environmental, and management signals into a coherent foundation-model research agenda.\n\n"
        f"I also see a strong intellectual fit with the supervisory team. I am excited by Prof. Athanasiadis' focus on "
        f"agricultural AI and knowledge-guided learning, Prof. Ricardo da Silva Torres' work on "
        f"{priority_map['Torres']}, and Dr. Taniya Kapoor's perspective on "
        f"{priority_map['Kapoor']}. Together, these directions make this PhD especially appealing because they support "
        "a research program that is at once methodologically ambitious, scientifically grounded, and highly relevant "
        "to agricultural applications.\n\n"
        "Through this project, I would aim to contribute not only to model development, but also to benchmark design, "
        "careful experimental evaluation, and clear scientific communication. I am motivated by research problems that "
        "require both technical depth and interdisciplinary collaboration, and I would value contributing to the "
        "AgriscienceFM effort in a team-oriented and rigorous way.\n\n"
        "Thank you for considering my application. I would be very happy to contribute my enthusiasm for multimodal "
        f"machine learning, {priority_map[priority]}, and robust agricultural AI to this PhD position.\n\n"
        "Sincerely,\n"
        f"{profile['name']}\n"
        f"{profile['email']}\n"
        f"{profile['github']}"
    )


style()

with st.sidebar:
    st.title("Project Map")
    selected_case = st.selectbox("Flagship use case", list(USE_CASES.keys()))
    supervisor_priority = st.selectbox("Strongest pitch angle", ["Athanasiadis", "Torres", "Kapoor"])
    novelty_priority = st.selectbox("Main novelty emphasis", ["self-supervision", "physics", "benchmarking"])
    st.markdown("---")
    st.markdown("### Applicant Profile")
    applicant_name = st.text_input("Name", value="[Your Name]")
    applicant_degree = st.text_input("Degree", value="MSc in Artificial Intelligence")
    applicant_thesis = st.text_input("Thesis or current focus", value="multimodal machine learning for remote sensing")
    applicant_email = st.text_input("Email", value="your.email@example.com")
    applicant_github = st.text_input("GitHub or portfolio", value="https://github.com/yourname")
    uploaded_log_file = st.file_uploader("Experiment log CSV", type=["csv"])
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Research Fit",
            "Pain Point Engine",
            "Athanasiadis Upgrade",
            "Torres Upgrade",
            "Three-Professor Blueprint",
            "Advanced Agenda",
            "Method Blueprint",
            "Poster Architecture",
            "Experimental Plan",
            "Vacancy Match",
            "Dataset Strategy",
            "Candidate Fit",
            "CV Value Pack",
            "Supervisor Alignment",
            "Proposal Builder",
            "One-Page Proposal",
            "Motivation Letter",
            "Application Pitch",
            "References",
        ],
    )
    st.caption("A WUR-targeted, professor-aware research portfolio app.")
    st.markdown("---")
    st.markdown("**Best one-line summary**")
    st.markdown("Agriculture-specific multimodal foundation models with OOD robustness and physics-aware adaptation.")

profile = {
    "name": applicant_name,
    "degree": applicant_degree,
    "thesis_topic": applicant_thesis,
    "email": applicant_email,
    "github": applicant_github,
}

if page == "Overview":
    st.markdown(
        """
        <div class="hero">
            <div class="kicker">Professor-aware PhD pitch</div>
            <div class="hero-title">Physics-Aware Agricultural Foundation Models</div>
            <div class="muted">
                A proposal-driven research app for the WUR PhD on foundation models for agricultural sciences.
                The project combines AgriscienceFM's G-E-M framing, multimodal self-supervised learning for
                remote sensing, retrieval-aware data science, and physics-aware scientific machine learning for
                robust agricultural prediction and analysis under geographic, temporal, climatic, and sensor shift.
            </div>
            <div class="muted" style="margin-top:0.8rem;">
                This app presents a research direction and benchmark plan. It does not present unpublished claims as finished results.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("Project structure", "G-E-M", "Matches AgriscienceFM framing")
    with c2:
        metric("Research pillars", "3", "SSL + OOD + physics")
    with c3:
        metric("Portfolio signal", "PhD fit", "Research pitch, not demo fluff")
    with c4:
        metric("Application value", "High", "Depth and focus together")

    st.markdown("### Vacancy pain points")
    cols = st.columns(len(PAIN_POINTS))
    for col, (title, text) in zip(cols, PAIN_POINTS):
        with col:
            card(title, text)

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("### Unified project thesis")
        st.markdown(
            """
            Build a domain-specific agricultural foundation-model pipeline that jointly learns from
            biological, environmental, and management signals. Use multimodal self-supervised learning
            to learn transferable agricultural representations, support retrieval and visual benchmark
            exploration, then improve reliability with physics-aware adaptation and explicit
            out-of-distribution evaluation.
            """
        )
        st.markdown("### Why this pitch is stronger")
        st.markdown(
            """
            - It mirrors the vacancy closely without sounding copied.
            - It combines the three supervisors in one coherent research arc.
            - It shows a realistic benchmark and failure-analysis plan instead of unsupported claims.
            - It adds retrieval and visual analytics, which makes the project more original and more aligned with Torres.
            - It gives reusable wording for your CV, portfolio, and motivation letter.
            """
        )
    with right:
        st.plotly_chart(bar_fig(), use_container_width=True)
        st.caption("Illustrative targets for the proposal narrative, not claimed published results.")

elif page == "Research Fit":
    st.header("Research Fit")
    tabs = st.tabs(["Professor Signals", "AgriscienceFM Lens", "Novelty"])
    with tabs[0]:
        for item in SUPERVISORS:
            with st.expander(item["name"], expanded=True):
                st.markdown(f"**Research fit:** {item['focus']}")
                st.markdown(f"**Project role:** {item['fit']}")
                st.markdown(f"**Application wording:** {item['line']}")
    with tabs[1]:
        st.markdown(
            """
            AgriscienceFM describes agriculture through three major drivers: **G** for biological material,
            **E** for environment, and **M** for management. That gives your project a much more advanced
            frame than a generic "multimodal model" story.
            """
        )
        st.dataframe(GEM, use_container_width=True, hide_index=True)
        st.info("Best pitch line: I want to study how G-E-M-aligned multimodal foundation models can improve transfer, robustness, and scientific usefulness across agricultural tasks.")
    with tabs[2]:
        st.markdown(
            """
            - Agriculture-specific pretraining instead of generic latent spaces.
            - Explicit multimodal alignment across G, E, and M drivers.
            - Scientific priors during adaptation rather than only post-hoc interpretation.
            - OOD benchmarking as a first-class goal.
            """
        )

elif page == "Pain Point Engine":
    st.header("Pain Point Engine")
    st.markdown(
        """
        This page turns the vacancy's core problems into concrete research responses. It is designed to show
        that the project is solving the actual scientific and application pain points of the position, not just
        repeating keywords from the advertisement.
        """
    )
    st.info(build_pain_point_summary(selected_case))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("Pain points", "6", "Directly mapped to methods")
    with c2:
        metric("Supervisor roles", "3", "Each one solves a real gap")
    with c3:
        metric("Benchmark tasks", "6", "More realistic than a single demo")
    with c4:
        metric("Portfolio depth", "Advanced", "From prediction to scientific analysis")

    tab1, tab2, tab3 = st.tabs(["Pain Points", "Benchmark Matrix", "Method Roles"])
    with tab1:
        st.dataframe(POSITION_PAINPOINT_ENGINE, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(BENCHMARK_MATRIX, use_container_width=True, hide_index=True)
        st.markdown(
            """
            Best message:
            the project is valuable because it does not optimize for one benchmark score only. It builds an evaluation culture
            around the tasks, shifts, and failure modes that matter in agricultural science.
            """
        )
    with tab3:
        st.dataframe(PROFESSOR_METHOD_MATRIX, use_container_width=True, hide_index=True)
        st.dataframe(ADVANCED_RESEARCH_QUESTIONS, use_container_width=True, hide_index=True)

elif page == "Athanasiadis Upgrade":
    st.header("Athanasiadis Upgrade")
    st.markdown(
        """
        This page sharpens the project toward Prof. Ioannis Athanasiadis' research style:
        responsible agricultural AI, benchmark realism, knowledge-guided machine learning,
        climate-stress-aware crop modeling, and collaboration with domain experts.
        """
    )
    st.info(build_athanasiadis_upgrade(selected_case))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("Professor fit", "Stronger", "More benchmarked and applied")
    with c2:
        metric("Athanasiadis themes", "7", "From AgML to digital twins")
    with c3:
        metric("Responsible AI", "Explicit", "Now visible in the project")
    with c4:
        metric("Upgrade path", "Hybrid", "Foundation models + crop knowledge")

    tab1, tab2, tab3, tab4 = st.tabs(["Research Signals", "Benchmark Upgrade", "Responsible AI", "Roadmap"])
    with tab1:
        st.dataframe(ATHANASIADIS_SIGNALS, use_container_width=True, hide_index=True)
        st.markdown("### Best oral framing")
        st.markdown(
            """
            I want to build agricultural AI that is not only accurate, but also responsibly benchmarked,
            scientifically grounded, and useful under the kinds of climate and management variation that matter in practice.
            """
        )
    with tab2:
        st.dataframe(BENCHMARK_PRINCIPLES, use_container_width=True, hide_index=True)
        st.markdown("### Project shift")
        st.markdown(
            """
            The app now reads more like a benchmark-and-method agenda for agricultural AI.
            That is a better match for Athanasiadis than a purely architecture-centered pitch.
            """
        )
    with tab3:
        st.dataframe(RESPONSIBLE_AI, use_container_width=True, hide_index=True)
        st.markdown("### Why this matters")
        st.markdown(
            """
            His group explicitly aims to advance AI methods for global challenges in a responsible way.
            Showing this mindset makes your project look more mature and more aligned with the chair group.
            """
        )
    with tab4:
        st.dataframe(HYBRID_ROADMAP, use_container_width=True, hide_index=True)
        st.markdown("### Interview talking points")
        for point in build_athanasiadis_talking_points(selected_case):
            st.markdown(f"- {point}")

elif page == "Torres Upgrade":
    st.header("Torres Upgrade")
    st.markdown(
        """
        This page sharpens the project toward Prof. Ricardo da Silva Torres' research style:
        data science, visual computing, multimedia analysis and retrieval, information visualisation,
        and scalable multimodal research workflows.
        """
    )
    st.info(build_torres_upgrade(selected_case))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("Torres fit", "Stronger", "More data-centric and visual")
    with c2:
        metric("Torres themes", "6", "Retrieval to eScience")
    with c3:
        metric("New angle", "Retrieval", "Not only classification")
    with c4:
        metric("Project edge", "Unique", "Visual analytics + benchmarks")

    tab1, tab2, tab3 = st.tabs(["Research Signals", "Project Upgrade", "CV Value"])
    with tab1:
        st.dataframe(TORRES_SIGNALS, use_container_width=True, hide_index=True)
    with tab2:
        st.markdown(
            """
            What changes in the project when Torres is taken seriously as a supervisor:
            - the model should learn representations that support retrieval as well as prediction
            - the repo should support exploratory visual analytics, not just final scores
            - multimodal learning should include text, metadata, and location as first-class views
            - the benchmark should help users inspect failure cases and cross-modal similarities
            """
        )
        st.dataframe(UNIQUE_ADVANTAGES, use_container_width=True, hide_index=True)
    with tab3:
        st.code(build_three_professor_cv_line(selected_case), language="text")
        st.markdown(
            """
            Strong interview line:
            I want this project to show not only that I can train multimodal models, but also that I can
            structure agricultural data for retrieval, exploration, and scientific analysis in a reusable way.
            """
        )

elif page == "Three-Professor Blueprint":
    st.header("Three-Professor Blueprint")
    st.markdown(
        """
        This is the project's most distinctive layer: one coherent research agenda that combines
        responsible agricultural AI, multimodal retrieval-oriented representation learning, and
        physics-aware scientific adaptation while staying tightly matched to the WUR PhD vacancy.
        """
    )
    st.info(build_three_professor_summary(selected_case))
    tab1, tab2, tab3, tab4 = st.tabs(["Synergy Map", "Kapoor Layer", "Distinctive Value", "CV Line"])
    with tab1:
        st.dataframe(SUPERVISOR_SYNERGY, use_container_width=True, hide_index=True)
        st.markdown(
            """
            Best synthesis:
            Athanasiadis gives the real agricultural benchmark problem, Torres gives the multimodal data-and-retrieval engine,
            and Kapoor gives the scientific reliability layer. That combination makes the project feel unusually complete.
            """
        )
    with tab2:
        st.dataframe(KAPOOR_SIGNALS, use_container_width=True, hide_index=True)
        st.dataframe(RESPONSIBLE_AI, use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(UNIQUE_ADVANTAGES, use_container_width=True, hide_index=True)
        st.markdown(
            """
            This is the uniqueness you can say on your CV:
            the project is not just about agricultural prediction, but about building a science-facing foundation-model
            portfolio that supports benchmark design, retrieval, visual exploration, and trustworthy adaptation.
            """
        )
    with tab4:
        tri_line = build_three_professor_cv_line(selected_case)
        st.text_area("Three-professor CV line", value=tri_line, height=110)
        st.text_area(
            "Three-professor oral pitch",
            value=(
                "I designed this project so that each supervisor's expertise has a real technical role: "
                "Athanasiadis for benchmarked agricultural impact, Torres for multimodal retrieval and visual analytics, "
                "and Kapoor for physics-aware scientific reliability."
            ),
            height=130,
        )

elif page == "Advanced Agenda":
    st.header("Advanced Agenda")
    st.markdown(
        """
        This page presents the upgraded project as a more advanced research platform. It shows how the project moves
        beyond a normal PhD application demo into a reusable scientific workflow for multimodal agricultural AI.
        """
    )
    st.info(build_advanced_project_line(selected_case))
    tab1, tab2, tab3 = st.tabs(["Research Modules", "Advanced Questions", "Distinctive Value"])
    with tab1:
        st.dataframe(ADVANCED_RESEARCH_MODULES, use_container_width=True, hide_index=True)
        st.markdown(
            """
            Strongest technical framing:
            the project combines representation learning, retrieval, scientific regularization, benchmark design,
            and visual analytics in one modular architecture.
            """
        )
    with tab2:
        st.dataframe(ADVANCED_RESEARCH_QUESTIONS, use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(UNIQUE_ADVANTAGES, use_container_width=True, hide_index=True)
        st.code(build_three_professor_cv_line(selected_case), language="text")

elif page == "Method Blueprint":
    st.header("Method Blueprint")
    st.plotly_chart(sankey_fig(), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown("### Pretraining objectives")
        st.markdown(
            """
            - Cross-modal contrastive learning between satellite imagery and weather context
            - Masked modeling for imagery patches and temporal segments
            - Retrieval-style alignment between image, field, and metadata views
            - Text-image-location alignment for agricultural reports, field notes, and geospatial context
            - Region-aware negatives or semantically aware positives for remote sensing
            """
        )
        st.markdown("### Advanced outputs")
        st.markdown(
            """
            - Predictive heads for crop tasks such as yield, disease, and crop failure
            - Retrieval heads for similar-case search across image, metadata, and text
            - Visual analytics views for failure clusters, shift diagnosis, and benchmark inspection
            - Scientific consistency heads for plausibility under climatic and agronomic variation
            """
        )
        st.markdown("### Physics-aware layer")
        st.markdown(
            """
            Use simple crop-growth priors such as accumulated heat, radiation-related biomass trends,
            or seasonal plausibility checks as regularizers. The goal is not to hard-code a crop model,
            but to encourage more stable and scientifically plausible representations.
            """
        )
    with right:
        st.markdown("### Example model block")
        st.code(
            """image_tokens = image_encoder(sentinel_patch)
weather_tokens = temporal_encoder(weather_cube)
management_tokens = metadata_encoder(field_geom, text_meta)
text_tokens = text_encoder(field_report)

joint_repr = cross_modal_fuser(image_tokens, weather_tokens, management_tokens, text_tokens)
ssl_loss = contrastive_loss(joint_repr) + masked_modeling_loss(joint_repr)
retrieval_loss = cross_modal_retrieval_loss(joint_repr, field_id)
physics_loss = crop_dynamics_regularizer(joint_repr, gdd, radiation, season_index)
analytics_head = benchmark_visual_analytics(joint_repr, split_id, task_id)

loss = ssl_loss + 0.2 * physics_loss + 0.15 * retrieval_loss
""",
            language="python",
        )
        st.dataframe(ADVANCED_RESEARCH_MODULES, use_container_width=True, hide_index=True)
    use_case = USE_CASES[selected_case]
    a, b, c = st.columns(3)
    with a:
        card("Candidate tasks", use_case["tasks"])
    with b:
        card("Core modalities", use_case["modalities"])
    with c:
        card("Why this is compelling", use_case["pitch"])

elif page == "Poster Architecture":
    st.header("Poster Architecture")
    st.markdown(
        """
        This page is designed as a visual interview aid: a compact poster-style summary of the project
        that you can talk through in a few minutes.
        """
    )
    render_poster(selected_case)
    st.plotly_chart(sankey_fig(), use_container_width=True)

elif page == "Experimental Plan":
    st.header("Experimental Plan")
    experiment_logs, log_source = load_experiment_logs(uploaded_log_file)
    has_evidence = not experiment_logs.dropna(subset=["score"]).empty
    pilot_findings = build_pilot_findings(experiment_logs)
    tab1, tab2, tab3, tab4 = st.tabs(["OOD Benchmarks", "Ablations", "ML Stack", "Hypotheses"])
    with tab1:
        if log_source and has_evidence:
            st.success(f"Loaded experiment evidence from {log_source}.")
            if "Domain-inspired synthetic agricultural benchmark" in experiment_logs["use_case"].astype(str).values:
                st.caption("This evidence comes from a reproducible synthetic pilot with domain-inspired crop, weather, management, and shift structure.")
            st.plotly_chart(evidence_fig(experiment_logs), use_container_width=True)
            st.dataframe(experiment_logs, use_container_width=True, hide_index=True)
            if pilot_findings:
                st.markdown("### What the pilot currently shows")
                for finding in pilot_findings:
                    st.markdown(f"- {finding}")
        else:
            st.info("No experiment log detected yet. The app falls back to the planned benchmark structure below. Add `data/experiment_runs.csv` or upload a CSV from the sidebar.")
            st.dataframe(OOD, use_container_width=True, hide_index=True)
            st.plotly_chart(heatmap_fig(), use_container_width=True)
        st.markdown("**Core message:** optimize for transfer and robustness, not only in-domain accuracy.")
        st.dataframe(BENCHMARK_PRINCIPLES, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(ABLATIONS, use_container_width=True, hide_index=True)
        st.markdown(
            """
            These ablations answer publishable questions:
            Does multimodal fusion beat image-only SSL?
            Do scientific priors improve robustness or calibration?
            Which modalities matter most under label scarcity and shift?
            """
        )
    with tab3:
        st.dataframe(STACK, use_container_width=True, hide_index=True)
    with tab4:
        st.dataframe(HYPOTHESES, use_container_width=True, hide_index=True)
        st.dataframe(RISKS, use_container_width=True, hide_index=True)

elif page == "Vacancy Match":
    st.header("Vacancy Match")
    st.markdown(
        """
        This page maps the project directly to the actual WUR PhD vacancy so the portfolio reads like a serious,
        position-specific application asset rather than a generic research demo.
        """
    )
    st.info(build_vacancy_fit(selected_case))
    tab1, tab2, tab3, tab4 = st.tabs(["Position Scope", "Duties", "Application Pack", "Timeline"])
    with tab1:
        st.dataframe(VACANCY_SCOPE, use_container_width=True, hide_index=True)
        st.dataframe(POSITION_PAINPOINT_ENGINE, use_container_width=True, hide_index=True)
        st.markdown(
            """
            Best framing:
            this project is about agriculture-specific foundation models that learn from multimodal heterogeneous data,
            especially image and time-series signals, are extended with retrieval and analysis capabilities, and are evaluated on tasks that matter for food security and agricultural decisions.
            """
        )
    with tab2:
        st.dataframe(DUTY_ALIGNMENT, use_container_width=True, hide_index=True)
        st.dataframe(POSITION_QUALITIES, use_container_width=True, hide_index=True)
        st.dataframe(PHD_CAPABILITIES, use_container_width=True, hide_index=True)
    with tab3:
        checklist = build_application_checklist(selected_case, profile)
        st.dataframe(APPLICATION_REQUIREMENTS, use_container_width=True, hide_index=True)
        st.text_area("Application checklist", value=checklist, height=320)
        st.download_button(
            "Download Application Checklist",
            data=checklist,
            file_name="wur_application_checklist.txt",
            mime="text/plain",
        )
    with tab4:
        st.dataframe(VACANCY_TIMELINE, use_container_width=True, hide_index=True)
        st.warning("The vacancy body states May 4, 2026, while the footer shows May 5, 2026. Verify the final deadline on the live submission page before applying.")

elif page == "Dataset Strategy":
    st.header("Dataset Strategy")
    st.markdown(
        """
        This section makes the project feel more executable. It shows which signals you would combine,
        why they matter scientifically, and which part of the supervisory team each signal naturally connects to.
        """
    )
    st.dataframe(DATASETS, use_container_width=True, hide_index=True)
    left, right = st.columns([1, 1])
    with left:
        card(
            "Flagship use case",
            f"{selected_case}<br><br><strong>Tasks:</strong> {USE_CASES[selected_case]['tasks']}",
        )
    with right:
        card(
            "Current rationale",
            f"<strong>Main novelty:</strong> {novelty_priority}<br><br><strong>Pitch emphasis:</strong> {supervisor_priority}",
        )

elif page == "Candidate Fit":
    st.header("Candidate Fit")
    st.markdown(
        """
        This section connects the project to the qualities and capabilities WUR explicitly asks for.
        It helps the app show not only *what* you want to research, but also *why you look prepared to do it*.
        """
    )
    tab1, tab2, tab3 = st.tabs(["Qualities", "Capabilities", "Libraries"])
    with tab1:
        st.dataframe(POSITION_QUALITIES, use_container_width=True, hide_index=True)
        st.dataframe(DUTY_ALIGNMENT, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(PHD_CAPABILITIES, use_container_width=True, hide_index=True)
        st.info(build_fit_statement(selected_case, profile))
    with tab3:
        st.dataframe(STACK_DETAILS, use_container_width=True, hide_index=True)
        st.dataframe(RESPONSIBLE_AI, use_container_width=True, hide_index=True)
        st.markdown(
            """
            Best message to send:
            this is not just a topic-aligned project, it is also a capability-aligned project that
            demonstrates Python, PyTorch, scikit-learn, remote sensing, time-series handling,
            experiment design, and scientific writing readiness.
            """
        )

elif page == "CV Value Pack":
    st.header("CV Value Pack")
    experiment_logs, _ = load_experiment_logs(uploaded_log_file)
    evidence_summary = summarize_experiment_evidence(experiment_logs)
    cv_bullets = build_cv_bullets(selected_case, novelty_priority, evidence_summary)
    interview_points = build_interview_points(selected_case, evidence_summary)
    recruiter_summary = build_recruiter_summary(selected_case, novelty_priority, evidence_summary)
    cv_value_pack = build_cv_value_pack(selected_case, supervisor_priority, novelty_priority, profile, evidence_summary)

    st.markdown(
        """
        This page turns the project into application-ready evidence. It helps you describe the repo as
        proof of research thinking, technical execution, and communication strength on your CV, LinkedIn,
        portfolio, and in interviews.
        """
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("Evidence rows", str(evidence_summary["runs"] or "Planned"), "Structured benchmark workflow")
    with c2:
        metric("OOD settings", str(evidence_summary["splits"] or len(OOD)), "Transfer-focused evaluation")
    with c3:
        metric("Repo assets", "6", "Code + docs + exports")
    with c4:
        best_score = f"{evidence_summary['best_score']:.1f}" if evidence_summary["best_score"] is not None else "Planned"
        metric("Best score", best_score, "Synthetic pilot, not claimed publication result")

    tab1, tab2, tab3, tab4 = st.tabs(["Recruiter Summary", "CV Bullets", "Repo Proof", "Interview Proof"])
    with tab1:
        st.info(recruiter_summary)
        st.dataframe(CV_SIGNAL_MAP, use_container_width=True, hide_index=True)
        st.markdown("### Suggested title variants")
        st.code(
            "\n".join(
                [
                    "Physics-Aware Agricultural Foundation Models for Robust Multimodal Learning",
                    "Multimodal Agricultural Foundation Models with OOD Benchmarking",
                    "Research Portfolio: G-E-M-Aligned Foundation Models for Agricultural AI",
                ]
            ),
            language="text",
        )
    with tab2:
        st.text_area(
            "Copy-ready CV bullets",
            value="\n".join(f"- {bullet}" for bullet in cv_bullets),
            height=220,
        )
        st.text_area(
            "One-line portfolio summary",
            value="Built a research-oriented portfolio on agricultural foundation models that combines multimodal SSL, OOD benchmarking, and physics-aware adaptation for WUR-targeted PhD applications.",
            height=90,
        )
        st.download_button(
            "Download CV Value Pack",
            data=cv_value_pack,
            file_name="wur_cv_value_pack.txt",
            mime="text/plain",
        )
    with tab3:
        st.dataframe(build_repo_assets(), use_container_width=True, hide_index=True)
        st.markdown(
            """
            Best framing:
            this repository demonstrates how you think, document, structure experiments, and communicate research.
            That combination is usually more valuable on a CV than a single flashy notebook.
            """
        )
    with tab4:
        st.markdown("### Talking points for supervisors or interviewers")
        for point in interview_points:
            st.markdown(f"- {point}")
        pilot_findings = build_pilot_findings(experiment_logs)
        if pilot_findings:
            st.markdown("### Evidence-aware talking points")
            for finding in pilot_findings:
                st.markdown(f"- {finding}")
        st.markdown("### Honest framing to keep")
        st.markdown(
            """
            - Treat current benchmark numbers as reproducible synthetic pilot evidence unless replaced by your own real experiments.
            - Emphasize the repo's strengths in problem framing, evaluation design, and implementation structure.
            - Present the project as a serious research portfolio that is ready for deeper empirical work.
            """
        )

elif page == "Supervisor Alignment":
    st.header("Supervisor Alignment")
    cols = st.columns(3)
    for col, item in zip(cols, SUPERVISORS):
        with col:
            card(item["name"], f"<strong>Focus:</strong> {item['focus']}<br><br><strong>Project role:</strong> {item['fit']}")
    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown(
            """
            ### Combined supervision story
            - **Athanasiadis layer:** responsible agricultural AI, food security, benchmark realism, and knowledge-guided learning.
            - **Torres layer:** multimodal representation learning, retrieval, visual analytics, and scalable data-centric eScience.
            - **Kapoor layer:** scientific priors, physics-aware reliability, and trustworthy adaptation.

            This gives you a clear answer if they ask why this project needs this exact supervisory team.
            """
        )
        st.dataframe(SUPERVISOR_SYNERGY, use_container_width=True, hide_index=True)
        st.markdown("### Best short oral pitch")
        st.markdown(
            """
            I want to investigate agriculture-specific foundation models that align biological,
            environmental, and management signals through multimodal self-supervision, make them explorable
            through retrieval and visual analytics, and then improve their robustness under real-world shift
            using scientifically grounded priors.
            """
        )
    with right:
        st.plotly_chart(radar_fig(), use_container_width=True)

elif page == "Proposal Builder":
    st.header("Proposal Builder")
    st.markdown(
        """
        This page builds a customizable mini abstract for your application, depending on the use case and
        research emphasis you want to foreground.
        """
    )
    pitch = build_pitch(selected_case, supervisor_priority, novelty_priority)
    fit_statement = build_fit_statement(selected_case, profile)
    st.text_area("Generated project abstract", value=pitch, height=220)
    st.text_area("Generated candidate-fit paragraph", value=fit_statement, height=170)
    st.download_button(
        "Download Abstract",
        data=pitch + "\n\n" + fit_statement,
        file_name="wur_project_pitch.txt",
        mime="text/plain",
    )
    st.markdown("### Suggested one-line title")
    st.code(
        f"{selected_case}: Agriculture-Specific Foundation Models with {novelty_priority.title()} Emphasis",
        language="text",
    )

elif page == "One-Page Proposal":
    st.header("One-Page Proposal")
    st.markdown(
        """
        This page generates a concise one-page proposal draft that you can adapt for your application,
        portfolio, or interview preparation.
        """
    )
    proposal = build_one_page_proposal(selected_case, supervisor_priority, novelty_priority, profile)
    st.text_area("Generated one-page proposal", value=proposal, height=520)
    st.download_button(
        "Download One-Page Proposal",
        data=proposal,
        file_name="wur_one_page_proposal.txt",
        mime="text/plain",
    )

elif page == "Motivation Letter":
    st.header("Motivation Letter")
    st.markdown(
        """
        This page generates a WUR-specific motivation letter draft based on the current research emphasis
        and flagship use case selected in the sidebar.
        """
    )
    letter = build_motivation_letter(selected_case, supervisor_priority, novelty_priority, profile)
    st.text_area("Generated motivation letter", value=letter, height=560)
    st.download_button(
        "Download Motivation Letter",
        data=letter,
        file_name="wur_motivation_letter.txt",
        mime="text/plain",
    )

elif page == "Application Pitch":
    st.header("Application Pitch")
    tabs = st.tabs(["CV Version", "Motivation Letter", "Interview Version"])
    with tabs[0]:
        st.markdown("### Project title")
        st.markdown("**Physics-Aware Agricultural Foundation Models for Robust Multimodal Learning**")
        st.markdown("### CV bullet")
        cv_summary = summarize_experiment_evidence(load_experiment_logs(uploaded_log_file)[0])
        st.markdown(build_cv_bullets(selected_case, novelty_priority, cv_summary)[0])
        st.markdown("### Athanasiadis-focused line")
        st.markdown(
            """
            Developed a benchmark-oriented agricultural AI portfolio that combines multimodal foundation models,
            knowledge-guided learning, and responsible evaluation under climate and geographic shift.
            """
        )
        st.markdown("### Three-professor line")
        st.markdown(build_three_professor_cv_line(selected_case))
    with tabs[1]:
        st.markdown(
            """
            My research interests lie at the intersection of multimodal representation learning,
            retrieval-oriented data science, scientific machine learning, and robust agricultural AI. To explore this direction,
            I developed a project concept around agriculture-specific foundation models that
            jointly learn from biological, environmental, and management signals using
            self-supervised objectives. Inspired by the AgriscienceFM vision, I am particularly
            interested in how such models can be evaluated under geographic and temporal shift,
            how multimodal retrieval and visual analytics can improve scientific exploration,
            and how physics-aware priors can improve trustworthiness for applications such as
            crop monitoring, yield-related prediction, crop-failure detection, and agricultural benchmark analysis.
            """
        )
    with tabs[2]:
        st.markdown(
            """
            I see this PhD as a rare opportunity because it combines three things I want to work on together:
            domain-specific foundation models, multimodal agricultural data science, and scientific constraints for robustness.
            What excites me most is not only building better models, but understanding how to make them transfer across
            regions, seasons, and sensing conditions in a way that remains meaningful for agricultural science,
            while also making the resulting representations explorable through retrieval and visual analysis.
            """
        )
    st.warning("Keep charts framed as targets, benchmarks, or planned experiments unless you have actually run and documented them.")

else:
    st.header("References")
    st.markdown("These links informed the professor-aware framing of the app.")
    for label, url in REFERENCES:
        st.markdown(f"- [{label}]({url})")
