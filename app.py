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
]

SUPERVISORS = [
    {
        "name": "Prof. Ioannis Athanasiadis",
        "focus": "Agricultural AI, food security, knowledge-guided ML",
        "fit": "Frames the project around meaningful agri benchmarks, realistic deployment, and knowledge-guided learning.",
        "line": "I want to build agricultural AI that remains useful under real-world variation and supports food-security-relevant tasks.",
    },
    {
        "name": "Prof. Ricardo da Silva Torres",
        "focus": "Self-supervised learning, computer vision, multimodal remote sensing",
        "fit": "Anchors the representation-learning side through contrastive and multimodal pretraining.",
        "line": "I want to study how agriculture-specific multimodal representations can be learned from satellite imagery and related signals.",
    },
    {
        "name": "Dr. Taniya Kapoor",
        "focus": "Physics-informed ML, scientific machine learning, foundation models for science",
        "fit": "Drives the scientific-ML layer through constraints, priors, and physically grounded adaptation.",
        "line": "I am interested in combining representation learning with scientific priors so models become more trustworthy under shift.",
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
        ["Curious", "Multiple use cases, ablation questions, and supervisor-aware novelty framing"],
        ["Dynamic team player", "G-E-M alignment and collaborative benchmark construction across modalities"],
        ["Applied ML background", "Remote sensing, multimodal SSL, OOD evaluation, and scientific ML positioning"],
        ["Scientific writing", "Built-in CV, motivation-letter, abstract, and interview phrasing"],
    ],
    columns=["Quality from vacancy", "How the project demonstrates it"],
)

DUTY_ALIGNMENT = pd.DataFrame(
    [
        ["State-of-the-art self-supervision", "Method Blueprint and literature-grounded SSL objectives"],
        ["Design self-supervised architectures", "Multimodal G-E-M pipeline and fusion-oriented model sketch"],
        ["Large-scale training on HPC", "PyTorch Lightning and reproducible training workflow emphasis"],
        ["Write papers and present results", "Paper-style hypotheses, ablations, and reusable research abstracts"],
        ["Collaborate on data and benchmarks", "Dataset Strategy page and OOD benchmark suite framing"],
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
    ],
    columns=["Capability", "How this project signals it"],
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
}

DATASETS = pd.DataFrame(
    [
        ["Sentinel-2", "Multispectral satellite imagery", "Crop type, field boundaries, vegetation dynamics", "Torres"],
        ["ERA5 or weather station data", "Temperature, rainfall, radiation, humidity", "Temporal context and environment module", "Athanasiadis + Kapoor"],
        ["Field boundaries", "Parcel polygons and geospatial joins", "Management context and spatial grounding", "Athanasiadis"],
        ["Agronomic metadata", "Region, season, management notes, task labels", "Cross-modal alignment and retrieval", "Torres + Athanasiadis"],
        ["Crop-growth priors", "GDD, radiation-use signals, seasonal plausibility", "Physics-aware regularization", "Kapoor"],
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


def style() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
            .stApp {
                background:
                    radial-gradient(circle at 10% 10%, rgba(181,107,63,0.18), transparent 24%),
                    radial-gradient(circle at 88% 12%, rgba(46,106,70,0.16), transparent 22%),
                    linear-gradient(180deg, #f8f4ec 0%, #f4efe6 100%);
            }
            html, body, [class*="css"] { font-family: "IBM Plex Sans", sans-serif; }
            h1, h2, h3, h4 { font-family: "Space Grotesk", sans-serif; color: #18231f; }
            [data-testid="stSidebar"] { background: rgba(248,244,236,0.97); }
            .hero, .card, .metric {
                background: rgba(255,252,247,0.84);
                border: 1px solid rgba(24,35,31,0.10);
                border-radius: 24px;
                box-shadow: 0 12px 32px rgba(24,35,31,0.05);
            }
            .hero { padding: 2rem 2.2rem; }
            .card { padding: 1rem; height: 100%; }
            .metric { padding: 1rem; }
            .kicker { color: #2e6a46; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.88rem; font-weight: 700; }
            .hero-title { font-size: 3rem; line-height: 1.0; margin: 0.35rem 0 1rem 0; }
            .muted { color: #5b685f; }
            .metric-label { color: #5b685f; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.06em; }
            .metric-value { font-family: "Space Grotesk", sans-serif; font-size: 1.95rem; margin: 0.2rem 0; }
            .metric-note { color: #2e6a46; font-weight: 600; }
            .poster-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            .poster-panel {
                background: linear-gradient(160deg, rgba(255,252,247,0.95), rgba(239,226,216,0.70));
                border: 1px solid rgba(24,35,31,0.10);
                border-radius: 26px;
                padding: 1rem 1.1rem;
                min-height: 190px;
                box-shadow: 0 18px 36px rgba(24,35,31,0.06);
            }
            .poster-panel h4 {
                margin-bottom: 0.55rem;
            }
            .poster-tag {
                display: inline-block;
                padding: 0.2rem 0.55rem;
                border-radius: 999px;
                background: rgba(46,106,70,0.12);
                color: #2e6a46;
                font-size: 0.78rem;
                font-weight: 700;
                margin-bottom: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
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
    fig.add_bar(name="Typical baseline", x=x, y=base, marker_color="#b56b3f")
    fig.add_bar(name="Project target", x=x, y=target, marker_color="#2e6a46")
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.62)",
        margin=dict(l=20, r=20, t=30, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
        yaxis_title="Illustrative OOD score",
    )
    fig.update_yaxes(gridcolor="rgba(24,35,31,0.08)")
    return fig


def radar_fig() -> go.Figure:
    axes = ["Agri impact", "SSL", "Multimodal RS", "Scientific ML", "Benchmarks"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[5, 3, 3, 4, 5], theta=axes, fill="toself", name="Athanasiadis", line_color="#2e6a46"))
    fig.add_trace(go.Scatterpolar(r=[3, 5, 5, 2, 4], theta=axes, fill="toself", name="Torres", line_color="#b56b3f"))
    fig.add_trace(go.Scatterpolar(r=[3, 3, 2, 5, 3], theta=axes, fill="toself", name="Kapoor", line_color="#607b68"))
    fig.add_trace(go.Scatterpolar(r=[5, 5, 5, 5, 5], theta=axes, fill="toself", name="Project fit", line_color="#18231f"))
    fig.update_layout(
        polar=dict(bgcolor="rgba(255,255,255,0.55)", radialaxis=dict(visible=True, range=[0, 5])),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=25, b=20),
        legend=dict(orientation="h", y=1.13, x=0),
    )
    return fig


def sankey_fig() -> go.Figure:
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=18,
            label=["G inputs", "E inputs", "M inputs", "Vision SSL", "Temporal SSL", "Geo fusion", "Cross-modal alignment", "Physics layer", "OOD suite", "Crop tasks"],
            color=["#b56b3f", "#b56b3f", "#b56b3f", "#2e6a46", "#2e6a46", "#2e6a46", "#607b68", "#18231f", "#607b68", "#2e6a46"],
        ),
        link=dict(
            source=[0, 1, 2, 3, 4, 5, 6, 7, 7],
            target=[3, 4, 5, 6, 6, 6, 7, 8, 9],
            value=[6, 6, 4, 7, 6, 4, 9, 6, 7],
        ),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=20, b=10))
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
            colorscale=[[0, "#efe2d8"], [0.5, "#9bbcaa"], [1, "#2e6a46"]],
            showscale=False,
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.62)",
        margin=dict(l=20, r=20, t=20, b=10),
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
        "baseline": "#b56b3f",
        "multimodal_ssl": "#607b68",
        "physics_aware_fm": "#2e6a46",
    }
    for variant in summary["model_variant"].unique():
        subset = summary[summary["model_variant"] == variant]
        fig.add_bar(
            name=variant,
            x=subset["split"],
            y=subset["score"],
            marker_color=palette.get(variant, "#18231f"),
        )
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.62)",
        margin=dict(l=20, r=20, t=25, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
        yaxis_title="Observed score from experiment logs",
    )
    fig.update_yaxes(gridcolor="rgba(24,35,31,0.08)")
    return fig


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
        "Athanasiadis": "agricultural relevance, food security, and benchmark realism",
        "Torres": "multimodal representation learning for remote sensing",
        "Kapoor": "scientific priors and trustworthy adaptation",
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
        "Athanasiadis": "agricultural impact, benchmark realism, and knowledge-guided learning",
        "Torres": "multimodal self-supervised representation learning for remote sensing",
        "Kapoor": "physics-informed scientific machine learning for robust adaptation",
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
            "Method Blueprint",
            "Poster Architecture",
            "Experimental Plan",
            "Dataset Strategy",
            "Candidate Fit",
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
                remote sensing, and physics-aware scientific machine learning for robust agricultural prediction
                under geographic, temporal, and sensor shift.
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
    cols = st.columns(3)
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
            to learn transferable agricultural representations, then improve reliability with
            physics-aware adaptation and explicit out-of-distribution evaluation.
            """
        )
        st.markdown("### Why this pitch is stronger")
        st.markdown(
            """
            - It mirrors the vacancy closely without sounding copied.
            - It combines the three supervisors in one coherent research arc.
            - It shows a realistic benchmark plan instead of unsupported claims.
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
            - Region-aware negatives or semantically aware positives for remote sensing
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

joint_repr = cross_modal_fuser(image_tokens, weather_tokens, management_tokens)
ssl_loss = contrastive_loss(joint_repr) + masked_modeling_loss(joint_repr)
physics_loss = crop_dynamics_regularizer(joint_repr, gdd, radiation, season_index)

loss = ssl_loss + 0.2 * physics_loss
""",
            language="python",
        )
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
    tab1, tab2, tab3, tab4 = st.tabs(["OOD Benchmarks", "Ablations", "ML Stack", "Hypotheses"])
    with tab1:
        if log_source and has_evidence:
            st.success(f"Loaded experiment evidence from {log_source}.")
            st.plotly_chart(evidence_fig(experiment_logs), use_container_width=True)
            st.dataframe(experiment_logs, use_container_width=True, hide_index=True)
        else:
            st.info("No experiment log detected yet. The app falls back to the planned benchmark structure below. Add `data/experiment_runs.csv` or upload a CSV from the sidebar.")
            st.dataframe(OOD, use_container_width=True, hide_index=True)
            st.plotly_chart(heatmap_fig(), use_container_width=True)
        st.markdown("**Core message:** optimize for transfer and robustness, not only in-domain accuracy.")
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
        st.markdown(
            """
            Best message to send:
            this is not just a topic-aligned project, it is also a capability-aligned project that
            demonstrates Python, PyTorch, scikit-learn, remote sensing, time-series handling,
            experiment design, and scientific writing readiness.
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
            - **Athanasiadis layer:** agricultural relevance, food security, benchmark realism.
            - **Torres layer:** self-supervised representation learning for multimodal remote sensing.
            - **Kapoor layer:** scientific priors and physics-aware reliability.

            This gives you a clear answer if they ask why this project needs this exact supervisory team.
            """
        )
        st.markdown("### Best short oral pitch")
        st.markdown(
            """
            I want to investigate agriculture-specific foundation models that align biological,
            environmental, and management signals through multimodal self-supervision, then make
            them more robust under real-world shift using scientifically grounded priors.
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
        st.markdown(
            """
            Designed a proposal-driven research project on agriculture-specific foundation models,
            combining multimodal self-supervised learning, G-E-M representation alignment, and
            physics-aware adaptation for robust crop-focused remote sensing and time-series tasks.
            """
        )
    with tabs[1]:
        st.markdown(
            """
            My research interests lie at the intersection of multimodal representation learning,
            scientific machine learning, and robust agricultural AI. To explore this direction,
            I developed a project concept around agriculture-specific foundation models that
            jointly learn from biological, environmental, and management signals using
            self-supervised objectives. Inspired by the AgriscienceFM vision, I am particularly
            interested in how such models can be evaluated under geographic and temporal shift,
            and how physics-aware priors can improve trustworthiness for applications such as
            crop monitoring, yield-related prediction, and crop-failure detection.
            """
        )
    with tabs[2]:
        st.markdown(
            """
            I see this PhD as a rare opportunity because it combines three things I want to work on together:
            domain-specific foundation models, multimodal agricultural data, and scientific constraints for robustness.
            What excites me most is not only building better models, but understanding how to make them transfer across
            regions, seasons, and sensing conditions in a way that remains meaningful for agricultural science.
            """
        )
    st.warning("Keep charts framed as targets, benchmarks, or planned experiments unless you have actually run and documented them.")

else:
    st.header("References")
    st.markdown("These links informed the professor-aware framing of the app.")
    for label, url in REFERENCES:
        st.markdown(f"- [{label}]({url})")
