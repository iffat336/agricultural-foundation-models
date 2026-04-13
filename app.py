from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="AgriFusion",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_TITLE = "AgriFusion"
APP_SUBTITLE = "Image + Weather Learning for Crop Stress and Yield Risk"
PROJECT_TAGLINE = (
    "A foundation-model-inspired multimodal agricultural prototype for studying whether "
    "image, weather, and field metadata improve generalization under agricultural shift."
)
PROJECT_NOTE = (
    "The benchmark charts come from a reproducible synthetic pilot. The live prediction page is an "
    "interactive prototype scorer designed to demonstrate multimodal reasoning and failure analysis, "
    "not a deployed agronomic model."
)
RUNS_PATH = Path("data/experiment_runs.csv")

PAGES = [
    "Home",
    "Data Explorer",
    "Model Lab",
    "Prediction Demo",
    "Explainability",
    "Generalization Test",
    "Research Reflection",
]

VACANCY_FIT = [
    "Multimodal agricultural data: image, weather/time-series, and field metadata are all part of the app story.",
    "Weak generalization in agriculture: the benchmark focuses on region, year, climate, sensor, and missing-modality shift.",
    "Domain-specific representation learning: the core comparison is image-only versus multimodal agricultural modeling.",
    "Scientific grounding: the physics-aware branch adds a simple consistency target instead of relying only on correlation fitting.",
    "One downstream task: the app stays focused on crop stress and yield-risk prediction rather than spreading across many unrelated demos.",
]

SUPERVISOR_ANGLES = pd.DataFrame(
    [
        ["Ioannis Athanasiadis", "Food security and agricultural decision support", "Frame the app as crop stress and yield-risk intelligence under realistic agricultural shift."],
        ["Ricardo da Silva Torres", "Multimodal learning, machine vision, and data science", "Frame the app as image plus weather representation learning for robust agricultural understanding."],
        ["Taniya Kapoor", "Physics-informed and scientifically grounded ML", "Frame the app as multimodal learning with simple scientific structure and explicit failure analysis."],
    ],
    columns=["Professor", "What to emphasize", "Best framing angle"],
)

INPUT_MODALITIES = pd.DataFrame(
    [
        ["Field image", "Canopy, leaf, or parcel appearance", "Visual crop condition and texture cues"],
        ["Weather sequence", "Temperature, rainfall, radiation, humidity", "Temporal stress context that images alone miss"],
        ["Field metadata", "Soil quality, growth stage, management, region", "Agronomic context needed for transfer and calibration"],
    ],
    columns=["Input", "Examples", "Why it matters"],
)

BENCHMARK_SPLITS = pd.DataFrame(
    [
        ["Region transfer", "New geography at test time", "Tests whether the representation travels across locations"],
        ["Year transfer", "New season or year", "Measures robustness to seasonal variation"],
        ["Climate-stress transfer", "Harsher weather stress", "Matches food-security-relevant failure conditions"],
        ["Sensor transfer", "Different image characteristics", "Checks whether the vision branch is brittle to observation shift"],
        ["Missing modality", "Dropped weather or metadata", "Shows whether the multimodal model depends too strongly on complete inputs"],
    ],
    columns=["Split", "What changes", "Why it matters"],
)

SCENARIOS = {
    "Healthy irrigated wheat": {
        "region": "Delta",
        "growth_stage": "Vegetative",
        "soil_quality": 0.78,
        "management_quality": 0.81,
        "ndvi_trend": 0.72,
        "temp_anomaly": 0.18,
        "rainfall_deficit": 0.22,
        "humidity_index": 0.59,
        "sensor_noise": 0.10,
    },
    "Heat-stressed maize field": {
        "region": "Semi-arid South",
        "growth_stage": "Flowering",
        "soil_quality": 0.48,
        "management_quality": 0.51,
        "ndvi_trend": 0.39,
        "temp_anomaly": 0.73,
        "rainfall_deficit": 0.68,
        "humidity_index": 0.36,
        "sensor_noise": 0.14,
    },
    "Late-season rice parcel": {
        "region": "Coastal East",
        "growth_stage": "Grain fill",
        "soil_quality": 0.64,
        "management_quality": 0.58,
        "ndvi_trend": 0.47,
        "temp_anomaly": 0.41,
        "rainfall_deficit": 0.31,
        "humidity_index": 0.72,
        "sensor_noise": 0.19,
    },
}

REGIONS = ["Delta", "Semi-arid South", "Coastal East", "River Plains"]
GROWTH_STAGES = ["Vegetative", "Flowering", "Grain fill", "Ripening"]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #050607;
            --bg-panel: #0f141b;
            --bg-soft: #141c26;
            --border: rgba(122, 166, 255, 0.24);
            --text-main: #f4f7fb;
            --text-soft: #b4c0cf;
            --accent: #84abff;
            --accent-2: #59e2bf;
            --accent-3: #f8c26a;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(89, 226, 191, 0.10), transparent 30%),
                radial-gradient(circle at top right, rgba(132, 171, 255, 0.16), transparent 28%),
                linear-gradient(180deg, #030405 0%, #07090d 100%);
            color: var(--text-main);
        }

        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0d12 0%, #06080b 100%);
            border-right: 1px solid var(--border);
        }

        h1, h2, h3, h4 {
            color: var(--text-main);
            letter-spacing: -0.02em;
        }

        .hero-card, .panel-card, .metric-card {
            background: linear-gradient(180deg, rgba(15, 20, 27, 0.98), rgba(11, 15, 21, 0.98));
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
        }

        .hero-card {
            padding: 1.5rem 1.6rem;
            margin-bottom: 1rem;
        }

        .panel-card, .metric-card {
            padding: 1.1rem 1.2rem;
        }

        .eyebrow {
            color: var(--accent-2);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .hero-title {
            font-size: 2.45rem;
            font-weight: 700;
            line-height: 1.03;
            margin: 0.35rem 0 0.4rem 0;
        }

        .hero-subtitle {
            color: var(--accent);
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.85rem;
        }

        .body-copy {
            color: var(--text-soft);
            font-size: 1rem;
            line-height: 1.65;
            margin: 0;
        }

        .section-label {
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .metric-title {
            color: var(--text-soft);
            font-size: 0.88rem;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            color: var(--text-main);
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1;
        }

        .metric-foot {
            color: var(--text-soft);
            font-size: 0.83rem;
            margin-top: 0.45rem;
        }

        .callout {
            border-left: 3px solid var(--accent-2);
            padding-left: 0.9rem;
            color: var(--text-soft);
            margin-top: 0.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_runs() -> pd.DataFrame:
    if not RUNS_PATH.exists():
        return pd.DataFrame(
            columns=["run_name", "use_case", "split", "model_variant", "score", "f1", "ece", "notes"]
        )

    frame = pd.read_csv(RUNS_PATH)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["f1"] = pd.to_numeric(frame["f1"], errors="coerce")
    frame["ece"] = pd.to_numeric(frame["ece"], errors="coerce")
    frame["model_label"] = frame["model_variant"].map(
        {
            "baseline": "Image-only baseline",
            "multimodal_ssl": "Multimodal SSL fusion",
            "physics_aware_fm": "Physics-aware fusion",
        }
    ).fillna(frame["model_variant"])
    return frame


@st.cache_data(show_spinner=False)
def build_demo_dataset() -> pd.DataFrame:
    rows = []
    for name, values in SCENARIOS.items():
        for season_idx, season_name in enumerate(["Early season", "Mid season", "Late season"]):
            rows.append(
                {
                    "scenario": name,
                    "season": season_name,
                    "region": values["region"],
                    "growth_stage": values["growth_stage"],
                    "soil_quality": round(values["soil_quality"] - 0.04 + 0.03 * season_idx, 2),
                    "management_quality": round(values["management_quality"] - 0.03 + 0.02 * season_idx, 2),
                    "ndvi_trend": round(values["ndvi_trend"] - 0.10 + 0.07 * season_idx, 2),
                    "temp_anomaly": round(values["temp_anomaly"] + 0.06 * season_idx, 2),
                    "rainfall_deficit": round(values["rainfall_deficit"] + 0.05 * season_idx, 2),
                    "humidity_index": round(values["humidity_index"] - 0.03 * season_idx, 2),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_class_distribution() -> pd.DataFrame:
    dataset = build_demo_dataset().copy()
    stress_signal = (
        0.46 * dataset["temp_anomaly"]
        + 0.36 * dataset["rainfall_deficit"]
        + 0.18 * (1 - dataset["humidity_index"])
    )
    yield_signal = (
        0.42 * stress_signal
        + 0.20 * (1 - dataset["soil_quality"])
        + 0.18 * (1 - dataset["management_quality"])
    )
    dataset["stress_class"] = pd.cut(
        stress_signal,
        bins=[-1, 0.38, 0.62, 2],
        labels=["Low", "Moderate", "High"],
    )
    dataset["yield_risk"] = pd.cut(
        yield_signal,
        bins=[-1, 0.34, 0.56, 2],
        labels=["Low", "Moderate", "High"],
    )
    return dataset


def chart_layout() -> dict:
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#f4f7fb"},
        "margin": {"l": 20, "r": 20, "t": 40, "b": 20},
    }


def class_distribution_chart(dataset: pd.DataFrame) -> go.Figure:
    melted = pd.concat(
        [
            dataset["stress_class"].value_counts().rename_axis("Class").reset_index(name="Count").assign(Target="Stress class"),
            dataset["yield_risk"].value_counts().rename_axis("Class").reset_index(name="Count").assign(Target="Yield-risk class"),
        ],
        ignore_index=True,
    )
    fig = px.bar(
        melted,
        x="Class",
        y="Count",
        color="Target",
        barmode="group",
        color_discrete_map={"Stress class": "#84abff", "Yield-risk class": "#59e2bf"},
    )
    fig.update_layout(**chart_layout(), height=300, xaxis_title="", yaxis_title="Synthetic rows")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_xaxes(showgrid=False)
    return fig


def metric_card(title: str, value: str, foot: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


def generate_canopy_image(stress: float, vigor: float, humidity: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = 180
    y, x = np.mgrid[0:size, 0:size]
    center = size / 2
    radius = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    canopy = np.clip(1.0 - radius / (size * 0.78), 0, 1)

    green = np.clip(0.25 + 0.65 * vigor - 0.35 * stress, 0, 1)
    red = np.clip(0.18 + 0.42 * stress + 0.10 * (1 - humidity), 0, 1)
    blue = np.clip(0.10 + 0.25 * humidity, 0, 1)

    image = np.zeros((size, size, 3), dtype=np.float32)
    image[..., 0] = red * canopy + 0.10 * rng.random((size, size))
    image[..., 1] = green * canopy + 0.08 * rng.random((size, size))
    image[..., 2] = blue * canopy + 0.05 * rng.random((size, size))
    stripes = ((x // 14) % 2) * 0.05
    image[..., 1] += stripes * canopy
    image[..., 0] += (1 - canopy) * 0.08
    return np.clip(image, 0, 1)


def create_weather_series(temp_anomaly: float, rainfall_deficit: float, humidity_index: float) -> pd.DataFrame:
    steps = np.arange(1, 13)
    temperature = 21 + 9 * temp_anomaly + 4 * np.sin((steps / 12) * np.pi)
    rainfall = 80 * (1 - rainfall_deficit) + 15 * np.cos((steps / 12) * np.pi * 1.3)
    humidity = 42 + 38 * humidity_index + 6 * np.sin((steps / 12) * np.pi * 1.5)
    return pd.DataFrame(
        {
            "Month": steps,
            "Temperature": np.round(temperature, 1),
            "Rainfall": np.round(rainfall, 1),
            "Humidity": np.round(humidity, 1),
        }
    )


def model_scores_for_inputs(
    image_signal: float,
    ndvi_trend: float,
    temp_anomaly: float,
    rainfall_deficit: float,
    humidity_index: float,
    soil_quality: float,
    management_quality: float,
    sensor_noise: float,
    missing_weather: bool,
) -> dict[str, float]:
    climate_pressure = 0.44 * temp_anomaly + 0.42 * rainfall_deficit + 0.16 * sensor_noise
    resilience = 0.28 * soil_quality + 0.22 * management_quality + 0.18 * ndvi_trend + 0.12 * humidity_index
    missing_penalty = 0.22 if missing_weather else 0.0

    image_only_stress = sigmoid(1.75 - 2.4 * image_signal + 0.75 * sensor_noise)
    tabular_only_stress = sigmoid(2.3 * climate_pressure - 1.7 * resilience + missing_penalty)
    multimodal_stress = sigmoid(1.9 * climate_pressure - 1.35 * image_signal - 1.25 * resilience + 0.55 * sensor_noise + missing_penalty)
    physics_stress = sigmoid(1.75 * climate_pressure - 1.20 * image_signal - 1.35 * resilience + 0.35 * sensor_noise + 0.65 * max(0.0, temp_anomaly - humidity_index) + missing_penalty)

    image_only_yield = sigmoid(0.95 * image_only_stress + 0.40 * sensor_noise - 0.30 * ndvi_trend)
    tabular_only_yield = sigmoid(1.05 * tabular_only_stress + 0.22 * rainfall_deficit - 0.30 * management_quality)
    multimodal_yield = sigmoid(0.98 * multimodal_stress + 0.16 * rainfall_deficit - 0.22 * management_quality)
    physics_yield = sigmoid(0.92 * physics_stress + 0.12 * rainfall_deficit - 0.26 * management_quality)

    return {
        "image_only_stress": float(image_only_stress),
        "tabular_only_stress": float(tabular_only_stress),
        "multimodal_stress": float(multimodal_stress),
        "physics_stress": float(physics_stress),
        "image_only_yield": float(image_only_yield),
        "tabular_only_yield": float(tabular_only_yield),
        "multimodal_yield": float(multimodal_yield),
        "physics_yield": float(physics_yield),
    }


def extract_image_signal(image_array: np.ndarray) -> tuple[float, np.ndarray]:
    rgb = image_array.astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    green = rgb[..., 1]
    red = rgb[..., 0]
    ndvi_proxy = np.clip((green - red + 1.0) / 2.0, 0, 1)
    image_signal = float(np.clip(ndvi_proxy.mean(), 0, 1))
    heatmap = np.clip(ndvi_proxy, 0, 1)
    return image_signal, heatmap


def build_benchmark_bar(frame: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.bar(
        frame,
        x="split",
        y=metric,
        color="model_label",
        barmode="group",
        text_auto=".3f",
        category_orders={
            "split": [
                "Region transfer",
                "Year transfer",
                "Climate-stress transfer",
                "Sensor transfer",
                "Missing modality",
            ]
        },
        color_discrete_map={
            "Image-only baseline": "#7f8ea3",
            "Multimodal SSL fusion": "#84abff",
            "Physics-aware fusion": "#59e2bf",
        },
    )
    fig.update_layout(**chart_layout(), height=420, xaxis_title="", yaxis_title=metric.upper())
    if metric == "ece":
        fig.update_yaxes(range=[0, max(0.7, frame[metric].max() + 0.05)], gridcolor="rgba(255,255,255,0.08)")
    else:
        fig.update_yaxes(range=[0, 1.08], gridcolor="rgba(255,255,255,0.08)")
    fig.update_xaxes(showgrid=False)
    return fig


def build_mean_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("model_label", as_index=False)
        .agg(score=("score", "mean"), f1=("f1", "mean"), ece=("ece", "mean"))
        .sort_values("score", ascending=False)
    )


def build_model_summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    summary = build_mean_summary(frame)
    description = {
        "Image-only baseline": "Vision branch only",
        "Multimodal SSL fusion": "Image + weather + metadata fusion",
        "Physics-aware fusion": "Multimodal fusion + scientific consistency target",
    }
    summary["setup"] = summary["model_label"].map(description)
    summary = summary.rename(
        columns={
            "model_label": "Model",
            "score": "Mean score",
            "f1": "Mean F1",
            "ece": "Mean calibration error",
            "setup": "Setup",
        }
    )
    return summary[["Model", "Setup", "Mean score", "Mean F1", "Mean calibration error"]]


def build_confusion_matrix(model_name: str) -> pd.DataFrame:
    matrices = {
        "Image-only baseline": np.array([[21, 4, 3], [5, 17, 6], [4, 7, 18]]),
        "Tabular-only proxy": np.array([[23, 3, 2], [4, 18, 5], [3, 6, 19]]),
        "Multimodal SSL fusion": np.array([[26, 1, 1], [1, 25, 2], [1, 2, 25]]),
        "Physics-aware fusion": np.array([[25, 2, 1], [1, 25, 2], [1, 2, 25]]),
    }
    matrix = matrices[model_name]
    return pd.DataFrame(
        matrix,
        index=["True low", "True moderate", "True high"],
        columns=["Pred low", "Pred moderate", "Pred high"],
    )


def build_error_profile(runs: pd.DataFrame) -> pd.DataFrame:
    frame = runs.copy()
    frame["error_rate"] = 1 - frame["score"]
    return frame[["split", "model_label", "error_rate"]]


def scenario_overview(scenario_name: str) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    scenario = SCENARIOS[scenario_name]
    image = generate_canopy_image(
        stress=max(scenario["temp_anomaly"], scenario["rainfall_deficit"]),
        vigor=scenario["ndvi_trend"],
        humidity=scenario["humidity_index"],
        seed=abs(hash(scenario_name)) % 10000,
    )
    weather = create_weather_series(
        temp_anomaly=scenario["temp_anomaly"],
        rainfall_deficit=scenario["rainfall_deficit"],
        humidity_index=scenario["humidity_index"],
    )
    metadata = pd.DataFrame(
        [
            ["Region", scenario["region"]],
            ["Growth stage", scenario["growth_stage"]],
            ["Soil quality", scenario["soil_quality"]],
            ["Management quality", scenario["management_quality"]],
            ["NDVI trend", scenario["ndvi_trend"]],
            ["Temperature anomaly", scenario["temp_anomaly"]],
            ["Rainfall deficit", scenario["rainfall_deficit"]],
            ["Sensor noise", scenario["sensor_noise"]],
        ],
        columns=["Field variable", "Value"],
    )
    return image, weather, metadata


def render_home(runs: pd.DataFrame) -> None:
    mean_summary = build_mean_summary(runs)
    best_model = str(mean_summary.iloc[0]["model_label"])
    hardest_split = runs.groupby("split", as_index=False)["score"].max().sort_values("score").iloc[0]

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="eyebrow">Multimodal Agricultural ML Prototype</div>
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-subtitle">{APP_SUBTITLE}</div>
            <p class="body-copy">{PROJECT_TAGLINE}</p>
            <p class="callout"><strong>Research question:</strong> Given field imagery and field conditions, can a
            multimodal agricultural model estimate crop stress and yield risk more robustly than a single-modality baseline?
            <br><br><strong>Current synthetic pilot signal:</strong> {best_model} is the strongest average performer, while
            <strong>{hardest_split['split']}</strong> remains the toughest evaluation condition.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(PROJECT_NOTE)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        metric_card("Main use case", "Crop stress + yield risk", "One focused downstream task instead of several disconnected demos.")
    with c2:
        metric_card("Core comparison", "Single vs multimodal", "Image-only, tabular-only live proxy, and fused multimodal reasoning.")
    with c3:
        metric_card("Pain point", "Generalization", "Built around shift rather than only in-distribution accuracy.")

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown('<div class="section-label">Why this project exists</div>', unsafe_allow_html=True)
        st.subheader("Agriculture is exactly where generic ML shortcuts break")
        st.write(
            "Agricultural data is fragmented, local, seasonal, and heterogeneous. An image of one field rarely "
            "tells the full story without weather context, crop stage, management conditions, and other field metadata. "
            "That is why this app is framed as a foundation-model-inspired prototype rather than just a classifier."
        )
        for item in VACANCY_FIT:
            st.markdown(f"- {item}")
    with right:
        st.markdown('<div class="section-label">Modalities</div>', unsafe_allow_html=True)
        st.dataframe(INPUT_MODALITIES, use_container_width=True, hide_index=True)


def render_data_explorer() -> None:
    st.markdown('<div class="section-label">Data Explorer</div>', unsafe_allow_html=True)
    st.subheader("Synthetic but domain-inspired agricultural inputs")

    selected_scenario = st.selectbox("Scenario", list(SCENARIOS.keys()), index=1)
    image, weather, metadata = scenario_overview(selected_scenario)

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.image(image, caption=f"Synthetic canopy sample: {selected_scenario}", use_container_width=True)
    with right:
        weather_long = weather.melt("Month", var_name="Signal", value_name="Value")
        weather_fig = px.line(
            weather_long,
            x="Month",
            y="Value",
            color="Signal",
            markers=True,
            color_discrete_map={
                "Temperature": "#f8c26a",
                "Rainfall": "#84abff",
                "Humidity": "#59e2bf",
            },
        )
        weather_fig.update_layout(**chart_layout(), height=320, xaxis_title="", yaxis_title="Synthetic seasonal signal")
        weather_fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        weather_fig.update_xaxes(showgrid=False)
        st.plotly_chart(weather_fig, use_container_width=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("### Field metadata")
        st.dataframe(metadata, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("### Split logic and missing data")
        st.dataframe(BENCHMARK_SPLITS, use_container_width=True, hide_index=True)

    dataset = build_demo_dataset()
    labeled_dataset = build_class_distribution()
    lower_left, lower_right = st.columns([1, 1], gap="large")
    with lower_left:
        stage_counts = dataset["growth_stage"].value_counts().rename_axis("Growth stage").reset_index(name="Count")
        fig = px.bar(stage_counts, x="Growth stage", y="Count", color="Growth stage")
        fig.update_layout(**chart_layout(), height=300, showlegend=False, xaxis_title="", yaxis_title="Demo rows")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig, use_container_width=True)
    with lower_right:
        missingness = pd.DataFrame(
            {
                "Signal": ["Image", "Weather", "Field metadata", "Location", "Management"],
                "Missing rate": [0.00, 0.18, 0.08, 0.04, 0.11],
            }
        )
        miss_fig = px.bar(missingness, x="Signal", y="Missing rate", color="Signal")
        miss_fig.update_layout(**chart_layout(), height=300, showlegend=False, xaxis_title="", yaxis_title="Prototype missingness")
        miss_fig.update_yaxes(range=[0, 0.25], gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(miss_fig, use_container_width=True)

    st.markdown("### Class distribution")
    st.plotly_chart(class_distribution_chart(labeled_dataset), use_container_width=True)


def render_model_lab(runs: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">Model Lab</div>', unsafe_allow_html=True)
    st.subheader("Compare simpler baselines with multimodal fusion")

    st.dataframe(build_model_summary_table(runs).round(3), use_container_width=True, hide_index=True)
    selected_model = st.selectbox(
        "Inspect model family",
        ["Image-only baseline", "Tabular-only proxy", "Multimodal SSL fusion", "Physics-aware fusion"],
        index=2,
    )
    metric = st.selectbox("Benchmark metric", ["score", "f1", "ece"], index=0, key="lab_metric")
    st.plotly_chart(build_benchmark_bar(runs, metric), use_container_width=True)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("### Confusion matrix")
        st.caption("Illustrative class-level view aligned with the synthetic pilot story.")
        confusion = build_confusion_matrix(selected_model)
        confusion_fig = px.imshow(
            confusion,
            text_auto=True,
            color_continuous_scale=["#0f141b", "#84abff", "#59e2bf"],
            aspect="auto",
        )
        confusion_fig.update_layout(**chart_layout(), height=320, coloraxis_showscale=False)
        st.plotly_chart(confusion_fig, use_container_width=True)
    with right:
        st.markdown("### Error distribution across shifts")
        error_profile = build_error_profile(runs)
        error_fig = px.bar(
            error_profile,
            x="split",
            y="error_rate",
            color="model_label",
            barmode="group",
            color_discrete_map={
                "Image-only baseline": "#7f8ea3",
                "Multimodal SSL fusion": "#84abff",
                "Physics-aware fusion": "#59e2bf",
            },
        )
        error_fig.update_layout(**chart_layout(), height=320, xaxis_title="", yaxis_title="Error rate")
        error_fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        error_fig.update_xaxes(showgrid=False)
        st.plotly_chart(error_fig, use_container_width=True)

    st.markdown(
        """
        <div class="panel-card">
            <div class="section-label">Technical Framing</div>
            <h3 style="margin-top:0;">Foundation-model-inspired, not inflated</h3>
            <p class="body-copy">
            The app does not claim to be a full agricultural foundation model. It is a compact prototype that borrows
            the right research ingredients: multimodal fusion, representation learning, a stronger comparison against
            simple baselines, and explicit evaluation under realistic agricultural shift.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_demo() -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    st.markdown('<div class="section-label">Prediction Demo</div>', unsafe_allow_html=True)
    st.subheader("Interactive crop stress and yield-risk prototype")

    uploaded = st.file_uploader("Upload a field or canopy image", type=["png", "jpg", "jpeg"])
    use_uploaded = uploaded is not None

    default = SCENARIOS["Heat-stressed maize field"]
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        if use_uploaded:
            pil_image = Image.open(uploaded).convert("RGB")
            image_array = np.array(pil_image)
            st.image(image_array, caption="Uploaded field image", use_container_width=True)
        else:
            image_array = generate_canopy_image(
                stress=max(default["temp_anomaly"], default["rainfall_deficit"]),
                vigor=default["ndvi_trend"],
                humidity=default["humidity_index"],
                seed=17,
            )
            st.image(image_array, caption="Using built-in synthetic field sample", use_container_width=True)

    with right:
        region = st.selectbox("Region", REGIONS, index=1)
        growth_stage = st.selectbox("Growth stage", GROWTH_STAGES, index=1)
        temp_anomaly = st.slider("Temperature anomaly", 0.0, 1.0, float(default["temp_anomaly"]), 0.01)
        rainfall_deficit = st.slider("Rainfall deficit", 0.0, 1.0, float(default["rainfall_deficit"]), 0.01)
        humidity_index = st.slider("Humidity index", 0.0, 1.0, float(default["humidity_index"]), 0.01)
        soil_quality = st.slider("Soil quality", 0.0, 1.0, float(default["soil_quality"]), 0.01)
        management_quality = st.slider("Management quality", 0.0, 1.0, float(default["management_quality"]), 0.01)
        ndvi_trend = st.slider("NDVI / vegetation trend", 0.0, 1.0, float(default["ndvi_trend"]), 0.01)
        sensor_noise = st.slider("Sensor / observation noise", 0.0, 0.5, float(default["sensor_noise"]), 0.01)
        missing_weather = st.toggle("Simulate missing weather input", value=False)

    image_signal, heatmap = extract_image_signal(image_array)
    scores = model_scores_for_inputs(
        image_signal=image_signal,
        ndvi_trend=ndvi_trend,
        temp_anomaly=temp_anomaly,
        rainfall_deficit=rainfall_deficit,
        humidity_index=humidity_index,
        soil_quality=soil_quality,
        management_quality=management_quality,
        sensor_noise=sensor_noise,
        missing_weather=missing_weather,
    )

    stage_factor = {"Vegetative": 0.18, "Flowering": 0.34, "Grain fill": 0.29, "Ripening": 0.22}[growth_stage]
    region_factor = {"Delta": 0.10, "Semi-arid South": 0.28, "Coastal East": 0.16, "River Plains": 0.14}[region]

    contributions = {
        "Heat stress": 1.9 * temp_anomaly,
        "Rainfall deficit": 1.7 * rainfall_deficit,
        "Weak image signal": 1.2 * (1 - image_signal),
        "Low soil quality": 1.1 * (1 - soil_quality),
        "Weak management": 0.9 * (1 - management_quality),
        "Vegetation decline": 1.0 * (1 - ndvi_trend),
        "Growth-stage exposure": stage_factor,
        "Regional transfer risk": region_factor,
    }

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        metric_card("Stress-risk score", f"{scores['multimodal_stress']:.2f}", "Multimodal prototype prediction.")
    with c2:
        risk_class = "High" if scores["multimodal_yield"] >= 0.6 else "Moderate" if scores["multimodal_yield"] >= 0.35 else "Low"
        metric_card("Yield-risk class", risk_class, "Derived from multimodal yield-risk score.")
    with c3:
        confidence = max(abs(scores["multimodal_stress"] - 0.5), abs(scores["multimodal_yield"] - 0.5)) * (0.92 - (0.14 if missing_weather else 0))
        metric_card("Confidence", f"{confidence:.2f}", "Reduced when missing weather is simulated.")

    comparison = pd.DataFrame(
        [
            ["Image-only", scores["image_only_stress"], scores["image_only_yield"]],
            ["Tabular-only proxy", scores["tabular_only_stress"], scores["tabular_only_yield"]],
            ["Multimodal fusion", scores["multimodal_stress"], scores["multimodal_yield"]],
            ["Physics-aware fusion", scores["physics_stress"], scores["physics_yield"]],
        ],
        columns=["Model", "Stress risk", "Yield risk"],
    )
    st.dataframe(comparison.round(3), use_container_width=True, hide_index=True)

    top_features = (
        pd.DataFrame({"Feature": list(contributions.keys()), "Influence": list(contributions.values())})
        .sort_values("Influence", ascending=False)
        .head(4)
    )
    st.markdown("### Top influential factors")
    st.dataframe(top_features.round(3), use_container_width=True, hide_index=True)

    limitations = [
        "Predictions become less trustworthy when weather inputs are missing.",
        "Single images can mislead the model when visual appearance shifts across regions or sensors.",
        "The live predictor is a transparent prototype scorer, not a deployed agronomic model.",
    ]
    st.markdown("### Limitations and likely failure cases")
    for item in limitations:
        st.markdown(f"- {item}")

    st.session_state["explain_image"] = image_array
    st.session_state["explain_heatmap"] = heatmap
    st.session_state["explain_scores"] = scores
    st.session_state["explain_contributions"] = contributions
    return image_array, heatmap, scores, contributions


def render_explainability(image_array: np.ndarray, heatmap: np.ndarray, scores: dict[str, float], contributions: dict[str, float]) -> None:
    st.markdown('<div class="section-label">Explainability</div>', unsafe_allow_html=True)
    st.subheader("Where the prototype thinks the risk is coming from")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.image(image_array, caption="Input image", use_container_width=True)
        heatmap_fig = px.imshow(heatmap, color_continuous_scale=["#2b1c1c", "#f8c26a", "#59e2bf"])
        heatmap_fig.update_layout(**chart_layout(), height=300, coloraxis_showscale=False)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    with right:
        contrib_df = pd.DataFrame(
            {"Factor": list(contributions.keys()), "Influence": list(contributions.values())}
        ).sort_values("Influence", ascending=True)
        contrib_fig = px.bar(
            contrib_df,
            x="Influence",
            y="Factor",
            orientation="h",
            color="Influence",
            color_continuous_scale=["#84abff", "#f8c26a", "#ff7c7c"],
        )
        contrib_fig.update_layout(**chart_layout(), height=420, coloraxis_showscale=False, yaxis_title="", xaxis_title="Relative influence on risk")
        st.plotly_chart(contrib_fig, use_container_width=True)

    st.markdown(
        """
        <div class="panel-card">
            <div class="section-label">Important Honesty Note</div>
            <p class="body-copy">
            This page uses a transparent prototype scorer rather than true Grad-CAM or SHAP from a saved trained model.
            That is intentional. The goal is to show the reasoning structure of a multimodal agricultural system and
            make failure analysis visible, without pretending the synthetic pilot is already a production-ready model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    small = pd.DataFrame(
        [
            ["Image-only stress", scores["image_only_stress"]],
            ["Multimodal stress", scores["multimodal_stress"]],
            ["Physics-aware stress", scores["physics_stress"]],
        ],
        columns=["Prediction", "Score"],
    )
    delta_fig = px.bar(small, x="Prediction", y="Score", color="Prediction")
    delta_fig.update_layout(**chart_layout(), height=280, showlegend=False, xaxis_title="", yaxis_title="Stress score")
    delta_fig.update_yaxes(range=[0, 1], gridcolor="rgba(255,255,255,0.08)")
    st.plotly_chart(delta_fig, use_container_width=True)


def render_generalization_test(runs: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">Generalization Test</div>', unsafe_allow_html=True)
    st.subheader("This is the page that speaks most directly to the vacancy")

    metric = st.selectbox("View benchmark metric", ["score", "f1", "ece"], index=0, key="gen_metric")
    st.plotly_chart(build_benchmark_bar(runs, metric), use_container_width=True)

    split_summary = (
        runs.groupby("split", as_index=False)
        .agg(best_score=("score", "max"), worst_score=("score", "min"))
        .sort_values("best_score")
    )
    split_summary["gap"] = split_summary["best_score"] - split_summary["worst_score"]

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.dataframe(split_summary.round(3), use_container_width=True, hide_index=True)
    with right:
        gap_fig = px.bar(split_summary, x="split", y="gap", color="gap", color_continuous_scale=["#84abff", "#59e2bf"])
        gap_fig.update_layout(**chart_layout(), height=320, coloraxis_showscale=False, xaxis_title="", yaxis_title="Best-minus-worst gap")
        gap_fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(gap_fig, use_container_width=True)

    st.markdown(
        """
        <div class="panel-card">
            <div class="section-label">Failure Analysis</div>
            <p class="body-copy">
            The synthetic benchmark suggests that multimodal and physics-aware variants help strongly under region,
            year, climate, and sensor shift, but they remain brittle when a whole modality disappears. That failure
            mode is important because real agricultural systems often face partial observations, missing sensor feeds,
            and inconsistent metadata.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_reflection() -> None:
    st.markdown('<div class="section-label">Research Reflection</div>', unsafe_allow_html=True)
    st.subheader("What this prototype gets right and what it still lacks")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("### What worked")
        st.markdown("- The project is now centered on one agricultural use case instead of several scattered ideas.")
        st.markdown("- It combines image, time-series, and field metadata, which is much closer to the vacancy.")
        st.markdown("- It treats generalization as the main challenge, not a side metric.")
        st.markdown("- It includes a scientifically grounded variant without overstating the maturity of the work.")
    with right:
        st.markdown("### What still needs real work")
        st.markdown("- Replace the synthetic pilot with one real agricultural benchmark.")
        st.markdown("- Train and save an actual multimodal model for live inference and true explainability.")
        st.markdown("- Add a stronger tabular-only learned baseline to complete the ablation story.")
        st.markdown("- Expand from crop stress and yield risk only after the first use case is solid.")

    st.markdown("### How to frame it for each professor")
    st.dataframe(SUPERVISOR_ANGLES, use_container_width=True, hide_index=True)


def main() -> None:
    inject_styles()
    runs = load_runs()
    if runs.empty:
        st.error("No benchmark file found at `data/experiment_runs.csv`.")
        return

    st.sidebar.title(APP_TITLE)
    st.sidebar.caption(APP_SUBTITLE)
    page = st.sidebar.radio("Navigate", PAGES, index=0)
    st.sidebar.markdown("### Core question")
    st.sidebar.write("Can multimodal agricultural learning generalize better than single-modality modeling?")
    st.sidebar.markdown("### Evidence")
    st.sidebar.write("Synthetic benchmark plus interactive prototype scorer")

    if page == "Home":
        render_home(runs)
    elif page == "Data Explorer":
        render_data_explorer()
    elif page == "Model Lab":
        render_model_lab(runs)
    elif page == "Prediction Demo":
        render_prediction_demo()
    elif page == "Explainability":
        if all(key in st.session_state for key in ["explain_image", "explain_heatmap", "explain_scores", "explain_contributions"]):
            render_explainability(
                st.session_state["explain_image"],
                st.session_state["explain_heatmap"],
                st.session_state["explain_scores"],
                st.session_state["explain_contributions"],
            )
        else:
            default_image = generate_canopy_image(stress=0.68, vigor=0.39, humidity=0.36, seed=17)
            image_signal, heatmap = extract_image_signal(default_image)
            scores = model_scores_for_inputs(
                image_signal=image_signal,
                ndvi_trend=0.39,
                temp_anomaly=0.73,
                rainfall_deficit=0.68,
                humidity_index=0.36,
                soil_quality=0.48,
                management_quality=0.51,
                sensor_noise=0.14,
                missing_weather=False,
            )
            contributions = {
                "Heat stress": 1.39,
                "Rainfall deficit": 1.16,
                "Weak image signal": 0.84,
                "Low soil quality": 0.57,
                "Weak management": 0.44,
                "Vegetation decline": 0.61,
            }
            render_explainability(default_image, heatmap, scores, contributions)
    elif page == "Generalization Test":
        render_generalization_test(runs)
    else:
        render_reflection()


if __name__ == "__main__":
    main()
