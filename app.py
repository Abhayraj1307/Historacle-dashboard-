import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats as sci_stats
from datetime import datetime, timedelta
import io
import json

try:
    import anthropic
except ImportError:
    anthropic = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HistOracle War Room",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&family=Inter:wght@300;400;600;800&display=swap');

* { box-sizing: border-box; }

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 20% 0%, rgba(180,100,20,0.12) 0%, transparent 50%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(100,50,10,0.1) 0%, transparent 50%),
        linear-gradient(160deg, #080400 0%, #0a0a0a 40%, #050810 100%);
    color: #f0e6cc;
    font-family: 'Inter', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0700 0%, #050505 100%);
    border-right: 1px solid rgba(212,168,67,0.2);
}

.warroom-header {
    text-align: center;
    padding: 52px 32px 36px;
    margin-bottom: 8px;
    border-bottom: 1px solid rgba(212,168,67,0.12);
}

.warroom-eyebrow {
    font-size: 11px;
    letter-spacing: 5px;
    color: #7a6030;
    text-transform: uppercase;
    margin-bottom: 16px;
}

.warroom-title {
    font-family: 'Cinzel', serif;
    font-size: 58px;
    font-weight: 900;
    color: #d4a843;
    letter-spacing: 8px;
    text-transform: uppercase;
    text-shadow: 0 0 30px rgba(212,168,67,0.35), 0 0 80px rgba(212,168,67,0.15);
    margin-bottom: 4px;
    line-height: 1.05;
}

.warroom-divider {
    width: 240px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #d4a843, transparent);
    margin: 16px auto;
}

.warroom-subtitle {
    font-size: 12px;
    color: #8a7040;
    letter-spacing: 3.5px;
    text-transform: uppercase;
}

.section-header {
    font-family: 'Cinzel', serif;
    font-size: 18px;
    color: #d4a843;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 36px 0 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(212,168,67,0.12);
}

.verdict-box {
    background: linear-gradient(135deg, rgba(212,168,67,0.07), rgba(0,0,0,0.5));
    border: 1px solid rgba(212,168,67,0.3);
    border-left: 3px solid #d4a843;
    border-radius: 14px;
    padding: 24px 28px;
    margin: 20px 0;
}

.verdict-label {
    font-size: 10px;
    letter-spacing: 3px;
    color: #d4a843;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.verdict-text {
    font-size: 15px;
    color: #e8dcc0;
    line-height: 1.75;
}

.insight-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(212,168,67,0.12);
    border-radius: 14px;
    padding: 20px;
    height: 100%;
}

.insight-card:hover {
    border-color: rgba(212,168,67,0.3);
}

.insight-title {
    font-size: 13px;
    font-weight: 700;
    color: #d4a843;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.insight-body {
    font-size: 13px;
    color: #b4a480;
    line-height: 1.65;
}

.persona-card {
    background: linear-gradient(145deg, rgba(212,168,67,0.06), rgba(0,0,0,0.35));
    border: 1px solid rgba(212,168,67,0.15);
    border-radius: 18px;
    padding: 24px;
    height: 100%;
    position: relative;
    overflow: hidden;
}

.persona-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(212,168,67,0.5), transparent);
}

.persona-name {
    font-family: 'Cinzel', serif;
    font-size: 15px;
    color: #f5d58a;
    margin-bottom: 4px;
}

.persona-age {
    font-size: 11px;
    color: #8a7040;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.persona-body {
    font-size: 13px;
    color: #b4a480;
    line-height: 1.65;
}

.persona-tag {
    display: inline-block;
    background: rgba(212,168,67,0.1);
    border: 1px solid rgba(212,168,67,0.2);
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 11px;
    color: #d4a843;
    margin: 3px 2px 0;
}

.roadmap-phase {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(212,168,67,0.1);
    border-left: 3px solid #d4a843;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 10px 0;
}

.roadmap-phase-label {
    font-size: 10px;
    color: #8a6030;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.roadmap-phase-title {
    font-family: 'Cinzel', serif;
    font-size: 17px;
    color: #d4a843;
    margin-bottom: 8px;
}

.chat-user {
    background: rgba(212,168,67,0.1);
    border: 1px solid rgba(212,168,67,0.2);
    border-radius: 14px 14px 4px 14px;
    padding: 14px 18px;
    margin: 10px 0 10px 48px;
    font-size: 14px;
    color: #f0e6cc;
}

.chat-ai {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px 14px 14px 4px;
    padding: 14px 18px;
    margin: 10px 48px 10px 0;
    font-size: 14px;
    color: #c4b490;
    line-height: 1.7;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(212,168,67,0.07), rgba(0,0,0,0.3));
    border: 1px solid rgba(212,168,67,0.15);
    border-radius: 14px;
    padding: 18px 20px;
}

div[data-testid="stMetricLabel"] {
    color: #8a7040 !important;
    letter-spacing: 1px;
    font-size: 12px !important;
}

div[data-testid="stMetricValue"] {
    color: #f5d58a !important;
    font-family: 'Cinzel', serif !important;
}

div[data-testid="stMetricDelta"] {
    color: #8a7040 !important;
    font-size: 11px !important;
}

div[data-baseweb="tab-list"] {
    background: rgba(0,0,0,0.5) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 3px !important;
    border: 1px solid rgba(212,168,67,0.1) !important;
}

div[data-baseweb="tab"] {
    color: #6a5828 !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
    border-radius: 10px !important;
    padding: 8px 16px !important;
}

div[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(212,168,67,0.15) !important;
    color: #d4a843 !important;
    border: 1px solid rgba(212,168,67,0.25) !important;
}

.sidebar-logo {
    text-align: center;
    padding: 20px 0 28px;
    border-bottom: 1px solid rgba(212,168,67,0.1);
    margin-bottom: 20px;
}

.footer {
    text-align: center;
    padding: 36px;
    color: #3a2808;
    font-size: 11px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    border-top: 1px solid rgba(212,168,67,0.06);
    margin-top: 60px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def fmt(n):
    if pd.isna(n):
        return "0"
    try:
        n = float(n)
    except Exception:
        return str(n)
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(int(n))


def generate_script(
    pred_platform,
    pred_topic,
    pred_hook,
    pred_format,
    top_tier_prob,
    view_tier,
    best_pred_hook,
    use_best_hook,
    draft_title
):
    final_hook = best_pred_hook if use_best_hook else pred_hook

    topic_key = str(pred_topic).lower()
    hook_key = str(final_hook).lower()

    figure_map = {
        "personalities": "Napoleon",
        "personality": "Napoleon",
        "people": "Napoleon",
        "war": "a battlefield commander",
        "wars": "a battlefield commander",
        "ancient": "an ancient ruler",
        "civilization": "an ancient ruler",
        "civilizations": "an ancient ruler",
        "mystery": "a forgotten historian",
        "empires": "a fallen emperor",
        "empire": "a fallen emperor",
    }

    problem_map = {
        "personalities": "why ambitious people fail",
        "personality": "why ambitious people fail",
        "people": "why ambitious people fail",
        "war": "what destroys strong leaders",
        "wars": "what destroys strong leaders",
        "ancient": "why societies collapse",
        "civilization": "why civilizations collapse",
        "civilizations": "why civilizations collapse",
        "mystery": "why the truth gets buried",
        "empires": "why empires rot from inside",
        "empire": "why empires rot from inside",
    }

    figure = "Napoleon"
    problem = "why ambitious people fail"

    for key in figure_map:
        if key in topic_key:
            figure = figure_map[key]
            problem = problem_map[key]
            break

    if "question" in hook_key:
        hook_line = f"I asked {figure} {problem}… and his answer was uncomfortable."
    elif "mystery" in hook_key:
        hook_line = f"History buried the real reason {problem}."
    elif "controversial" in hook_key:
        hook_line = f"Most people are wrong about {problem}."
    elif "emotional" in hook_key:
        hook_line = "The reason people fail is uglier than you think."
    else:
        hook_line = f"I asked {figure} {problem}… and the answer exposed a brutal pattern."

    if top_tier_prob < 0.30:
        warning = "⚠️ LOW SIGNAL: Do not post this version blindly. Change the hook, format, or platform first."
    elif top_tier_prob < 0.60:
        warning = "🟡 MODERATE SIGNAL: Usable, but the first 3 seconds must be sharper."
    else:
        warning = "🔥 STRONG SIGNAL: This concept is worth testing immediately."

    return f"""
HOOK:
{hook_line}

60-SECOND SCRIPT:

[0–2 sec — PATTERN BREAK]
"You're not failing because you're lazy."

[3–8 sec — CURIOSITY LOOP]
"You're failing because you're repeating the same mistakes powerful people made centuries ago."

[9–16 sec — AUTHORITY]
"I asked {figure} {problem}."

"And the answer came down to one thing: timing."

[17–32 sec — TENSION]
"You move too late when it matters."

"You trust applause before results."

"You confuse attention with actual power."

[33–44 sec — CONTRADICTION]
"The scary part?"

"Most people never realize they're doing it."

[45–54 sec — PRODUCT MOMENT]
"That is why HistOracle Teleport exists."

"You do not just read history."

"You interrogate it."

[55–60 sec — LOOP CLOSE]
"If you could ask one dead genius why you're stuck… who would it be?"

CAPTION:
History is not boring. It was just taught like a dead subject. HistOracle Teleport turns the past into a conversation.

CTA:
Comment the historical figure you would question first.

HASHTAGS:
#HistOracle #AIHistory #HistoryTok #EdTech #LearnWithAI #ArtificialIntelligence

PERFORMANCE INTEL:
Draft Idea: {draft_title}
Platform: {pred_platform}
Topic: {pred_topic}
Hook: {final_hook}
Format: {pred_format}

Top-Tier Probability: {top_tier_prob * 100:.1f}%
Predicted View Tier: {view_tier}

{warning}
"""


PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(212,168,67,0.02)",
    font=dict(color="#b4a070", family="Inter", size=12),
    xaxis=dict(gridcolor="rgba(212,168,67,0.06)", linecolor="rgba(212,168,67,0.15)", tickfont=dict(color="#7a6030")),
    yaxis=dict(gridcolor="rgba(212,168,67,0.06)", linecolor="rgba(212,168,67,0.15)", tickfont=dict(color="#7a6030")),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#a08040")),
    title_font=dict(color="#d4a843", size=14, family="Inter")
)

GOLD_SCALE = ["#1a0d00", "#3a2008", "#6a4018", "#a06828", "#d4a843", "#f5d58a"]


# ============================================================
# DATA + MODEL
# ============================================================

@st.cache_data
def load_and_enrich(uploaded_file=None):
    try:
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_excel("historacle_600_rows_clean.xlsx")
    except Exception:
        st.error("Dataset not found. Upload `historacle_600_rows_clean.xlsx` to continue.")
        st.stop()

    df.columns = [str(c).strip() for c in df.columns]

    required_cols = [
        "Platform", "Title/Caption", "Views", "Likes", "Comments",
        "Content Type", "Topic", "Hook Type", "Engagement Score"
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    for c in ["Views", "Likes", "Comments", "Engagement Score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    scaler = MinMaxScaler()

    base_cols = ["Views", "Likes", "Comments"]
    df["_vs_base"] = scaler.fit_transform(df[base_cols]).mean(axis=1)
    df["_vs_eng"] = scaler.fit_transform(df[["Engagement Score"]])
    df["Viral Score"] = df["_vs_base"] * 0.6 + df["_vs_eng"].squeeze() * 0.4
    df.drop(columns=["_vs_base", "_vs_eng"], inplace=True)

    df["Comment Rate"] = df["Comments"] / df["Views"].replace(0, np.nan) * 100
    df["Like Rate"] = df["Likes"] / df["Views"].replace(0, np.nan) * 100
    df["Comment Rate"] = df["Comment Rate"].fillna(0)
    df["Like Rate"] = df["Like Rate"].fillna(0)

    cluster_feats = scaler.fit_transform(df[["Views", "Engagement Score"]].fillna(0))
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["_cluster_raw"] = km.fit_predict(cluster_feats)

    rank_map = {
        v: i for i, v in enumerate(
            df.groupby("_cluster_raw")["Viral Score"]
            .mean()
            .sort_values(ascending=False)
            .index
        )
    }

    tier_map = {
        0: "🔥 Viral",
        1: "⚡ Average",
        2: "⚠️ Underperforming"
    }

    df["Content Tier"] = df["_cluster_raw"].map(rank_map).map(tier_map)
    df.drop(columns=["_cluster_raw"], inplace=True)

    return df


@st.cache_data
def train_viral_predictor(df):
    model_df = df.copy()

    viral_threshold = model_df["Viral Score"].quantile(0.75)
    model_df["Is Viral"] = (model_df["Viral Score"] >= viral_threshold).astype(int)

    # Tier classifier replaces unreliable views regression
    model_df["View Tier"] = pd.qcut(
        model_df["Views"],
        q=4,
        labels=["Low", "Medium", "High", "Breakout"],
        duplicates="drop"
    )

    features = ["Platform", "Topic", "Hook Type", "Content Type"]
    X = pd.get_dummies(model_df[features], drop_first=False)

    y_viral = model_df["Is Viral"]
    y_tier = model_df["View Tier"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_viral,
        test_size=0.25,
        random_state=42,
        stratify=y_viral
    )

    viral_clf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        max_depth=7,
        min_samples_leaf=4,
        class_weight="balanced"
    )
    viral_clf.fit(X_train, y_train)
    viral_accuracy = accuracy_score(y_test, viral_clf.predict(X_test))

    Xt_train, Xt_test, yt_train, yt_test = train_test_split(
        X,
        y_tier,
        test_size=0.25,
        random_state=42,
        stratify=y_tier
    )

    tier_clf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        max_depth=7,
        min_samples_leaf=4,
        class_weight="balanced"
    )
    tier_clf.fit(Xt_train, yt_train)
    tier_accuracy = accuracy_score(yt_test, tier_clf.predict(Xt_test))

    return viral_clf, tier_clf, X.columns.tolist(), viral_accuracy, tier_accuracy


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style='font-family:Cinzel,serif;font-size:22px;color:#d4a843;letter-spacing:4px;'>⚔️ HISTORACLE</div>
        <div style='font-size:9px;color:#5a4020;letter-spacing:3px;margin-top:5px;text-transform:uppercase;'>Elite War Room</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload Dataset (.xlsx)", type=["xlsx"])

    st.markdown("---")
    api_key = st.text_input(
        "🤖 Anthropic API Key",
        type="password",
        placeholder="Optional: only for AI Analyst"
    )

    st.markdown("---")
    df_raw = load_and_enrich(uploaded_file)

    platforms = sorted(df_raw["Platform"].dropna().unique())
    topics = sorted(df_raw["Topic"].dropna().unique())
    hooks = sorted(df_raw["Hook Type"].dropna().unique())
    content_types = sorted(df_raw["Content Type"].dropna().unique())

    st.markdown("**🎯 FILTERS**")
    sel_p = st.multiselect("Platforms", platforms, default=platforms)
    sel_t = st.multiselect("Topics", topics, default=topics)
    sel_h = st.multiselect("Hooks", hooks, default=hooks)
    sel_c = st.multiselect("Content Types", content_types, default=content_types)

    df = df_raw.copy()
    if sel_p:
        df = df[df["Platform"].isin(sel_p)]
    if sel_t:
        df = df[df["Topic"].isin(sel_t)]
    if sel_h:
        df = df[df["Hook Type"].isin(sel_h)]
    if sel_c:
        df = df[df["Content Type"].isin(sel_c)]

    if df.empty:
        st.error("No data available after filters. Adjust your filters.")
        st.stop()

    st.markdown(f"""
    <div style='text-align:center;margin-top:12px;padding:10px;
    background:rgba(212,168,67,0.05);border-radius:8px;
    border:1px solid rgba(212,168,67,0.12);font-size:12px;color:#8a7040;'>
        <b style='color:#d4a843;font-size:18px;'>{len(df)}</b><br>posts in view
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HEADER + COMPUTED VALUES
# ============================================================

st.markdown("""
<div class="warroom-header">
    <div class="warroom-eyebrow">Social Intelligence Command Center</div>
    <div class="warroom-title">HistOracle</div>
    <div class="warroom-divider"></div>
    <div class="warroom-subtitle">Elite War Room · Teleport Experience · Viral Predictor · AI Strategy</div>
</div>
""", unsafe_allow_html=True)


total_views = df["Views"].sum()
total_eng = df["Engagement Score"].sum()
avg_eng = df["Engagement Score"].mean()
total_posts = len(df)

best_platform = df.groupby("Platform")["Engagement Score"].sum().idxmax()
best_topic = df.groupby("Topic")["Engagement Score"].sum().idxmax()
best_hook = df.groupby("Hook Type")["Engagement Score"].sum().idxmax()
best_format = df.groupby("Content Type")["Engagement Score"].sum().idxmax()

viral_clf, tier_clf, model_columns, clf_accuracy, tier_accuracy = train_viral_predictor(df)

top_topics_list = df.groupby("Topic")["Viral Score"].mean().nlargest(4).index.tolist()
top_hooks_list = df.groupby("Hook Type")["Viral Score"].mean().nlargest(5).index.tolist()
top_plats_list = df.groupby("Platform")["Engagement Score"].sum().nlargest(4).index.tolist()
top_formats_list = df.groupby("Content Type")["Viral Score"].mean().nlargest(3).index.tolist()

templates = [
    "What did {topic} reveal that history books will not teach you?",
    "The untold truth of {topic} — this changes everything",
    "I talked to a historical {topic} expert on Teleport. Here is what happened.",
    "The {topic} mystery historians still argue about",
    "What would a {topic} figure say about the world today?",
    "The moment that defined {topic} forever",
    "Why {topic} is the most misunderstood chapter of history",
    "POV: Interviewing a {topic} figure before your history exam",
]

calendar_rows = []
start_date = datetime.today()

for i in range(30):
    topic = top_topics_list[i % len(top_topics_list)]
    hook = top_hooks_list[i % len(top_hooks_list)]
    platform = top_plats_list[i % len(top_plats_list)]
    content_format = top_formats_list[i % len(top_formats_list)]
    concept = templates[i % len(templates)].replace("{topic}", topic)

    calendar_rows.append({
        "Day": i + 1,
        "Date": (start_date + timedelta(days=i)).strftime("%b %d"),
        "Platform": platform,
        "Topic": topic,
        "Hook": hook,
        "Format": content_format,
        "Post Concept": concept,
        "Goal": ["Reach", "Comments", "Saves", "Shares", "Signups", "Comments"][i % 6]
    })

cal_df = pd.DataFrame(calendar_rows)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "🏛️ TELEPORT",
    "⚠️ DAILY BRIEF",
    "⚔️ COMMAND",
    "📊 REALITY CHECK",
    "📡 PLATFORM",
    "🔥 CONTENT",
    "🧬 VIRAL DNA",
    "🔮 PREDICTOR",
    "📝 COACH + LAB",
    "👁️ AUDIENCE",
    "⚡ COMPETITION",
    "💰 BUSINESS",
    "🤖 AI ANALYST",
    "📅 LAUNCH",
    "📥 EXPORT"
])


# ============================================================
# TAB 0 — TELEPORT
# ============================================================

with tabs[0]:
    st.markdown('<div class="section-header">🏛️ Teleport Experience</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="verdict-box">
        <div class="verdict-label">Product Experience First</div>
        <div class="verdict-text">
        HistOracle should not be sold as a dashboard or another education app.
        The product promise is simple: <strong>talk to the past like it is alive.</strong>
        This demo makes the strategy visible before the analytics prove it.
        </div>
    </div>
    """, unsafe_allow_html=True)

    figures = {
        "Julius Caesar": {
            "role": "Power, ambition, leadership",
            "reply": "Leaders fail when they confuse noise with loyalty. Power survives only when timing, discipline, and perception move together."
        },
        "Cleopatra": {
            "role": "Influence, diplomacy, charisma",
            "reply": "Influence is not beauty. Influence is knowing what people want before they are brave enough to say it."
        },
        "Napoleon": {
            "role": "Strategy, conquest, execution",
            "reply": "Most people lose before the battle begins because they wait for certainty. Victory belongs to those who move first with discipline."
        },
        "Abraham Lincoln": {
            "role": "Leadership, morality, crisis",
            "reply": "A leader is tested when the room wants comfort and the moment demands truth."
        },
        "Socrates": {
            "role": "Questions, wisdom, thinking",
            "reply": "The weak mind wants answers quickly. The strong mind survives better questions."
        }
    }

    left, right = st.columns([1, 1.2])
    with left:
        selected_figure = st.selectbox("Choose a historical figure", list(figures.keys()))
        user_question = st.text_input("Ask the past", "Why do leaders fail?")

        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">{selected_figure}</div>
            <div class="insight-body">
                <b>Strategic Lens:</b> {figures[selected_figure]["role"]}<br><br>
                <b>Use Case:</b> Students, teachers, creators, and history lovers turn passive learning into active dialogue.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div class="chat-user">
            🧠 {user_question}
        </div>
        <div class="chat-ai">
            🏛️ <b>{selected_figure}:</b><br>
            {figures[selected_figure]["reply"]}
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 1 — VIRAL PREDICTOR
# ============================================================

with tabs[7]:
    st.markdown('<div class="section-header">🔮 Viral Predictor</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="verdict-box">
        <div class="verdict-label">Predictive Brain</div>
        <div class="verdict-text">
        This engine predicts whether a draft content idea has top-tier potential before posting.
        It uses platform, topic, hook type, and content format patterns learned from the dataset.
        Exact view regression was removed because it was unreliable; this version uses a stronger view-tier classifier.
        </div>
    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        pred_platform = st.selectbox(
            "Platform",
            platforms,
            index=platforms.index(best_platform) if best_platform in platforms else 0
        )
    with p2:
        pred_topic = st.selectbox(
            "Topic",
            topics,
            index=topics.index(best_topic) if best_topic in topics else 0
        )
    with p3:
        pred_hook = st.selectbox(
            "Hook Type",
            hooks,
            index=hooks.index(best_hook) if best_hook in hooks else 0
        )
    with p4:
        pred_format = st.selectbox(
            "Content Type",
            content_types,
            index=content_types.index(best_format) if best_format in content_types else 0
        )

    draft_title = st.text_input("Draft Post Idea", "I asked Napoleon why modern leaders fail")

    input_row = pd.DataFrame([{
        "Platform": pred_platform,
        "Topic": pred_topic,
        "Hook Type": pred_hook,
        "Content Type": pred_format
    }])

    input_encoded = pd.get_dummies(input_row)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    top_tier_prob = viral_clf.predict_proba(input_encoded)[0][1]
    view_tier = tier_clf.predict(input_encoded)[0]
    view_tier_probs = tier_clf.predict_proba(input_encoded)[0]
    view_tier_confidence = max(view_tier_probs)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Top-Tier Probability", f"{top_tier_prob * 100:.1f}%", "Chance of reaching top viral tier")
    with c2:
        st.metric("Predicted View Tier", view_tier, f"{view_tier_confidence * 100:.1f}% confidence")
    with c3:
        st.metric("Model Accuracy", f"{clf_accuracy * 100:.1f}%", "Top-tier classifier")

    st.markdown('<div class="section-header">🧠 Optimization Recommendation</div>', unsafe_allow_html=True)

    hook_tests = []
    for h in hooks:
        test_row = pd.DataFrame([{
            "Platform": pred_platform,
            "Topic": pred_topic,
            "Hook Type": h,
            "Content Type": pred_format
        }])
        test_encoded = pd.get_dummies(test_row)
        test_encoded = test_encoded.reindex(columns=model_columns, fill_value=0)

        prob = viral_clf.predict_proba(test_encoded)[0][1]
        tier = tier_clf.predict(test_encoded)[0]

        hook_tests.append({
            "Hook Type": h,
            "Top-Tier Probability": prob,
            "Predicted View Tier": tier
        })

    hook_test_df = pd.DataFrame(hook_tests).sort_values("Top-Tier Probability", ascending=False)
    best_pred_hook = hook_test_df.iloc[0]["Hook Type"]
    best_pred_prob = hook_test_df.iloc[0]["Top-Tier Probability"]

    if best_pred_hook != pred_hook:
        recommendation = f"Switch from {pred_hook} to {best_pred_hook}. Top-tier probability improves to {best_pred_prob * 100:.1f}%."
    else:
        recommendation = f"Keep {pred_hook}. It is currently the strongest predicted hook for this setup."

    st.markdown(f"""
    <div class="verdict-box">
        <div class="verdict-label">Final Prediction</div>
        <div class="verdict-text">
        Draft idea: <strong>{draft_title}</strong><br><br>
        Current setup has a <strong>{top_tier_prob * 100:.1f}%</strong> predicted chance of reaching the top-tier content group.
        Predicted view tier: <strong>{view_tier}</strong> with <strong>{view_tier_confidence * 100:.1f}%</strong> confidence.<br><br>
        <strong>Optimization:</strong> {recommendation}
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig = px.bar(
        hook_test_df,
        x="Hook Type",
        y="Top-Tier Probability",
        color="Top-Tier Probability",
        color_continuous_scale=GOLD_SCALE,
        title="Hook Optimization Test"
    )
    fig.update_layout(**PLOTLY_BASE)
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">⚔️ Content Agent — Elite Psychology Mode</div>', unsafe_allow_html=True)

    use_best_hook = st.checkbox("Use optimized hook recommendation", value=True)

    if st.button("⚔️ Generate Viral Script"):
        script = generate_script(
            pred_platform,
            pred_topic,
            pred_hook,
            pred_format,
            top_tier_prob,
            view_tier,
            best_pred_hook,
            use_best_hook,
            draft_title
        )

        st.markdown(f"""
        <div class="verdict-box">
            <div class="verdict-label">Generated Viral Script</div>
            <div class="verdict-text">
            <pre style="white-space: pre-wrap; color:#e8dcc0; font-family:Inter; font-size:14px; line-height:1.6;">{script}</pre>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.download_button(
            label="⬇️ Download Script",
            data=script,
            file_name="historacle_viral_script.txt",
            mime="text/plain"
        )

    with st.expander("Model details"):
        st.write(f"Top-tier classifier accuracy: {clf_accuracy:.3f}")
        st.write(f"View tier classifier accuracy: {tier_accuracy:.3f}")
        st.dataframe(hook_test_df, use_container_width=True)


# ============================================================
# TAB 2 — COMMAND OVERVIEW
# ============================================================

with tabs[2]:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Views", fmt(total_views), "Market attention captured")
    with c2:
        st.metric("Total Engagement", fmt(total_eng), "Cross-platform signal")
    with c3:
        st.metric("Avg Engagement", fmt(avg_eng), "Per post baseline")
    with c4:
        st.metric("Posts Analyzed", str(total_posts), "Evidence base")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Best Platform", best_platform, "Lead here first")
    with c6:
        st.metric("Best Topic", best_topic, "Proven content territory")
    with c7:
        st.metric("Best Hook", best_hook, "Use this formula")
    with c8:
        st.metric("Best Format", best_format, "Primary content type")

    st.markdown(f"""
    <div class="verdict-box">
        <div class="verdict-label">⚔️ Executive Verdict</div>
        <div class="verdict-text">
        Analysis of <strong>{total_posts:,} posts</strong> confirms:
        <strong>{best_platform}</strong> is the highest-leverage distribution channel,
        <strong>{best_topic}</strong> is the strongest topic territory,
        <strong>{best_hook}</strong> is the strongest hook type, and
        <strong>{best_format}</strong> should lead the content calendar.<br><br>
        <strong>Strategic imperative:</strong> HistOracle is not an education app.
        It is an <strong>AI-powered historical interaction platform</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        plat_ov = df.groupby("Platform").agg(
            Views=("Views", "sum"),
            Engagement=("Engagement Score", "sum")
        ).reset_index().sort_values("Engagement", ascending=False)

        fig = go.Figure()
        fig.add_bar(x=plat_ov["Platform"], y=plat_ov["Engagement"], name="Engagement", marker_color="#d4a843")
        fig.add_bar(x=plat_ov["Platform"], y=plat_ov["Views"] * 0.1, name="Views ÷10", marker_color="rgba(212,168,67,0.3)")
        fig.update_layout(**PLOTLY_BASE, barmode="group", title_text="Platform Overview")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        topic_ov = df.groupby("Topic")["Engagement Score"].sum().reset_index().sort_values("Engagement Score")
        fig2 = px.bar(
            topic_ov,
            x="Engagement Score",
            y="Topic",
            orientation="h",
            color="Engagement Score",
            color_continuous_scale=GOLD_SCALE,
            title="Topic Engagement Ranking"
        )
        fig2.update_layout(**PLOTLY_BASE)
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# TAB 3 — PLATFORM WAR ROOM
# ============================================================

with tabs[4]:
    st.markdown('<div class="section-header">📡 Platform Intelligence</div>', unsafe_allow_html=True)

    plat_full = df.groupby("Platform").agg(
        Posts=("Platform", "count"),
        Total_Views=("Views", "sum"),
        Total_Likes=("Likes", "sum"),
        Total_Comments=("Comments", "sum"),
        Total_Engagement=("Engagement Score", "sum"),
        Avg_Engagement=("Engagement Score", "mean"),
        Avg_Viral_Score=("Viral Score", "mean")
    ).reset_index().sort_values("Total_Engagement", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            plat_full,
            x="Platform",
            y="Total_Engagement",
            color="Total_Engagement",
            color_continuous_scale=GOLD_SCALE,
            title="Total Engagement by Platform",
            text="Posts"
        )
        fig.update_layout(**PLOTLY_BASE)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.scatter(
            plat_full,
            x="Total_Views",
            y="Avg_Engagement",
            size="Posts",
            color="Avg_Viral_Score",
            text="Platform",
            color_continuous_scale=GOLD_SCALE,
            title="Platform Efficiency Map"
        )
        fig2.update_traces(textposition="top center")
        fig2.update_layout(**PLOTLY_BASE)
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# TAB 4 — CONTENT INTELLIGENCE
# ============================================================

with tabs[5]:
    st.markdown('<div class="section-header">🔥 Content Intelligence</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        hook_df = df.groupby("Hook Type").agg(
            Posts=("Hook Type", "count"),
            Total_Engagement=("Engagement Score", "sum"),
            Avg_Viral=("Viral Score", "mean")
        ).reset_index().sort_values("Total_Engagement", ascending=False)

        fig = px.bar(
            hook_df,
            x="Hook Type",
            y="Total_Engagement",
            color="Avg_Viral",
            color_continuous_scale=GOLD_SCALE,
            title="Which Hooks Win?",
            text="Posts"
        )
        fig.update_layout(**PLOTLY_BASE)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        ct_df = df.groupby("Content Type").agg(
            Posts=("Content Type", "count"),
            Total_Engagement=("Engagement Score", "sum"),
            Avg_Viral=("Viral Score", "mean")
        ).reset_index().sort_values("Total_Engagement", ascending=False)

        fig2 = px.bar(
            ct_df,
            x="Content Type",
            y="Total_Engagement",
            color="Avg_Viral",
            color_continuous_scale=GOLD_SCALE,
            title="Content Format Performance",
            text="Posts"
        )
        fig2.update_layout(**PLOTLY_BASE)
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">🗺️ Strategic Heatmaps</div>', unsafe_allow_html=True)

    h1, h2 = st.columns(2)
    with h1:
        hm1 = df.pivot_table(index="Topic", columns="Platform", values="Engagement Score", aggfunc="sum", fill_value=0)
        fig3 = px.imshow(hm1, text_auto=".2s", aspect="auto", color_continuous_scale=GOLD_SCALE, title="Topic × Platform Engagement")
        fig3.update_layout(**PLOTLY_BASE)
        st.plotly_chart(fig3, use_container_width=True)

    with h2:
        hm2 = df.pivot_table(index="Hook Type", columns="Platform", values="Viral Score", aggfunc="mean", fill_value=0)
        fig4 = px.imshow(hm2, text_auto=".3f", aspect="auto", color_continuous_scale=GOLD_SCALE, title="Hook × Platform Viral Score")
        fig4.update_layout(**PLOTLY_BASE)
        st.plotly_chart(fig4, use_container_width=True)


# ============================================================
# TAB 5 — VIRAL DNA
# ============================================================

with tabs[6]:
    st.markdown('<div class="section-header">🧬 Viral DNA Engine</div>', unsafe_allow_html=True)

    viral_df = df[df["Content Tier"] == "🔥 Viral"]
    avg_df = df[df["Content Tier"] == "⚡ Average"]
    under_df = df[df["Content Tier"] == "⚠️ Underperforming"]

    cv1, cv2, cv3, cv4 = st.columns(4)
    with cv1:
        st.metric("🔥 Viral Posts", len(viral_df), f"{len(viral_df)/len(df)*100:.1f}%")
    with cv2:
        st.metric("⚡ Average Posts", len(avg_df), f"{len(avg_df)/len(df)*100:.1f}%")
    with cv3:
        st.metric("⚠️ Underperforming", len(under_df), f"{len(under_df)/len(df)*100:.1f}%")
    with cv4:
        st.metric("Peak Viral Score", f"{df['Viral Score'].max():.3f}", "Maximum recorded")

    fig = px.scatter(
        df,
        x="Views",
        y="Engagement Score",
        color="Content Tier",
        size="Viral Score",
        hover_data=["Platform", "Topic", "Hook Type"],
        color_discrete_map={
            "🔥 Viral": "#22c55e",
            "⚡ Average": "#f59e0b",
            "⚠️ Underperforming": "#ef4444"
        },
        title="Content DNA Cluster Map"
    )
    fig.update_layout(**PLOTLY_BASE)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">🏆 Winning Content Formula</div>', unsafe_allow_html=True)

    best_combos = (
        viral_df.groupby(["Topic", "Hook Type", "Content Type"])["Viral Score"]
        .mean()
        .nlargest(10)
        .reset_index()
    )

    best_combos.columns = ["Topic", "Hook Type", "Format", "Avg Viral Score"]
    best_combos.index = range(1, len(best_combos) + 1)
    st.dataframe(best_combos, use_container_width=True)


# ============================================================
# TAB 6 — AUDIENCE
# ============================================================

with tabs[9]:
    st.markdown('<div class="section-header">👁️ Target Audience & ICP</div>', unsafe_allow_html=True)

    personas = [
        {
            "icon": "⚔️",
            "name": "The Exam Survivor",
            "age": "Ages 14–22",
            "pain": "Textbooks are dead to them. They need history to feel alive, fast, visual, and shareable.",
            "channels": ["TikTok", "Instagram Reels", "YouTube Shorts"],
            "hook": "Mystery + Question hooks",
            "format": "Short-form video ≤60 sec",
            "value": "High volume · Viral sharers · Future brand evangelists",
            "why": "Teleport gives them interactive conversations with historical figures. That is an exam prep weapon."
        },
        {
            "icon": "📚",
            "name": "The Modern Teacher",
            "age": "Ages 26–45",
            "pain": "Students are disengaged. They need safe, curriculum-aligned, engaging AI tools.",
            "channels": ["YouTube", "LinkedIn", "Facebook Groups"],
            "hook": "Emotional + Controversial hooks",
            "format": "Long-form, carousels, guides",
            "value": "High LTV · Institutional deals · B2B revenue",
            "why": "Teachers unlock classrooms. One pilot can convert many student users."
        },
        {
            "icon": "🏛️",
            "name": "The History Binger",
            "age": "Ages 25–50",
            "pain": "They have watched every documentary. They crave interaction, not passive consumption.",
            "channels": ["YouTube", "Reddit", "Podcasts"],
            "hook": "Mystery + You did not know",
            "format": "Deep dives, threads, analysis",
            "value": "Loyal base · Word of mouth · Community builders",
            "why": "Teleport lets them go deeper than any documentary."
        },
        {
            "icon": "👨‍👩‍👧",
            "name": "The AI-Curious Parent",
            "age": "Ages 30–50",
            "pain": "Screen time guilt. They want AI that educates rather than numbs.",
            "channels": ["Instagram", "Facebook", "YouTube"],
            "hook": "Emotional + Question hooks",
            "format": "Short demos, proof clips",
            "value": "Purchasing power · Family accounts · Safe AI narrative",
            "why": "HistOracle feels educational, safe, and visibly impressive."
        },
    ]

    cols = st.columns(4)
    for col, p in zip(cols, personas):
        tags_html = "".join([f'<span class="persona-tag">{t}</span>' for t in p["channels"]])

        with col:
            st.markdown(f"""
            <div class="persona-card">
                <div style="font-size:28px;margin-bottom:8px;">{p['icon']}</div>
                <div class="persona-name">{p['name']}</div>
                <div class="persona-age">{p['age']}</div>
                <div class="persona-body">
                    <b style="color:#d4a843">Core Pain:</b><br>{p['pain']}<br><br>
                    <b style="color:#d4a843">Best Hook:</b> {p['hook']}<br>
                    <b style="color:#d4a843">Best Format:</b> {p['format']}<br><br>
                    <b style="color:#d4a843">Channels:</b><br>{tags_html}<br><br>
                    <b style="color:#d4a843">Why HistOracle wins:</b><br>{p['why']}<br><br>
                    <span style="color:#6a5030;font-size:11px;">{p['value']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# TAB 7 — AI ANALYST
# ============================================================

with tabs[12]:
    st.markdown('<div class="section-header">🤖 AI Strategy Analyst</div>', unsafe_allow_html=True)

    if anthropic is None:
        st.warning("Install Anthropic first: `python3 -m pip install anthropic`")
    elif not api_key:
        st.warning("Enter your Anthropic API key in the sidebar to activate the AI Analyst.")
    else:
        plat_ctx = df.groupby("Platform").agg(
            Posts=("Platform", "count"),
            Avg_Views=("Views", "mean"),
            Total_Engagement=("Engagement Score", "sum"),
            Avg_Viral=("Viral Score", "mean")
        ).round(2).to_string()

        topic_ctx = df.groupby("Topic").agg(
            Posts=("Topic", "count"),
            Avg_Viral=("Viral Score", "mean"),
            Total_Engagement=("Engagement Score", "sum")
        ).round(3).to_string()

        hook_ctx = df.groupby("Hook Type").agg(
            Posts=("Hook Type", "count"),
            Avg_Viral=("Viral Score", "mean"),
            Total_Engagement=("Engagement Score", "sum")
        ).round(3).to_string()

        viral_df = df[df["Content Tier"] == "🔥 Viral"]

        if len(viral_df) > 0:
            viral_formula = (
                viral_df.groupby(["Topic", "Hook Type", "Content Type"])["Viral Score"]
                .mean()
                .nlargest(5)
                .reset_index()
                .to_string()
            )
        else:
            viral_formula = "N/A"

        system_prompt = f"""
You are an elite social media strategist and data analyst for HistOracle.
HistOracle Teleport lets users talk to historical figures through AI.

Dataset:
- Posts analyzed: {total_posts}
- Total views: {fmt(total_views)}
- Total engagement: {fmt(total_eng)}
- Best platform: {best_platform}
- Best topic: {best_topic}
- Best hook: {best_hook}
- Best format: {best_format}

Platform analytics:
{plat_ctx}

Topic analytics:
{topic_ctx}

Hook analytics:
{hook_ctx}

Top viral formulas:
{viral_formula}

Be ruthless, specific, and data-backed. No generic advice.
"""

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            css_class = "chat-user" if msg["role"] == "user" else "chat-ai"
            icon = "🧠" if msg["role"] == "user" else "⚔️"
            st.markdown(f'<div class="{css_class}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

        user_input = st.text_input(
            "Ask the AI Analyst...",
            placeholder="What is the single most important move HistOracle should make this week?"
        )

        btn_col, clear_col = st.columns([1, 6])

        with btn_col:
            send_btn = st.button("⚔️ Send")

        with clear_col:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

        if send_btn and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.spinner("Analyzing intelligence..."):
                try:
                    client = anthropic.Anthropic(api_key=api_key)
                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=1000,
                        system=system_prompt,
                        messages=st.session_state.chat_history
                    )
                    answer = response.content[0].text
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()
                except Exception as e:
                    st.error(f"API Error: {str(e)}")


# ============================================================
# TAB 8 — LAUNCH COMMAND
# ============================================================

with tabs[13]:
    st.markdown('<div class="section-header">📅 90-Day Launch Command</div>', unsafe_allow_html=True)

    phases = [
        ("PHASE I · Days 1–15", "Ignition", "Launch awareness. Hit first 10K engaged followers.",
         f"Focus 80% effort on {best_platform}. Post daily. Use {best_hook} hooks on {best_topic} content. Short-form only.",
         "Reach, views, follower velocity"),

        ("PHASE I · Days 16–30", "Signal Testing", "Find what converts — not just what gets views.",
         "A/B test topic × hook × format combinations. Kill losers weekly. Double down on winners within 48 hours.",
         "Engagement rate, comment depth, saves"),

        ("PHASE II · Days 31–45", "Domination", "Scale the winning formula. Increase content frequency.",
         "All-in on top 2 platforms. Produce proven content at volume. Start creator collaborations.",
         "Follower growth, share velocity"),

        ("PHASE II · Days 46–60", "Community Activation", "Turn passive audience into active participants.",
         "Run polls, challenges, and ask-a-historical-figure sessions.",
         "Comment rate, UGC volume"),

        ("PHASE III · Days 61–75", "Product Demo Wave", "Convert audience to users.",
         "Post Teleport walkthroughs, user reaction clips, and I-talked-to-Caesar content.",
         "Demo clicks, signups"),

        ("PHASE III · Days 76–90", "Conversion & Scale", "Drive paid users and educator pilots.",
         "Launch referral program. Activate educator outreach. Put paid ads behind top organic posts.",
         "CAC, signups, partnerships"),
    ]

    for label, title, objective, action, kpi in phases:
        st.markdown(f"""
        <div class="roadmap-phase">
            <div class="roadmap-phase-label">{label}</div>
            <div class="roadmap-phase-title">{title}</div>
            <div style="color:#e8dcc0;font-size:14px;font-weight:600;margin-bottom:8px;">{objective}</div>
            <div style="color:#8a7848;font-size:13px;line-height:1.6;">{action}</div>
            <div style="color:#5a4828;font-size:11px;margin-top:8px;letter-spacing:1px;">KPI: {kpi}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📆 Auto-Generated 30-Day Content Calendar</div>', unsafe_allow_html=True)
    st.dataframe(cal_df, use_container_width=True, height=500)

    st.markdown('<div class="section-header">🚀 CEO Decision</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="verdict-box">
        <div class="verdict-label">Final Strategic Move</div>
        <div class="verdict-text">
        For the next 14 days: ignore broad omnichannel expansion.
        Focus on <strong>{best_platform}</strong>, publish <strong>{best_format}</strong> content,
        lead with <strong>{best_topic}</strong>, and use <strong>{best_hook}</strong> hooks.
        Every post must show or imply the Teleport experience.
        If it does not make people want to ask the past a question, cut it.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 1 — DAILY BRIEF + ANOMALIES
# ============================================================

with tabs[1]:
    st.markdown('<div class="section-header">⚠️ Daily Brief — What A Marketing Lead Reads First</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="verdict-box">
        <div class="verdict-label">📅 BRIEF FOR {datetime.today().strftime('%A, %B %d, %Y').upper()}</div>
        <div class="verdict-text">
        Real marketing leaders do not stare at charts. They scan a one-page brief that tells them
        what changed, what broke the pattern, and what needs decisions today. Below is that brief — auto-generated
        from your data and refreshed every time you load the dashboard.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Anomaly detection: posts > 2 std above their cluster mean
    anomaly_rows = []
    cluster_keys = ['Platform', 'Topic', 'Hook Type', 'Content Type']
    for keys, group in df.groupby(cluster_keys):
        if len(group) < 5:
            continue
        mean_eng = group['Engagement Score'].mean()
        std_eng = group['Engagement Score'].std()
        if std_eng > 0:
            for _, row in group.iterrows():
                z = (row['Engagement Score'] - mean_eng) / std_eng
                if z > 2.0:
                    anomaly_rows.append({
                        'Type': '🟢 OVERPERFORMER',
                        'Title': row.get('Title/Caption', '')[:60],
                        'Platform': row['Platform'],
                        'Topic': row['Topic'],
                        'Hook': row['Hook Type'],
                        'Format': row['Content Type'],
                        'Views': fmt(row['Views']),
                        'Engagement': fmt(row['Engagement Score']),
                        'Z-Score': round(z, 2),
                        'Lift vs Avg': f"+{((row['Engagement Score']/mean_eng - 1)*100):.0f}%"
                    })
                elif z < -2.0:
                    anomaly_rows.append({
                        'Type': '🔴 UNDERPERFORMER',
                        'Title': row.get('Title/Caption', '')[:60],
                        'Platform': row['Platform'],
                        'Topic': row['Topic'],
                        'Hook': row['Hook Type'],
                        'Format': row['Content Type'],
                        'Views': fmt(row['Views']),
                        'Engagement': fmt(row['Engagement Score']),
                        'Z-Score': round(z, 2),
                        'Lift vs Avg': f"{((row['Engagement Score']/mean_eng - 1)*100):.0f}%"
                    })

    over_count = sum(1 for r in anomaly_rows if 'OVERPERFORMER' in r['Type'])
    under_count = sum(1 for r in anomaly_rows if 'UNDERPERFORMER' in r['Type'])

    # Top-line metrics
    db1, db2, db3, db4 = st.columns(4)
    with db1: st.metric("🟢 Overperformers", over_count, "z > 2.0")
    with db2: st.metric("🔴 Underperformers", under_count, "z < -2.0")
    with db3: st.metric("Best Combo Today", f"{best_platform} × {best_topic}", "Lead with this")
    with db4: st.metric("Filtered Posts", total_posts, "After sidebar filters")

    # Auto-generated executive bullets
    st.markdown('<div class="section-header">🎯 Executive Bullets</div>', unsafe_allow_html=True)

    plat_breakdown = df.groupby('Platform')['Engagement Score'].sum().sort_values(ascending=False)
    plat_lead_pct = (plat_breakdown.iloc[0] / plat_breakdown.sum()) * 100 if len(plat_breakdown) > 0 else 0

    topic_breakdown = df.groupby('Topic')['Engagement Score'].mean().sort_values(ascending=False)
    topic_lift = ((topic_breakdown.iloc[0] / topic_breakdown.iloc[-1]) - 1) * 100 if len(topic_breakdown) > 1 else 0

    bullets = [
        f"<b>{best_platform}</b> drives {plat_lead_pct:.1f}% of total engagement across the dataset — concentrate creative output here.",
        f"<b>{best_topic}</b> outperforms the weakest topic by {topic_lift:.0f}% on engagement — but check the Reality Check tab to see if this is statistically significant.",
        f"<b>{over_count} posts</b> broke the pattern positively. Reverse-engineer them — see the table below.",
        f"<b>{under_count} posts</b> failed unusually hard. Audit them — these are content the calendar should never produce again.",
        f"Predicted next-best content: <b>{best_platform} + {best_topic} + {best_hook} + {best_format}</b>. Run this combination as your highest-confidence post this week."
    ]

    for b in bullets:
        st.markdown(f"""
        <div class="insight-card" style="margin-bottom:10px;">
            <div class="insight-body">▸ {b}</div>
        </div>
        """, unsafe_allow_html=True)

    # Anomaly tables
    if anomaly_rows:
        st.markdown('<div class="section-header">📈 Outlier Report — Posts That Broke The Pattern</div>', unsafe_allow_html=True)

        anomaly_df = pd.DataFrame(anomaly_rows).sort_values('Z-Score', ascending=False)

        ovr_df = anomaly_df[anomaly_df['Type'].str.contains('OVERPERFORMER')]
        und_df = anomaly_df[anomaly_df['Type'].str.contains('UNDERPERFORMER')]

        if len(ovr_df) > 0:
            st.markdown("**🟢 Overperformers — Reverse-engineer these:**")
            st.dataframe(ovr_df.head(15), use_container_width=True, hide_index=True)

        if len(und_df) > 0:
            st.markdown("**🔴 Underperformers — Cut these patterns from your calendar:**")
            st.dataframe(und_df.head(10), use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="verdict-box">
            <div class="verdict-label">🧠 What To Do With This</div>
            <div class="verdict-text">
            Anomalies are where strategy lives. Average performance is noise — outliers are signal.
            Schedule a 15-minute weekly review of these tables. For every overperformer, ask
            <em>"what made this break the pattern?"</em> For every underperformer, ask
            <em>"what assumption did we get wrong?"</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No statistically significant anomalies detected in current filter view. Try expanding filters.")


# ============================================================
# TAB 3 — STATISTICAL REALITY CHECK
# ============================================================

with tabs[3]:
    st.markdown('<div class="section-header">📊 Statistical Reality Check — Are These Wins Real?</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="verdict-box">
        <div class="verdict-label">⚠️ The Question Every Senior Reviewer Asks</div>
        <div class="verdict-text">
        <b>"Is this signal or is this noise?"</b><br><br>
        Most marketing dashboards report a 12% engagement lift between platforms and call one a "winner."
        That number could easily be random variation. This tab runs <b>actual ANOVA tests</b>,
        confidence intervals, and effect sizes so you don't bet a launch budget on randomness.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ANOVA tests across the 4 dimensions
    def run_anova(group_col, value_col='Engagement Score'):
        groups = [g[value_col].values for _, g in df.groupby(group_col)]
        if len(groups) < 2 or any(len(g) < 2 for g in groups):
            return None, None, None
        try:
            f_stat, p_val = sci_stats.f_oneway(*groups)
            grand_mean = df[value_col].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            ss_total = ((df[value_col] - grand_mean) ** 2).sum()
            eta_sq = ss_between / ss_total if ss_total > 0 else 0
            return f_stat, p_val, eta_sq
        except Exception:
            return None, None, None

    test_results = []
    for dim in ['Platform', 'Topic', 'Hook Type', 'Content Type']:
        f, p, eta = run_anova(dim)
        if p is not None:
            verdict = "✅ Significant" if p < 0.05 else "❌ Not significant"
            magnitude = (
                "Large effect" if eta and eta > 0.14
                else "Medium effect" if eta and eta > 0.06
                else "Small effect" if eta and eta > 0.01
                else "Negligible"
            )
            test_results.append({
                'Dimension': dim,
                'F-statistic': round(f, 3),
                'p-value': f"{p:.4f}",
                'Eta²': round(eta, 4),
                'Significant?': verdict,
                'Effect Size': magnitude,
            })

    if test_results:
        anova_df = pd.DataFrame(test_results)
        st.markdown("**ANOVA Results — Are differences between groups real?**")
        st.dataframe(anova_df, use_container_width=True, hide_index=True)

        sig_count = sum(1 for r in test_results if "✅" in r['Significant?'])
        large_count = sum(1 for r in test_results if r['Effect Size'] == "Large effect")

        if sig_count == 0:
            verdict_text = (
                "<b>None</b> of the differences in engagement (across platform, topic, hook, or format) "
                "reach statistical significance at p&lt;0.05. <b>Translation:</b> any \"best platform\" "
                "claim from this dataset is statistically indistinguishable from random variation.<br><br>"
                "<b>Strategic implication:</b> Treat the dataset as descriptive, not prescriptive. "
                "Diversify across 2–3 channels and run real A/B tests in-market."
            )
        elif large_count == 0:
            verdict_text = (
                f"<b>{sig_count} of {len(test_results)}</b> dimensions show statistically significant differences, "
                f"but <b>none exceed a 'medium' effect size</b>. The signal exists, but it's modest. "
                "Reasonable strategy: diversify, then double down once live data confirms a pattern."
            )
        else:
            verdict_text = (
                f"<b>{large_count} dimension(s) show large effect sizes.</b> These are real, meaningful differences "
                "that justify concentrated bets. Other dimensions should be diversified."
            )

        st.markdown(f"""
        <div class="verdict-box">
            <div class="verdict-label">📈 What This Actually Means</div>
            <div class="verdict-text">{verdict_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # Confidence intervals
    st.markdown('<div class="section-header">📐 95% Confidence Intervals — Where Real Differences Live</div>', unsafe_allow_html=True)

    def ci_summary(group_col, value_col='Engagement Score'):
        rows = []
        for name, g in df.groupby(group_col):
            n = len(g)
            mean = g[value_col].mean()
            se = g[value_col].std() / np.sqrt(n) if n > 1 else 0
            ci = 1.96 * se
            rows.append({group_col: name, 'Mean': mean, 'CI_Lower': mean - ci, 'CI_Upper': mean + ci, 'N': n})
        return pd.DataFrame(rows).sort_values('Mean', ascending=False)

    rc1, rc2 = st.columns(2)
    with rc1:
        plat_ci = ci_summary('Platform')
        fig_pci = go.Figure()
        fig_pci.add_trace(go.Scatter(
            x=plat_ci['Platform'], y=plat_ci['Mean'],
            error_y=dict(
                type='data', symmetric=False,
                array=plat_ci['CI_Upper'] - plat_ci['Mean'],
                arrayminus=plat_ci['Mean'] - plat_ci['CI_Lower'],
                color='#d4a843', thickness=2, width=8
            ),
            mode='markers', marker=dict(size=14, color='#d4a843', line=dict(color='#f5d58a', width=2)),
            name='Mean ± 95% CI'
        ))
        fig_pci.update_layout(**PLOTLY_BASE)
        fig_pci.update_layout(title_text='Platform: Mean Engagement ± 95% CI')
        st.plotly_chart(fig_pci, use_container_width=True)

    with rc2:
        topic_ci = ci_summary('Topic')
        fig_tci = go.Figure()
        fig_tci.add_trace(go.Scatter(
            x=topic_ci['Topic'], y=topic_ci['Mean'],
            error_y=dict(
                type='data', symmetric=False,
                array=topic_ci['CI_Upper'] - topic_ci['Mean'],
                arrayminus=topic_ci['Mean'] - topic_ci['CI_Lower'],
                color='#d4a843', thickness=2, width=8
            ),
            mode='markers', marker=dict(size=14, color='#d4a843', line=dict(color='#f5d58a', width=2)),
            name='Mean ± 95% CI'
        ))
        fig_tci.update_layout(**PLOTLY_BASE)
        fig_tci.update_layout(title_text='Topic: Mean Engagement ± 95% CI')
        st.plotly_chart(fig_tci, use_container_width=True)

    st.markdown('<div class="section-header">🔗 What Actually Predicts Engagement?</div>', unsafe_allow_html=True)
    numeric_cols = ['Views', 'Likes', 'Comments', 'Engagement Score', 'Viral Score', 'Comment Rate', 'Like Rate']
    available_num = [c for c in numeric_cols if c in df.columns]
    corr_matrix = df[available_num].corr()
    fig_corr = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                         color_continuous_scale='RdYlGn', zmin=-1, zmax=1,
                         title='Correlation Matrix — Which Metrics Move Together?')
    fig_corr.update_layout(**PLOTLY_BASE)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    <div class="verdict-box">
        <div class="verdict-label">🎯 Senior Analyst's Verdict</div>
        <div class="verdict-text">
        1. <b>Don't over-concentrate</b> based on small effect sizes. Diversify across top 2–3 channels.<br>
        2. <b>Run controlled experiments</b> (n &gt; 100 per cell) before scaling.<br>
        3. <b>Track effect size, not just p-values.</b> A statistically significant 2% lift is rarely worth pivoting strategy for.<br>
        4. <b>Treat dataset insights as hypotheses</b> to test in-market, not conclusions to act on.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 8 — CONTENT COACH + HOOK LAB
# ============================================================

with tabs[8]:
    st.markdown('<div class="section-header">📝 Content Coach + Hook Lab — AI Production Tools</div>', unsafe_allow_html=True)

    coach_tab, hook_tab = st.tabs(["🎙️ Content Coach", "💡 Hook Lab"])

    with coach_tab:
        st.markdown("""
        <div class="verdict-box">
            <div class="verdict-label">🎙️ Live Script & Caption Analyzer</div>
            <div class="verdict-text">
            Paste any script, caption, or video transcript. Claude scores it on 5 dimensions
            (Hook Strength · Curiosity Gap · Emotional Tension · CTA Quality · Pacing) and rewrites the weak parts.
            This is what a Head of Content does for every script before it ships.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not api_key:
            st.warning("⚠️ Enter your Anthropic API key in the sidebar to use the Content Coach.")
        else:
            sample_script = st.text_area(
                "Paste your script, caption, or transcript:",
                height=180,
                placeholder="e.g. Today I asked Napoleon why leaders fail. The answer surprised me..."
            )

            coach_btn = st.button("🎙️ Analyze Script", key="coach_btn", type="primary")

            if coach_btn and sample_script.strip():
                with st.spinner("Coach is analyzing..."):
                    try:
                        import anthropic as anthropic_mod
                        client = anthropic_mod.Anthropic(api_key=api_key)

                        coach_prompt = f"""You are an elite content coach for HistOracle, an AI history platform.
Analyze this script/caption ruthlessly. Be specific, no fluff.

SCRIPT:
\"\"\"{sample_script}\"\"\"

Return a JSON object with EXACTLY this structure (no other text):
{{
  "hook_strength": <number 1-10>,
  "curiosity_gap": <number 1-10>,
  "emotional_tension": <number 1-10>,
  "cta_quality": <number 1-10>,
  "pacing": <number 1-10>,
  "overall_score": <number 1-10>,
  "verdict": "<one-line ruthless assessment>",
  "biggest_weakness": "<the single biggest problem>",
  "rewrite_suggestion": "<a stronger version of the first 3 seconds>",
  "what_works": "<one strength to keep>"
}}"""

                        response = client.messages.create(
                            model="claude-opus-4-5",
                            max_tokens=1500,
                            messages=[{"role": "user", "content": coach_prompt}]
                        )
                        raw = response.content[0].text.strip()

                        # Extract JSON
                        start_idx = raw.find('{')
                        end_idx = raw.rfind('}') + 1
                        if start_idx >= 0 and end_idx > start_idx:
                            scores = json.loads(raw[start_idx:end_idx])

                            # Score visualization
                            score_dims = ['hook_strength', 'curiosity_gap', 'emotional_tension', 'cta_quality', 'pacing']
                            score_labels = ['Hook Strength', 'Curiosity Gap', 'Emotional Tension', 'CTA Quality', 'Pacing']
                            score_values = [scores.get(k, 0) for k in score_dims]

                            fig_scores = go.Figure(go.Scatterpolar(
                                r=score_values + [score_values[0]],
                                theta=score_labels + [score_labels[0]],
                                fill='toself',
                                fillcolor='rgba(212,168,67,0.3)',
                                line=dict(color='#d4a843', width=2),
                                marker=dict(size=8, color='#f5d58a')
                            ))
                            fig_scores.update_layout(
                                polar=dict(
                                    bgcolor='rgba(0,0,0,0.2)',
                                    radialaxis=dict(visible=True, range=[0, 10], color='#7a6030'),
                                    angularaxis=dict(color='#d4a843')
                                ),
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#b4a070'),
                                showlegend=False,
                                height=400,
                                title=dict(text=f"Overall Score: {scores.get('overall_score', 0)}/10",
                                           font=dict(color='#d4a843', size=18))
                            )
                            st.plotly_chart(fig_scores, use_container_width=True)

                            # Insights
                            st.markdown(f"""
                            <div class="verdict-box">
                                <div class="verdict-label">⚔️ COACH VERDICT</div>
                                <div class="verdict-text"><b>{scores.get('verdict', 'No verdict.')}</b></div>
                            </div>
                            """, unsafe_allow_html=True)

                            ic1, ic2 = st.columns(2)
                            with ic1:
                                st.markdown(f"""
                                <div class="insight-card">
                                    <div class="insight-title">🔴 Biggest Weakness</div>
                                    <div class="insight-body">{scores.get('biggest_weakness', '')}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with ic2:
                                st.markdown(f"""
                                <div class="insight-card">
                                    <div class="insight-title">🟢 What Works</div>
                                    <div class="insight-body">{scores.get('what_works', '')}</div>
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown(f"""
                            <div class="insight-card" style="margin-top:14px;border-left:3px solid #d4a843;">
                                <div class="insight-title">✏️ Stronger First 3 Seconds</div>
                                <div class="insight-body" style="font-style:italic;color:#f5d58a;">
                                "{scores.get('rewrite_suggestion', '')}"
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Could not parse scores. Raw output below.")
                            st.code(raw)
                    except Exception as e:
                        st.error(f"Coach error: {str(e)}")

    with hook_tab:
        st.markdown("""
        <div class="verdict-box">
            <div class="verdict-label">💡 AI Hook Generator</div>
            <div class="verdict-text">
            Type any topic. Claude generates 10 hook variants across all 5 hook types (Mystery, Question,
            Controversial, Emotional, You-didn't-know). This is the brainstorming engine — pick winners,
            test them on TikTok, double down on what works.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not api_key:
            st.warning("⚠️ Enter your Anthropic API key in the sidebar to use Hook Lab.")
        else:
            hook_topic = st.text_input(
                "Topic to generate hooks for:",
                placeholder="e.g. Roman emperors, World War II, Cleopatra, ancient mysteries"
            )

            hook_btn = st.button("💡 Generate 10 Hooks", key="hook_btn", type="primary")

            if hook_btn and hook_topic.strip():
                with st.spinner("Generating viral hooks..."):
                    try:
                        import anthropic as anthropic_mod
                        client = anthropic_mod.Anthropic(api_key=api_key)

                        hook_prompt = f"""You are HistOracle's lead creative strategist. Generate 10 viral video hooks
about "{hook_topic}" — 2 each of these styles: Mystery, Question, Controversial, Emotional, You-didn't-know.

Each hook should:
- Be under 12 words
- Trigger an immediate stop-scroll
- Tease HistOracle's Teleport (talking to historical figures) without naming the product

Return ONLY a valid JSON array (no other text):
[
  {{"style": "Mystery", "hook": "...", "viral_potential": <1-10>, "why": "<one-line>"}},
  ...
]"""

                        response = client.messages.create(
                            model="claude-opus-4-5",
                            max_tokens=2000,
                            messages=[{"role": "user", "content": hook_prompt}]
                        )
                        raw = response.content[0].text.strip()
                        start_idx = raw.find('[')
                        end_idx = raw.rfind(']') + 1
                        if start_idx >= 0 and end_idx > start_idx:
                            hooks_data = json.loads(raw[start_idx:end_idx])
                            hooks_df = pd.DataFrame(hooks_data).sort_values('viral_potential', ascending=False)
                            hooks_df.index = range(1, len(hooks_df) + 1)
                            st.dataframe(hooks_df, use_container_width=True)

                            top_hook = hooks_df.iloc[0]
                            st.markdown(f"""
                            <div class="verdict-box">
                                <div class="verdict-label">🏆 Top Pick — Test This First</div>
                                <div class="verdict-text">
                                <b style="color:#f5d58a;font-size:18px;">"{top_hook['hook']}"</b><br><br>
                                Style: {top_hook['style']} · Viral Score: {top_hook['viral_potential']}/10<br>
                                <em>{top_hook['why']}</em>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Could not parse hooks.")
                            st.code(raw)
                    except Exception as e:
                        st.error(f"Hook Lab error: {str(e)}")


# ============================================================
# TAB 10 — COMPETITIVE BATTLEFIELD
# ============================================================

with tabs[10]:
    st.markdown('<div class="section-header">⚡ Competitive Battlefield — Why HistOracle Doesn\'t Get Eaten</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="verdict-box">
        <div class="verdict-label">⚠️ The Question Every Investor Asks First</div>
        <div class="verdict-text">
        <b>"Why won't character.ai eat your lunch?"</b><br><br>
        Character.ai has 25M+ MAU and already lets users talk to historical figures for free.
        ChatGPT does roleplay better than most dedicated apps. Khan Academy AI has institutional trust.
        If HistOracle has no answer to these threats, the marketing strategy is irrelevant.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🥊 Competitive Comparison Matrix</div>', unsafe_allow_html=True)

    competitors = pd.DataFrame([
        {'Product': '🏛️ HistOracle', 'Specialization': 'History only', 'Conversation w/ Figures': '✅ Native (Teleport)',
         'Educational Rigor': '🟢 High', 'Source Citation': '✅ Planned', 'Price': '$9.99 / $49 edu',
         'MAU': 'Pre-launch', 'Threat Level': '—'},
        {'Product': '🎭 character.ai', 'Specialization': 'General persona chat', 'Conversation w/ Figures': '✅ User-generated',
         'Educational Rigor': '🔴 Low', 'Source Citation': '❌ No', 'Price': 'Free + $9.99 Pro',
         'MAU': '25M+', 'Threat Level': '🚨 EXTREME'},
        {'Product': '🤖 ChatGPT', 'Specialization': 'General-purpose AI', 'Conversation w/ Figures': '✅ Via roleplay',
         'Educational Rigor': '🟡 Variable', 'Source Citation': '🟡 With browsing', 'Price': 'Free + $20 Plus',
         'MAU': '600M+', 'Threat Level': '🚨 EXTREME'},
        {'Product': '📚 Khan Academy', 'Specialization': 'K–12 education', 'Conversation w/ Figures': '🟡 Limited',
         'Educational Rigor': '🟢 Very high', 'Source Citation': '✅ Yes', 'Price': 'Free + $4/mo',
         'MAU': '40M+', 'Threat Level': '⚠️ HIGH (B2B)'},
        {'Product': '🎓 Quizlet', 'Specialization': 'Study tools', 'Conversation w/ Figures': '❌ No',
         'Educational Rigor': '🟡 Medium', 'Source Citation': '🟡 Sometimes', 'Price': 'Free + $7.99',
         'MAU': '60M+', 'Threat Level': '⚠️ MEDIUM'},
        {'Product': '🗿 Hello History', 'Specialization': 'History persona chat', 'Conversation w/ Figures': '✅ Native',
         'Educational Rigor': '🟡 Medium', 'Source Citation': '❌ No', 'Price': '$3.99/wk',
         'MAU': '~500K', 'Threat Level': '⚠️ DIRECT'},
    ])
    st.dataframe(competitors, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">🗺️ Strategic Positioning Map</div>', unsafe_allow_html=True)

    positioning = pd.DataFrame([
        {'Product': 'HistOracle',     'Educational_Rigor': 9.0, 'Conversational_Interactivity': 9.0, 'MAU_Size': 5},
        {'Product': 'character.ai',   'Educational_Rigor': 3.0, 'Conversational_Interactivity': 9.5, 'MAU_Size': 80},
        {'Product': 'ChatGPT',        'Educational_Rigor': 6.5, 'Conversational_Interactivity': 8.0, 'MAU_Size': 100},
        {'Product': 'Khan Academy',   'Educational_Rigor': 9.5, 'Conversational_Interactivity': 5.0, 'MAU_Size': 60},
        {'Product': 'Quizlet',        'Educational_Rigor': 6.5, 'Conversational_Interactivity': 4.5, 'MAU_Size': 70},
        {'Product': 'Hello History',  'Educational_Rigor': 5.5, 'Conversational_Interactivity': 7.5, 'MAU_Size': 8},
    ])

    fig_pos = px.scatter(
        positioning, x='Educational_Rigor', y='Conversational_Interactivity',
        text='Product', size='MAU_Size', size_max=60, color='Product',
        title='Positioning Map — Educational Rigor vs Conversational Interactivity (bubble = MAU)',
        color_discrete_sequence=['#d4a843', '#ef4444', '#f59e0b', '#22c55e', '#8b5cf6', '#06b6d4']
    )
    fig_pos.update_traces(textposition='top center', textfont=dict(color='#f5d58a', size=12))
    fig_pos.update_layout(**PLOTLY_BASE)
    fig_pos.update_layout(
        xaxis=dict(range=[0, 11], title='Educational Rigor →'),
        yaxis=dict(range=[0, 11], title='Conversational Interactivity →'),
    )
    fig_pos.add_shape(type="rect", x0=7.5, y0=7.5, x1=11, y1=11,
                      line=dict(color="rgba(212,168,67,0.4)", width=2, dash="dash"),
                      fillcolor="rgba(212,168,67,0.05)")
    fig_pos.add_annotation(x=9.25, y=10.5, text="🏆 HISTORACLE WHITESPACE",
                           showarrow=False, font=dict(color="#d4a843", size=11, family="Cinzel"))
    st.plotly_chart(fig_pos, use_container_width=True)

    st.markdown('<div class="section-header">🛡️ The Three-Part Moat</div>', unsafe_allow_html=True)

    moat_cols = st.columns(3)
    moats = [
        {"icon": "📚", "title": "1. Curated Historical Corpus",
         "body": "Not 'ChatGPT pretending to be Lincoln.' A vetted, source-cited corpus of primary historical documents fine-tuned per figure. Hallucination rate measurably lower than general LLMs. This is what makes teachers trust it and students cite it."},
        {"icon": "🏫", "title": "2. Education Distribution",
         "body": "B2B school partnerships create switching costs consumer apps can't match. Once a curriculum is built around HistOracle Teleport, switching to character.ai means rebuilding lesson plans. character.ai cannot pivot fast enough into this channel without alienating its core user base."},
        {"icon": "🎯", "title": "3. Specialization Brand",
         "body": "ChatGPT does everything badly enough. character.ai is a teen entertainment product. HistOracle owns the category 'history specifically' the way Strava owns 'running' against general fitness apps. Specialization beats generalization in trust-driven verticals."}
    ]
    for col, m in zip(moat_cols, moats):
        with col:
            st.markdown(f"""
            <div class="insight-card">
                <div style="font-size:32px;margin-bottom:8px;">{m['icon']}</div>
                <div class="insight-title">{m['title']}</div>
                <div class="insight-body">{m['body']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">⚠️ Threat Response Playbook</div>', unsafe_allow_html=True)
    threats = pd.DataFrame([
        {'Threat': 'character.ai launches verified history bots', 'Likelihood': 'Medium', 'Impact': 'High',
         'Response': 'Lean harder into education partnerships. Consumer war is theirs; institutional war is ours.'},
        {'Threat': 'OpenAI GPTs marketplace ships history personas', 'Likelihood': 'High', 'Impact': 'Medium',
         'Response': 'Differentiate on UX (Teleport interface), source citation, curriculum integration. Price below ChatGPT Plus.'},
        {'Threat': 'Khan Academy adds figure conversations to Khanmigo', 'Likelihood': 'Medium', 'Impact': 'Very High',
         'Response': 'Move fast on B2B education channel before they do. Lock in district contracts now.'},
        {'Threat': 'Hello History improves UX significantly', 'Likelihood': 'High', 'Impact': 'Low',
         'Response': 'They are not solving the rigor problem. We compete on accuracy, not novelty.'},
    ])
    st.dataframe(threats, use_container_width=True, hide_index=True)


# ============================================================
# TAB 11 — BUSINESS MODEL & UNIT ECONOMICS
# ============================================================

with tabs[11]:
    st.markdown('<div class="section-header">💰 Business Model & Unit Economics</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="verdict-box">
        <div class="verdict-label">📈 Why This Tab Exists</div>
        <div class="verdict-text">
        Every brilliant marketing strategy dies the moment someone asks <b>"what are the unit economics?"</b>
        Below are <b>interactive sliders</b> — change any assumption and LTV, payback, and ARR projections
        update live. This is the slide that turns a marketing pitch into a business case.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🎚️ Pricing & Acquisition Assumptions</div>', unsafe_allow_html=True)

    sl1, sl2, sl3 = st.columns(3)
    with sl1:
        st.markdown("**Consumer Tier**")
        consumer_price = st.slider("Monthly price ($)", 4.99, 19.99, 9.99, 1.0, key="cprice")
        consumer_retention = st.slider("Avg retention (months)", 1, 36, 14, key="cret")
        consumer_margin = st.slider("Gross margin (%)", 30, 90, 65, key="cmar")

    with sl2:
        st.markdown("**Education Tier**")
        edu_price = st.slider("Per-seat monthly ($)", 19, 99, 49, 5, key="eprice")
        edu_retention = st.slider("Avg retention (months)", 6, 60, 36, key="eret")
        edu_margin = st.slider("Gross margin (%)", 40, 90, 75, key="emar")

    with sl3:
        st.markdown("**Acquisition Costs**")
        cac_organic = st.slider("Organic CAC ($)", 0, 50, 12, key="caco")
        cac_paid = st.slider("Paid CAC ($)", 10, 200, 45, key="cacp")
        cac_edu = st.slider("Education CAC ($)", 50, 1000, 250, key="cacedu")

    consumer_ltv = consumer_price * consumer_retention * (consumer_margin / 100)
    edu_ltv = edu_price * edu_retention * (edu_margin / 100)
    consumer_ltv_cac_paid = consumer_ltv / cac_paid if cac_paid > 0 else float('inf')
    edu_ltv_cac = edu_ltv / cac_edu if cac_edu > 0 else float('inf')
    consumer_payback_paid = cac_paid / (consumer_price * (consumer_margin / 100)) if consumer_margin > 0 else float('inf')
    edu_payback = cac_edu / (edu_price * (edu_margin / 100)) if edu_margin > 0 else float('inf')

    st.markdown('<div class="section-header">📊 Live Unit Economics</div>', unsafe_allow_html=True)

    ue1, ue2, ue3, ue4 = st.columns(4)
    with ue1: st.metric("Consumer LTV", f"${consumer_ltv:.0f}", f"{consumer_retention} mo retention")
    with ue2: st.metric("Education LTV", f"${edu_ltv:.0f}", f"{edu_retention} mo retention")
    with ue3: st.metric("LTV/CAC (paid)", f"{consumer_ltv_cac_paid:.1f}x", "Healthy if ≥3x")
    with ue4: st.metric("Edu LTV/CAC", f"{edu_ltv_cac:.1f}x", "Healthy if ≥5x")

    pp1, pp2, pp3, pp4 = st.columns(4)
    with pp1: st.metric("Consumer Payback", f"{consumer_payback_paid:.1f} mo", "Healthy if <12 mo")
    with pp2: st.metric("Edu Payback", f"{edu_payback:.1f} mo", "Healthy if <18 mo")
    with pp3: st.metric("Consumer Margin", f"${consumer_price * (consumer_margin/100):.2f}/mo", "Per-user GP")
    with pp4: st.metric("Edu Margin", f"${edu_price * (edu_margin/100):.2f}/mo", "Per-seat GP")

    health_msgs = []
    if consumer_ltv_cac_paid < 3:
        health_msgs.append("🔴 Consumer LTV/CAC on paid acquisition is below 3x — paid growth would burn cash.")
    elif consumer_ltv_cac_paid < 5:
        health_msgs.append("🟡 Consumer LTV/CAC on paid acquisition is okay (3–5x). Tight but workable.")
    else:
        health_msgs.append("🟢 Consumer LTV/CAC is healthy on paid acquisition.")

    if edu_ltv_cac < 5:
        health_msgs.append("🔴 Education LTV/CAC is below 5x — B2B sales motion would not be efficient.")
    else:
        health_msgs.append("🟢 Education LTV/CAC is strong. B2B growth is viable.")

    if consumer_payback_paid > 18:
        health_msgs.append("🔴 Payback period over 18 months — too long for venture-scale growth.")

    health_html = "<br>".join(health_msgs)
    st.markdown(f"""
    <div class="verdict-box">
        <div class="verdict-label">🩺 Health Check</div>
        <div class="verdict-text">{health_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🚀 Path to $1M ARR</div>', unsafe_allow_html=True)

    rs1, rs2, rs3 = st.columns(3)
    with rs1:
        consumer_users_target = st.number_input("Target paid consumers", 100, 1000000, 5000, step=500)
    with rs2:
        edu_seats_target = st.number_input("Target education seats", 10, 100000, 1000, step=100)
    with rs3:
        free_to_paid_conv = st.slider("Free → paid conversion (%)", 1, 30, 8) / 100

    consumer_arr = consumer_users_target * consumer_price * 12
    edu_arr = edu_seats_target * edu_price * 12
    total_arr = consumer_arr + edu_arr
    free_users_needed = consumer_users_target / free_to_paid_conv if free_to_paid_conv > 0 else float('inf')

    months = list(range(1, 25))
    consumer_growth_curve = [consumer_users_target * (1 - np.exp(-m/8)) for m in months]
    edu_growth_curve = [edu_seats_target * (1 - np.exp(-m/12)) for m in months]
    monthly_consumer_rev = [u * consumer_price for u in consumer_growth_curve]
    monthly_edu_rev = [s * edu_price for s in edu_growth_curve]
    monthly_total_rev = [c + e for c, e in zip(monthly_consumer_rev, monthly_edu_rev)]
    arr_curve = [r * 12 for r in monthly_total_rev]

    fig_arr = go.Figure()
    fig_arr.add_trace(go.Scatter(x=months, y=arr_curve, mode='lines',
                                 line=dict(color='#d4a843', width=3),
                                 name='Total ARR', fill='tozeroy',
                                 fillcolor='rgba(212,168,67,0.15)'))
    fig_arr.add_hline(y=1_000_000, line_dash="dash", line_color="#22c55e",
                      annotation_text="$1M ARR target", annotation_position="top right",
                      annotation_font=dict(color="#22c55e"))
    fig_arr.update_layout(**PLOTLY_BASE)
    fig_arr.update_layout(title_text='ARR Projection — Months from Launch',
                          xaxis_title='Month', yaxis_title='ARR ($)')
    st.plotly_chart(fig_arr, use_container_width=True)

    arr1, arr2, arr3, arr4 = st.columns(4)
    with arr1: st.metric("Projected ARR", f"${total_arr/1000:.0f}K", f"${total_arr:,.0f}")
    with arr2: st.metric("Consumer ARR", f"${consumer_arr/1000:.0f}K", f"{consumer_users_target:,} users")
    with arr3: st.metric("Education ARR", f"${edu_arr/1000:.0f}K", f"{edu_seats_target:,} seats")
    with arr4: st.metric("Free Users Needed", f"{free_users_needed/1000:.0f}K", f"@ {free_to_paid_conv*100:.0f}% conv")

    st.markdown('<div class="section-header">🎯 Sensitivity Analysis</div>', unsafe_allow_html=True)
    sensitivity_data = []
    for retention in [6, 12, 18, 24, 36]:
        for margin in [50, 65, 80]:
            ltv = consumer_price * retention * (margin / 100)
            ratio = ltv / cac_paid if cac_paid > 0 else 0
            sensitivity_data.append({'Retention (mo)': retention, 'Margin (%)': margin,
                                     'LTV ($)': round(ltv, 0), 'LTV/CAC': round(ratio, 1)})
    sens_df = pd.DataFrame(sensitivity_data)
    sens_pivot = sens_df.pivot(index='Retention (mo)', columns='Margin (%)', values='LTV/CAC')
    fig_sens = px.imshow(sens_pivot, text_auto='.1f', aspect='auto',
                         color_continuous_scale=['#ef4444', '#f59e0b', '#22c55e'],
                         title='LTV/CAC Sensitivity to Retention & Margin')
    fig_sens.update_layout(**PLOTLY_BASE)
    st.plotly_chart(fig_sens, use_container_width=True)

    st.markdown("""
    <div class="verdict-box">
        <div class="verdict-label">💡 Key Strategic Implications</div>
        <div class="verdict-text">
        <b>1.</b> The B2B education channel has dramatically better unit economics than consumer.
        A single school district contract is worth thousands of consumer users.<br><br>
        <b>2.</b> Free → paid conversion is the most sensitive variable. A 5% lift here changes everything.<br><br>
        <b>3.</b> Paid acquisition is only viable once organic CAC channels saturate.
        Phase 1 should be 100% organic.<br><br>
        <b>4.</b> The model breaks if consumer retention drops below 8 months. Build retention-first features
        (streaks, conversation history, achievement badges) into the launch product, not v2.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 14 — EXPORT REPORT
# ============================================================

with tabs[14]:
    st.markdown('<div class="section-header">📥 Executive Report — Export & Share</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="verdict-box">
        <div class="verdict-label">📊 Boardroom-Ready Deliverable</div>
        <div class="verdict-text">
        Senior people don't open dashboards — they read PDFs and forward emails.
        Below: download a clean Markdown executive brief, the full filtered dataset as CSV, or copy
        a Slack-ready summary to share instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Build executive markdown report
    report_date = datetime.today().strftime("%B %d, %Y")
    report_md = f"""# HistOracle Executive Brief — {report_date}

## Summary

Analysis of **{total_posts:,} posts** across **{len(sel_p)} platforms** identified the
following strategic priorities for HistOracle's social media launch.

## Top-Line Metrics

| Metric | Value |
|---|---|
| Total Views Analyzed | {fmt(total_views)} |
| Total Engagement | {fmt(total_eng)} |
| Average Engagement / Post | {fmt(avg_eng)} |
| Best Performing Platform | {best_platform} |
| Best Performing Topic | {best_topic} |
| Best Performing Hook | {best_hook} |
| Best Performing Format | {best_format} |
| Viral Predictor Accuracy | {clf_accuracy:.1%} |

## Strategic Recommendations

1. **Lead with {best_platform}** as primary distribution channel.
2. **{best_topic}** content should anchor the editorial calendar.
3. Use **{best_hook}** hooks as the default creative weapon.
4. Default to **{best_format}** format for new posts.
5. Position HistOracle as **interactive AI history**, not another education app.
6. Winning brand message: *"Talk to the past like it is alive."*

## Statistical Caveat

Differences between platforms are modest. Effect sizes should be validated through
controlled in-market A/B testing before concentrating budget. Treat dataset insights
as hypotheses, not conclusions.

## Competitive Context

- **character.ai** (25M+ MAU) is the most direct threat — it already supports historical persona chat.
- **HistOracle's moat** must be: curated educational corpus, B2B school distribution, specialization brand.

## Unit Economics Assumptions (Default Model)

| Tier | Price | Retention | Margin | LTV |
|---|---|---|---|---|
| Consumer | $9.99/mo | 14 months | 65% | $90.91 |
| Education | $49/seat/mo | 36 months | 75% | $1,323.00 |

**Path to $1M ARR:** ~5,000 paid consumers + ~1,000 education seats.

## 30-Day Content Calendar

The dashboard includes an auto-generated 30-day calendar with platform, topic, hook,
and format mapped to each day, optimized for the Viral Catalyst content profile.

---

*Generated from HistOracle Elite War Room dashboard*
"""

    st.markdown('<div class="section-header">📄 Executive Brief Preview</div>', unsafe_allow_html=True)
    with st.expander("📖 View Full Brief", expanded=True):
        st.markdown(report_md)

    # Download buttons
    st.markdown('<div class="section-header">📥 Downloads</div>', unsafe_allow_html=True)

    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.download_button(
            label="📄 Download Executive Brief (.md)",
            data=report_md,
            file_name=f"HistOracle_Executive_Brief_{datetime.today().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            use_container_width=True
        )

    with dl2:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Filtered Dataset (.csv)",
            data=csv_data,
            file_name=f"HistOracle_Data_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with dl3:
        cal_csv = cal_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📅 Download 30-Day Calendar (.csv)",
            data=cal_csv,
            file_name=f"HistOracle_Calendar_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Slack/email summary
    st.markdown('<div class="section-header">💬 Slack / Email Summary</div>', unsafe_allow_html=True)
    slack_summary = f"""*HistOracle Brief — {report_date}*

📊 Analyzed {total_posts:,} posts | {fmt(total_views)} views | {fmt(total_eng)} engagement

🎯 *Top Findings:*
• Best platform: *{best_platform}*
• Best topic: *{best_topic}*
• Best hook: *{best_hook}*
• Best format: *{best_format}*

🚨 *Key Decisions Needed:*
• Approve {best_platform}-first distribution strategy
• Validate persona research with real interviews
• Lock in B2B education pilot before character.ai pivots

⚔️ Run the Viral Predictor before publishing. Use the Content Coach to score scripts.

Full dashboard: <link>
"""
    st.code(slack_summary, language="text")
    st.caption("👆 Copy this directly into Slack, email, or any messaging tool.")


# ============================================================
# RAW DATA + FOOTER
# ============================================================

with st.expander("📊 View Raw Dataset"):
    st.dataframe(df, use_container_width=True)

st.markdown("""
<div class="footer">
    HistOracle Elite War Room · Teleport Experience · Viral Predictor · Content Agent · AI Strategy · Built for Execution
</div>
""", unsafe_allow_html=True)