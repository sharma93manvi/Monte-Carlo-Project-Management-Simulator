"""
Monte Carlo Project Simulator
A flexible, visually rich Streamlit app for project management simulation.
Supports multiple probability distributions, CSV upload, sensitivity analysis,
network graph, Gantt chart, PERT comparison, buffer analysis, and more.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from collections import defaultdict
from scipy import stats as sp_stats
import io

# --- Page Config ---
st.set_page_config(
    page_title="Monte Carlo Project Simulator",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
    text-align: center; color: white; box-shadow: 0 8px 32px rgba(0,0,0,0.18);
}
.main-header h1 { font-size: 2.4rem; margin: 0; letter-spacing: -0.5px; }
.main-header p { font-size: 1.05rem; opacity: 0.85; margin-top: 0.5rem; }
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 14px; padding: 1.4rem 1.2rem; text-align: center;
    color: white; box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    border: 1px solid rgba(255,255,255,0.08);
}
.metric-card .metric-value { font-size: 2rem; font-weight: 700; color: #4fc3f7; }
.metric-card .metric-label {
    font-size: 0.85rem; opacity: 0.7; margin-top: 0.3rem;
    text-transform: uppercase; letter-spacing: 1px;
}
.section-header {
    background: linear-gradient(90deg, #4fc3f7, #0288d1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 1.5rem; font-weight: 700; margin: 1.5rem 0 0.8rem 0;
}
.insight-box {
    background: linear-gradient(135deg, #1b1b2f, #162447);
    border-left: 4px solid #4fc3f7; border-radius: 8px;
    padding: 1rem 1.2rem; color: #e0e0e0; margin: 0.8rem 0; font-size: 0.95rem;
}
.dist-info {
    background: rgba(79, 195, 247, 0.08); border: 1px solid rgba(79, 195, 247, 0.2);
    border-radius: 10px; padding: 1rem; margin: 0.5rem 0; font-size: 0.9rem;
}
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
div.stButton > button[kind="primary"] {
    font-size: 1.25rem; padding: 0.85rem 2rem; letter-spacing: 0.5px;
}
/* Fix file uploader button text overlap */
section[data-testid="stFileUploader"] label {
    font-size: 1rem !important;
    margin-bottom: 0.5rem !important;
}
section[data-testid="stFileUploader"] button {
    margin-top: 0.25rem !important;
}
section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] {
    padding: 1rem !important;
}
section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] span {
    white-space: normal !important;
    line-height: 1.4 !important;
}

/* ---- Responsive / Mobile ---- */
</style>
""", unsafe_allow_html=True)

# Responsive CSS in separate block to avoid truncation
st.markdown("""
<style>
@media (max-width: 768px) {
    .main-header {
        padding: 1.5rem 1rem;
        border-radius: 10px;
    }
    .main-header h1 {
        font-size: 1.5rem;
    }
    .main-header p {
        font-size: 0.85rem;
    }
    .metric-card {
        padding: 1rem 0.8rem;
        margin-bottom: 0.6rem;
    }
    .metric-card .metric-value {
        font-size: 1.4rem;
    }
    .metric-card .metric-label {
        font-size: 0.7rem;
    }
    .section-header {
        font-size: 1.15rem;
    }
    .insight-box {
        padding: 0.75rem 0.9rem;
        font-size: 0.85rem;
    }
    /* Stack metric columns vertically on mobile */
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        min-width: 48% !important;
        flex: 1 1 48% !important;
        margin-bottom: 0.5rem;
    }
    /* Make charts scroll horizontally if needed */
    div[data-testid="stImage"], .stPlotlyChart, div[data-testid="stPyplot"] {
        overflow-x: auto !important;
    }
    /* Smaller button text on mobile */
    div.stButton > button[kind="primary"] {
        font-size: 1rem;
        padding: 0.7rem 1rem;
    }
    /* Data editor / tables scroll */
    div[data-testid="stDataFrame"] {
        overflow-x: auto !important;
    }
}

@media (max-width: 480px) {
    .main-header h1 {
        font-size: 1.25rem;
    }
    .metric-card .metric-value {
        font-size: 1.2rem;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>Monte Carlo Project Simulator</h1>
    <p>Estimate project duration under uncertainty using simulation &mdash; works for any project network</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## Simulation Settings")
num_simulations = st.sidebar.slider("Number of Simulations", 1000, 100000, 10000, step=1000)
seed = st.sidebar.number_input("Random Seed (0 = random)", min_value=0, max_value=99999, value=42)
service_level = st.sidebar.slider("Service Level (%)", 50, 99, 95)

st.sidebar.markdown("---")
st.sidebar.markdown("## Distribution Type")
dist_type = st.sidebar.selectbox(
    "Choose probability distribution",
    ["Uniform", "Triangular", "PERT (Beta)", "Normal (truncated)", "Lognormal"],
    help="Triangular uses min/mode/max. PERT gives more weight to the mode. "
         "Uniform treats all values equally. Normal uses avg as mean. "
         "Lognormal is right-skewed, good for approvals/procurement."
)
DIST_DESCRIPTIONS = {
    "Triangular": "Uses min, mode (most likely), and max. Good default when you have three-point estimates.",
    "PERT (Beta)": "A Beta distribution weighted toward the mode (lambda=4). Gives less probability to extremes than Triangular. Widely used in project management.",
    "Uniform": "Every value between min and max is equally likely. Use when you have no idea about the most likely value.",
    "Normal (truncated)": "Bell curve centered on the average, truncated at min/max. Std dev = (max-min)/6. Use when durations cluster tightly around the mean.",
    "Lognormal": "Right-skewed distribution. Mean approx mode; spread from (max-min)/6. Good for activities like approvals or procurement that can have long tails.",
}
st.sidebar.markdown(f'<div class="dist-info"><b>{dist_type}</b><br>{DIST_DESCRIPTIONS[dist_type]}</div>',
                    unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit")

# ============================================================
# DEFAULT DATA
# ============================================================
DEFAULT_DATA = {
    'Label':        ['A','B','C','D','E','F','G'],
    'Activity':     ['Design','Build prototype','Evaluate equipment','Test prototype',
                     'Write equipment report','Write methods report','Write final report'],
    'Predecessors': ['','A','A','B','C,D','C,D','E,F'],
    'Min Duration': [16,3,5,2,4,6,1],
    'Avg Duration': [21,6,7,3,6,8,2],
    'Max Duration': [26,9,9,4,8,10,3],
}

# ============================================================
# DATA INPUT
# ============================================================
st.markdown('<div class="section-header">Project Activities</div>', unsafe_allow_html=True)
tab_edit, tab_upload, tab_template = st.tabs(["Edit Table", "Upload CSV/Excel", "Download Template"])

with tab_edit:
    st.markdown("Edit the table below directly. Add or remove rows for any project. "
                "Use comma-separated labels for multiple predecessors (e.g., `C,D`).")
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame(DEFAULT_DATA)
    edited_df = st.data_editor(
        st.session_state.df, num_rows="dynamic", use_container_width=True, key="activity_table",
        column_config={
            "Label": st.column_config.TextColumn("Label", width="small"),
            "Activity": st.column_config.TextColumn("Activity Name", width="medium"),
            "Predecessors": st.column_config.TextColumn("Predecessors", width="small",
                                                         help="Comma-separated predecessor labels"),
            "Min Duration": st.column_config.NumberColumn("Min", min_value=0.0, format="%.1f"),
            "Avg Duration": st.column_config.NumberColumn("Avg (Mode)", min_value=0.0, format="%.1f"),
            "Max Duration": st.column_config.NumberColumn("Max", min_value=0.0, format="%.1f"),
        }
    )

with tab_upload:
    st.markdown("Upload a CSV or Excel file with columns: `Label`, `Activity`, `Predecessors`, "
                "`Min Duration`, `Avg Duration`, `Max Duration`")
    st.markdown("")
    st.markdown("**Select your project file (.csv, .xlsx, .xls):**")
    uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"],
                                     label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                upload_df = pd.read_csv(uploaded_file)
            else:
                upload_df = pd.read_excel(uploaded_file)
            required_cols = {'Label', 'Activity', 'Predecessors', 'Min Duration', 'Avg Duration', 'Max Duration'}
            if required_cols.issubset(set(upload_df.columns)):
                upload_df['Predecessors'] = upload_df['Predecessors'].fillna('')
                st.session_state.df = upload_df[list(required_cols)].copy()
                edited_df = st.session_state.df
                st.success(f"Loaded {len(upload_df)} activities from {uploaded_file.name}")
                st.dataframe(upload_df, use_container_width=True, hide_index=True)
            else:
                missing = required_cols - set(upload_df.columns)
                st.error(f"Missing columns: {', '.join(missing)}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab_template:
    st.markdown("Download a template file pre-filled with the Computer Design project data.")
    template_df = pd.DataFrame(DEFAULT_DATA)
    csv_buffer = template_df.to_csv(index=False)
    st.download_button(label="Download CSV Template", data=csv_buffer,
                       file_name="project_template.csv", mime="text/csv", use_container_width=True)
    st.dataframe(template_df, use_container_width=True, hide_index=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_data(df):
    """Comprehensive validation with clear error messages."""
    errors = []
    if df is None or len(df) == 0:
        return ["The activity table is empty. Please add at least one activity."]
    required_cols = {'Label', 'Activity', 'Predecessors', 'Min Duration', 'Avg Duration', 'Max Duration'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        return [f"Missing required columns: {', '.join(missing_cols)}"]

    labels = []
    for idx, row in df.iterrows():
        label = str(row.get('Label', '')).strip()
        if not label or label == 'nan':
            errors.append(f"Row {idx + 1}: Activity label is empty.")
            continue
        labels.append(label)
        act_name = str(row.get('Activity', '')).strip()
        if not act_name or act_name == 'nan':
            errors.append(f"Activity '{label}': Activity name is empty.")
        for col in ['Min Duration', 'Avg Duration', 'Max Duration']:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                errors.append(f"Activity '{label}': {col} is missing.")
        try:
            lo = float(row['Min Duration'])
            avg = float(row['Avg Duration'])
            hi = float(row['Max Duration'])
        except (ValueError, TypeError):
            errors.append(f"Activity '{label}': Duration values must be numbers.")
            continue
        if lo < 0 or avg < 0 or hi < 0:
            errors.append(f"Activity '{label}': Durations cannot be negative.")
        if lo > avg:
            errors.append(f"Activity '{label}': Min duration ({lo}) > Avg duration ({avg}).")
        if avg > hi:
            errors.append(f"Activity '{label}': Avg duration ({avg}) > Max duration ({hi}).")
        if lo > hi:
            errors.append(f"Activity '{label}': Min duration ({lo}) > Max duration ({hi}).")
        if lo == hi == avg == 0:
            errors.append(f"Activity '{label}': All durations are zero.")

    seen = set()
    for l in labels:
        if l in seen:
            errors.append(f"Duplicate activity label: '{l}'.")
        seen.add(l)

    label_set = set(labels)
    for _, row in df.iterrows():
        label = str(row.get('Label', '')).strip()
        if not label or label == 'nan':
            continue
        preds_raw = str(row.get('Predecessors', '')).strip()
        if not preds_raw or preds_raw == 'nan':
            continue
        preds = [p.strip() for p in preds_raw.split(',') if p.strip()]
        for p in preds:
            if p == label:
                errors.append(f"Activity '{label}': Cannot be its own predecessor (self-loop).")
            elif p not in label_set:
                errors.append(f"Activity '{label}': Predecessor '{p}' does not exist in the table.")
    return errors


def topo_sort(df):
    """Kahn's algorithm. Returns ordered list or None if cycle detected."""
    labels = [str(row['Label']).strip() for _, row in df.iterrows()]
    preds_map = {}
    for _, row in df.iterrows():
        label = str(row['Label']).strip()
        raw = str(row.get('Predecessors', '')).strip()
        preds = [p.strip() for p in raw.split(',') if p.strip() and p.strip() != 'nan']
        preds_map[label] = preds
    in_degree = {l: len(preds_map[l]) for l in labels}
    successors = defaultdict(list)
    for l in labels:
        for p in preds_map[l]:
            successors[p].append(l)
    queue = sorted([l for l in labels if in_degree[l] == 0])
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
        queue.sort()
    return order if len(order) == len(labels) else None


def sample_duration(rng, dist_type, lo, mode, hi):
    """Sample a single activity duration from the chosen distribution."""
    if lo == hi:
        return lo

    if dist_type == "Triangular":
        return rng.triangular(lo, mode, hi)

    elif dist_type == "PERT (Beta)":
        lam = 4
        mu = (lo + lam * mode + hi) / (lam + 2)
        mu = max(lo + 1e-9, min(hi - 1e-9, mu))
        alpha_num = (mu - lo) * (2 * mode - lo - hi)
        alpha_den = (mode - mu) * (hi - lo)
        if abs(alpha_den) < 1e-12:
            alpha = 1 + lam * (mode - lo) / (hi - lo)
        else:
            alpha = alpha_num / alpha_den
        if alpha <= 0:
            alpha = 1 + lam * (mode - lo) / (hi - lo)
        beta_param = alpha * (hi - mu) / (mu - lo) if (mu - lo) > 1e-12 else alpha
        if alpha <= 0 or beta_param <= 0:
            return rng.triangular(lo, mode, hi)
        return lo + (hi - lo) * rng.beta(max(alpha, 0.01), max(beta_param, 0.01))

    elif dist_type == "Uniform":
        return rng.uniform(lo, hi)

    elif dist_type == "Normal (truncated)":
        mu = mode
        sigma = (hi - lo) / 6.0
        if sigma <= 0:
            return mode
        for _ in range(1000):
            val = rng.normal(mu, sigma)
            if lo <= val <= hi:
                return val
        return mu

    elif dist_type == "Lognormal":
        mu_ = max(mode, 1e-6)
        std = (hi - lo) / 6.0
        if std <= 0:
            return mode
        sig2 = np.log(1 + (std / mu_) ** 2)
        val = rng.lognormal(np.log(mu_) - sig2 / 2, np.sqrt(sig2))
        return np.clip(val, lo, hi)

    return rng.triangular(lo, mode, hi)


def pert_mean(lo, mode, hi):
    """PERT analytical mean: (a + 4m + b) / 6"""
    return (lo + 4 * mode + hi) / 6.0


def pert_variance(lo, hi):
    """PERT analytical variance: ((b - a) / 6)^2"""
    return ((hi - lo) / 6.0) ** 2


def run_simulation(topo, act_info, rng, dist_type, n_sim):
    """Run Monte Carlo simulation. Returns project durations, per-activity durations, criticality counts."""
    project_durations = np.zeros(n_sim)
    activity_durations = {label: np.zeros(n_sim) for label in topo}
    activity_on_critical = {label: 0 for label in topo}

    succs = defaultdict(list)
    for label in topo:
        for p in act_info[label]['preds']:
            succs[p].append(label)

    for i in range(n_sim):
        durations = {}
        for label in topo:
            info = act_info[label]
            d = sample_duration(rng, dist_type, info['min'], info['avg'], info['max'])
            durations[label] = d
            activity_durations[label][i] = d

        ES, EF = {}, {}
        for label in topo:
            preds = act_info[label]['preds']
            ES[label] = max((EF[p] for p in preds), default=0)
            EF[label] = ES[label] + durations[label]

        proj_dur = max(EF.values())
        project_durations[i] = proj_dur

        LF, LS = {}, {}
        for label in reversed(topo):
            if not succs[label]:
                LF[label] = proj_dur
            else:
                LF[label] = min(LS[s] for s in succs[label])
            LS[label] = LF[label] - durations[label]

        for label in topo:
            if abs(LS[label] - ES[label]) < 1e-9:
                activity_on_critical[label] += 1

    return project_durations, activity_durations, activity_on_critical


def compute_deterministic_schedule(topo, act_info):
    """Forward + backward pass using PERT mean durations. Returns schedule dict."""
    mean_dur = {l: pert_mean(act_info[l]['min'], act_info[l]['avg'], act_info[l]['max']) for l in topo}

    succs = defaultdict(list)
    for label in topo:
        for p in act_info[label]['preds']:
            succs[p].append(label)

    ES, EF = {}, {}
    for label in topo:
        preds = act_info[label]['preds']
        ES[label] = max((EF[p] for p in preds), default=0)
        EF[label] = ES[label] + mean_dur[label]

    pf = max(EF.values())

    LF, LS = {}, {}
    for label in reversed(topo):
        if not succs[label]:
            LF[label] = pf
        else:
            LF[label] = min(LS[s] for s in succs[label])
        LS[label] = LF[label] - mean_dur[label]

    slack = {l: round(LS[l] - ES[l], 6) for l in topo}
    critical = [l for l in topo if slack[l] < 1e-6]

    return {
        'ES': ES, 'EF': EF, 'LS': LS, 'LF': LF,
        'mean_dur': mean_dur, 'slack': slack,
        'critical': critical, 'project_finish': pf, 'succs': succs,
    }


def draw_network_graph(df, act_info, topo, criticality_pct=None):
    """Draw a project network diagram using matplotlib."""
    depth = {}
    for label in topo:
        preds = act_info[label]['preds']
        depth[label] = 0 if not preds else max(depth[p] for p in preds) + 1

    layers = defaultdict(list)
    for label in topo:
        layers[depth[label]].append(label)

    max_depth = max(depth.values()) if depth else 0
    max_layer_size = max(len(v) for v in layers.values()) if layers else 1

    pos = {}
    for d in range(max_depth + 1):
        nodes = layers[d]
        n = len(nodes)
        for i, label in enumerate(nodes):
            pos[label] = (d * 2.5, (i - (n - 1) / 2.0) * 2.0)

    name_map = {str(row['Label']).strip(): str(row['Activity']).strip() for _, row in df.iterrows()}

    fig_w = max(8, (max_depth + 1) * 3)
    fig_h = max(4, max_layer_size * 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    for label in topo:
        x1, y1 = pos[label]
        for pred in act_info[label]['preds']:
            x0, y0 = pos[pred]
            ax.annotate("", xy=(x1 - 0.55, y1), xytext=(x0 + 0.55, y0),
                        arrowprops=dict(arrowstyle="-|>", color='#4fc3f7',
                                        lw=1.8, connectionstyle="arc3,rad=0.1"))

    for label in topo:
        x, y = pos[label]
        if criticality_pct and criticality_pct.get(label, 0) > 70:
            nc = '#ff5252'
        elif criticality_pct and criticality_pct.get(label, 0) > 40:
            nc = '#ffd740'
        else:
            nc = '#4fc3f7'
        circle = plt.Circle((x, y), 0.5, color=nc, ec='white', lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y + 0.05, label, ha='center', va='center', fontsize=13,
                fontweight='bold', color='white', zorder=4)
        short_name = name_map.get(label, label)
        if len(short_name) > 18:
            short_name = short_name[:16] + '..'
        ax.text(x, y - 0.75, short_name, ha='center', va='top', fontsize=8,
                color='#aaaaaa', zorder=4)
        info = act_info[label]
        ax.text(x, y + 0.72, f"[{info['min']:.0f}, {info['avg']:.0f}, {info['max']:.0f}]",
                ha='center', va='bottom', fontsize=7, color='#888888', zorder=4)

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    margin = 1.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Project Network Diagram', color='white', fontsize=14, fontweight='bold', pad=15)
    if criticality_pct:
        legend_elements = [
            mpatches.Patch(facecolor='#ff5252', edgecolor='white', label='High criticality (>70%)'),
            mpatches.Patch(facecolor='#ffd740', edgecolor='white', label='Medium (40-70%)'),
            mpatches.Patch(facecolor='#4fc3f7', edgecolor='white', label='Low (<40%)'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
                  facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')
    plt.tight_layout()
    return fig


def dark_fig(figsize=(12, 5)):
    # Use slightly narrower default that works better on mobile
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='white', labelsize=9)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#555')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    return fig, ax


def draw_gantt(topo, act_info, schedule, name_map, criticality_pct=None):
    """Draw a Gantt chart showing mean schedule with float/slack."""
    ES = schedule['ES']
    EF = schedule['EF']
    slack = schedule['slack']
    mean_dur = schedule['mean_dur']
    pf = schedule['project_finish']

    sorted_labels = sorted(topo, key=lambda l: (ES.get(l, 0), l))
    n = len(sorted_labels)

    fig, ax = dark_fig((13, max(4, n * 0.65)))

    for i, lbl in enumerate(sorted_labels):
        es = ES[lbl]
        dur = mean_dur[lbl]
        fl = max(0.0, slack.get(lbl, 0))
        crit = (criticality_pct or {}).get(lbl, 0)

        color = '#ff5252' if crit > 70 else '#ffd740' if crit > 40 else '#4fc3f7'

        # Float bar (dashed)
        if fl > 0.05:
            ax.barh(i, fl, left=EF[lbl], color=color, edgecolor=color,
                    height=0.45, alpha=0.15, zorder=2, linewidth=1.2, linestyle='--')

        # Work bar
        ax.barh(i, dur, left=es, color=color, edgecolor='white',
                height=0.55, alpha=0.88, zorder=3)

        # Duration label inside bar
        if dur >= 1.0:
            ax.text(es + dur / 2, i, f"{dur:.1f}w", ha='center', va='center',
                    fontsize=8.5, color='white', fontweight='bold', zorder=4)

        # Float label
        if fl > 0.8:
            ax.text(EF[lbl] + fl / 2, i, f"+{fl:.1f}w", ha='center', va='center',
                    fontsize=7, color=color, alpha=0.75, zorder=4)

        # Criticality badge
        if criticality_pct is not None:
            ax.text(pf * 1.02, i, f"{crit:.0f}%", ha='left', va='center',
                    fontsize=7.5, color=color, fontweight='bold', zorder=4)

    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{l}  {name_map.get(l, l)}" for l in sorted_labels],
                       fontsize=9.5, color='white')
    ax.invert_yaxis()
    ax.set_xlabel('Week from Project Start', color='white', fontsize=11)
    ax.set_xlim(-0.5, pf * 1.12)
    ax.set_title('Gantt Chart — Mean Schedule  (bar = work  |  dashed = float/slack)',
                 color='white', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.15, color='white')

    legend_patches = [
        mpatches.Patch(color='#ff5252', label='High criticality (>70%)'),
        mpatches.Patch(color='#ffd740', label='Medium (40-70%)'),
        mpatches.Patch(color='#4fc3f7', label='Low (<40%)'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8,
              facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')

    if criticality_pct is not None:
        ax.text(pf * 1.02, -0.8, "Crit%", fontsize=7.5, color='#888',
                fontweight='bold', va='center')

    plt.tight_layout()
    return fig

# ============================================================
# RUN SIMULATION BUTTON
# ============================================================
st.markdown("---")
run_col1, run_col2, run_col3 = st.columns([1, 3, 1])
with run_col2:
    run_button = st.button("Run Monte Carlo Simulation", type="primary", use_container_width=True)

if run_button:
    errors = validate_data(edited_df)
    if errors:
        for e in errors:
            st.error(e)
    else:
        topo = topo_sort(edited_df)
        if topo is None:
            st.error("Cycle detected in the activity network. Please check predecessors.")
        else:
            act_info = {}
            for _, row in edited_df.iterrows():
                label = str(row['Label']).strip()
                preds = [p.strip() for p in str(row.get('Predecessors', '')).split(',')
                         if p.strip() and p.strip() != 'nan']
                act_info[label] = {
                    'preds': preds,
                    'min': float(row['Min Duration']),
                    'avg': float(row['Avg Duration']),
                    'max': float(row['Max Duration']),
                }

            name_map = {str(row['Label']).strip(): str(row['Activity']).strip()
                        for _, row in edited_df.iterrows()}

            rng = np.random.default_rng(seed if seed > 0 else None)

            with st.spinner("Running simulation..."):
                project_durations, activity_durations, activity_on_critical = run_simulation(
                    topo, act_info, rng, dist_type, num_simulations
                )

            # Deterministic schedule (PERT analytical)
            schedule = compute_deterministic_schedule(topo, act_info)

            mean_dur = np.mean(project_durations)
            std_dur = np.std(project_durations)
            p_service = np.percentile(project_durations, service_level)
            median_dur = np.median(project_durations)
            cv_pct = (std_dur / mean_dur * 100) if mean_dur > 0 else 0
            crit_pct = {l: activity_on_critical[l] / num_simulations * 100 for l in topo}

            # PERT analytical estimate
            cp = schedule['critical']
            pert_project_mean = schedule['project_finish']
            pert_cp_std = np.sqrt(sum(pert_variance(act_info[l]['min'], act_info[l]['max']) for l in cp))
            from scipy.stats import norm
            z = norm.ppf(service_level / 100)
            pert_service_level = pert_project_mean + z * pert_cp_std

            # ============================================================
            # PROJECT HEALTH INDICATOR
            # ============================================================
            st.markdown("---")
            if cv_pct < 5:
                health_color, health_text = '#69f0ae', 'Low Risk'
            elif cv_pct < 10:
                health_color, health_text = '#ffd740', 'Moderate Risk'
            else:
                health_color, health_text = '#ff5252', 'High Risk'

            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 1rem;">
                <span style="font-size: 1.3rem; font-weight: 700; color: {health_color};">
                    Schedule Risk: {health_text}
                </span>
                <span style="color: #888; font-size: 0.95rem; margin-left: 1rem;">
                    (CV = {cv_pct:.1f}%)
                </span>
            </div>
            """, unsafe_allow_html=True)

            # ============================================================
            # RESULTS — METRICS
            # ============================================================
            st.markdown('<div class="section-header">Simulation Results</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="insight-box">Distribution: <b>{dist_type}</b> &nbsp;|&nbsp; '
                        f'Iterations: <b>{num_simulations:,}</b> &nbsp;|&nbsp; '
                        f'Service Level: <b>{service_level}%</b></div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            for col, val, lbl in [
                (m1, mean_dur, "Mean Duration (wks)"),
                (m2, std_dur, "Std Deviation (wks)"),
                (m3, p_service, f"{service_level}% Service Level (wks)"),
                (m4, median_dur, "Median Duration (wks)"),
            ]:
                with col:
                    st.markdown(f'<div class="metric-card">'
                                f'<div class="metric-value">{val:.2f}</div>'
                                f'<div class="metric-label">{lbl}</div></div>',
                                unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ============================================================
            # PERT vs MONTE CARLO COMPARISON
            # ============================================================
            st.markdown('<div class="section-header">PERT Analytical vs Monte Carlo Comparison</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="insight-box">PERT uses the normal approximation along the critical path. '
                        'Monte Carlo simulates all paths and captures path-switching effects.</div>',
                        unsafe_allow_html=True)

            comp_df = pd.DataFrame({
                'Method': ['PERT (Analytical)', 'Monte Carlo'],
                'Mean Duration (wks)': [f"{pert_project_mean:.2f}", f"{mean_dur:.2f}"],
                f'{service_level}% Completion (wks)': [f"{pert_service_level:.2f}", f"{p_service:.2f}"],
                'Critical Path Std Dev (wks)': [f"{pert_cp_std:.2f}", f"{std_dur:.2f}"],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Critical path display
            cp_str = " -> ".join([f"{l} ({name_map.get(l, l)})" for l in cp])
            st.markdown(f'<div class="insight-box">Dominant Critical Path: <b>{cp_str}</b><br>'
                        f'Total mean duration along critical path: <b>{pert_project_mean:.1f} weeks</b></div>',
                        unsafe_allow_html=True)

            # ============================================================
            # SCHEDULE TABLE (ES, EF, LS, LF, Float, Critical)
            # ============================================================
            st.markdown('<div class="section-header">Schedule Table (PERT Mean Durations)</div>',
                        unsafe_allow_html=True)

            sched_rows = []
            for lbl in topo:
                sched_rows.append({
                    'Label': lbl,
                    'Activity': name_map.get(lbl, lbl),
                    'PERT Mean': f"{schedule['mean_dur'][lbl]:.2f}",
                    'ES': f"{schedule['ES'][lbl]:.2f}",
                    'EF': f"{schedule['EF'][lbl]:.2f}",
                    'LS': f"{schedule['LS'][lbl]:.2f}",
                    'LF': f"{schedule['LF'][lbl]:.2f}",
                    'Float': f"{schedule['slack'][lbl]:.2f}",
                    'Critical': 'Yes' if lbl in cp else '',
                })
            st.dataframe(pd.DataFrame(sched_rows), use_container_width=True, hide_index=True)

            # ============================================================
            # NETWORK GRAPH — colored by criticality
            # ============================================================
            st.markdown('<div class="section-header">Network Diagram (Criticality Colored)</div>',
                        unsafe_allow_html=True)
            fig_net = draw_network_graph(edited_df, act_info, topo, criticality_pct=crit_pct)
            st.pyplot(fig_net)
            plt.close(fig_net)

            # ============================================================
            # GANTT CHART
            # ============================================================
            st.markdown('<div class="section-header">Gantt Chart</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-box">Bars show each activity\'s mean duration (ES to EF). '
                        'Dashed extension = float (slack). Color = criticality level. '
                        'Percentage on the right = criticality index.</div>', unsafe_allow_html=True)
            fig_gantt = draw_gantt(topo, act_info, schedule, name_map, criticality_pct=crit_pct)
            st.pyplot(fig_gantt)
            plt.close(fig_gantt)

            # ============================================================
            # HISTOGRAM
            # ============================================================
            st.markdown('<div class="section-header">Duration Distribution</div>', unsafe_allow_html=True)

            fig, ax = dark_fig((12, 5))
            n_vals, bins, patches = ax.hist(project_durations, bins=60, edgecolor='none', alpha=0.85)
            cm = plt.cm.get_cmap('cool')
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col_norm = plt.Normalize(bin_centers.min(), bin_centers.max())
            for c, p in zip(bin_centers, patches):
                p.set_facecolor(cm(col_norm(c)))
            ax.axvline(mean_dur, color='#ff5252', linestyle='--', linewidth=2.5,
                       label=f'Mean = {mean_dur:.2f} wks')
            ax.axvline(p_service, color='#ffd740', linestyle='--', linewidth=2.5,
                       label=f'{service_level}th %ile = {p_service:.2f} wks')
            ax.axvline(median_dur, color='#69f0ae', linestyle=':', linewidth=2,
                       label=f'Median = {median_dur:.2f} wks')
            ax.set_xlabel('Project Duration (weeks)', color='white', fontsize=12)
            ax.set_ylabel('Frequency', color='white', fontsize=12)
            ax.set_title(f'Monte Carlo Simulation — {num_simulations:,} Iterations ({dist_type})',
                         color='white', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11, facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')
            ax.grid(axis='y', alpha=0.15, color='white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # ============================================================
            # CUMULATIVE S-CURVE
            # ============================================================
            st.markdown('<div class="section-header">Cumulative Probability (S-Curve)</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="insight-box">The S-curve shows the probability of completing the project '
                        'within a given number of weeks.</div>', unsafe_allow_html=True)

            fig2, ax2 = dark_fig((12, 5))
            sorted_dur = np.sort(project_durations)
            cumulative = np.arange(1, len(sorted_dur) + 1) / len(sorted_dur) * 100
            ax2.plot(sorted_dur, cumulative, color='#4fc3f7', linewidth=2.5)
            ax2.axhline(service_level, color='#ffd740', linestyle='--', linewidth=1.5,
                        label=f'{service_level}% -> {p_service:.2f} wks')
            ax2.axvline(p_service, color='#ffd740', linestyle='--', linewidth=1.5)
            ax2.axhline(50, color='#69f0ae', linestyle=':', linewidth=1.2,
                        label=f'50% -> {median_dur:.2f} wks')
            ax2.fill_between(sorted_dur, cumulative, alpha=0.08, color='#4fc3f7')
            ax2.set_xlabel('Project Duration (weeks)', color='white', fontsize=12)
            ax2.set_ylabel('Cumulative Probability (%)', color='white', fontsize=12)
            ax2.set_title('Cumulative Distribution Function', color='white', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11, facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')
            ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
            ax2.grid(alpha=0.15, color='white')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            # ============================================================
            # CRITICALITY INDEX BAR CHART
            # ============================================================
            st.markdown('<div class="section-header">Sensitivity Analysis — Activity Criticality Index</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="insight-box">The Criticality Index shows how often each activity '
                        'appears on the critical path. Activities with high criticality '
                        'are the biggest risk drivers.</div>', unsafe_allow_html=True)

            crit_data = []
            for label in topo:
                crit_data.append({'Label': label, 'Activity': name_map.get(label, label),
                                  'Criticality (%)': crit_pct[label]})
            crit_df = pd.DataFrame(crit_data).sort_values('Criticality (%)', ascending=True)

            fig3, ax3 = dark_fig((10, max(3, len(topo) * 0.6)))
            colors = ['#ff5252' if v > 70 else '#ffd740' if v > 40 else '#69f0ae'
                      for v in crit_df['Criticality (%)']]
            bars = ax3.barh(
                [f"{r['Label']}: {r['Activity']}" for _, r in crit_df.iterrows()],
                crit_df['Criticality (%)'], color=colors, edgecolor='none', height=0.6)
            for bar, val in zip(bars, crit_df['Criticality (%)']):
                ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                         f'{val:.1f}%', va='center', color='white', fontsize=10)
            ax3.set_xlabel('Criticality Index (%)', color='white', fontsize=12)
            ax3.set_title('How Often Each Activity Is on the Critical Path',
                          color='white', fontsize=14, fontweight='bold')
            ax3.set_xlim(0, 110)
            ax3.grid(axis='x', alpha=0.15, color='white')
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            # ============================================================
            # TORNADO CHART
            # ============================================================
            st.markdown('<div class="section-header">Tornado Chart — Duration Sensitivity</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="insight-box">Spearman rank correlation between each '
                        'activity\'s sampled duration and total project duration. '
                        'Higher absolute correlation = more influence on project length.</div>',
                        unsafe_allow_html=True)

            correlations = []
            for label in topo:
                corr, _ = sp_stats.spearmanr(activity_durations[label], project_durations)
                correlations.append({
                    'Label': label, 'Activity': name_map.get(label, label),
                    'Correlation': corr if not np.isnan(corr) else 0.0,
                })
            corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=True)

            fig4, ax4 = dark_fig((10, max(3, len(topo) * 0.6)))
            bar_colors = ['#ff5252' if v > 0 else '#4fc3f7' for v in corr_df['Correlation']]
            ax4.barh([f"{r['Label']}: {r['Activity']}" for _, r in corr_df.iterrows()],
                     corr_df['Correlation'], color=bar_colors, edgecolor='none', height=0.6)
            for i, (_, r) in enumerate(corr_df.iterrows()):
                offset = 0.02 if r['Correlation'] >= 0 else -0.02
                ha = 'left' if r['Correlation'] >= 0 else 'right'
                ax4.text(r['Correlation'] + offset, i, f"{r['Correlation']:.3f}",
                         va='center', ha=ha, color='white', fontsize=9)
            ax4.axvline(0, color='#555', linewidth=1)
            ax4.set_xlabel('Spearman Rank Correlation with Project Duration', color='white', fontsize=12)
            ax4.set_title('Activity Duration Sensitivity (Tornado)',
                          color='white', fontsize=14, fontweight='bold')
            ax4.grid(axis='x', alpha=0.15, color='white')
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

            # ============================================================
            # BOX PLOTS
            # ============================================================
            st.markdown('<div class="section-header">Activity Duration Box Plots</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="insight-box">Box plots show the spread of sampled durations. '
                        'Box = IQR (25th-75th), line = median, whiskers = 1.5x IQR.</div>',
                        unsafe_allow_html=True)

            fig5, ax5 = dark_fig((12, max(4, len(topo) * 0.55)))
            box_data = [activity_durations[label] for label in topo]
            box_labels = [f"{l}: {name_map.get(l, l)}" for l in topo]
            bp = ax5.boxplot(box_data, vert=False, labels=box_labels, patch_artist=True, widths=0.5,
                             boxprops=dict(facecolor='#1a1a2e', edgecolor='#4fc3f7', linewidth=1.5),
                             whiskerprops=dict(color='#4fc3f7', linewidth=1.2),
                             capprops=dict(color='#4fc3f7', linewidth=1.2),
                             medianprops=dict(color='#ff5252', linewidth=2),
                             flierprops=dict(marker='o', markerfacecolor='#ffd740',
                                             markeredgecolor='none', markersize=3, alpha=0.5))
            ax5.set_xlabel('Duration (weeks)', color='white', fontsize=12)
            ax5.set_title('Sampled Duration Distributions per Activity',
                          color='white', fontsize=14, fontweight='bold')
            ax5.tick_params(axis='y', colors='white')
            ax5.grid(axis='x', alpha=0.15, color='white')
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close(fig5)

            # ============================================================
            # SCHEDULE CONTINGENCY BUFFER CHART
            # ============================================================
            st.markdown('<div class="section-header">Schedule Contingency Buffer</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="insight-box">Baseline (PERT mean) = <b>{pert_project_mean:.1f} weeks</b>. '
                        f'Buffer needed above baseline to reach each confidence level.</div>',
                        unsafe_allow_html=True)

            pcts = [50, 60, 70, 75, 80, 85, 90, 95, 97, 99]
            buffers = [max(0.0, float(np.percentile(project_durations, p)) - pert_project_mean) for p in pcts]
            buf_colors = ['#ff5252' if p >= 90 else '#ffd740' if p >= 75 else '#4fc3f7' for p in pcts]

            fig6, ax6 = dark_fig((10, 4.5))
            buf_bars = ax6.bar([f"{p}%" for p in pcts], buffers, color=buf_colors,
                               edgecolor='none', width=0.6)
            for bar, val in zip(buf_bars, buffers):
                ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f"{val:.1f}w", ha='center', va='bottom', color='white', fontsize=9)
            ax6.set_xlabel('Confidence Level', color='white', fontsize=12)
            ax6.set_ylabel('Buffer Required (weeks)', color='white', fontsize=12)
            ax6.set_title(f'Schedule Contingency Buffer (above baseline of {pert_project_mean:.1f} wks)',
                          color='white', fontsize=13, fontweight='bold')
            ax6.grid(axis='y', alpha=0.15, color='white')
            plt.tight_layout()
            st.pyplot(fig6)
            plt.close(fig6)

            # Buffer table
            buf_df = pd.DataFrame({
                'Confidence': [f"{p}%" for p in pcts],
                'Completion (wks)': [f"{np.percentile(project_durations, p):.1f}" for p in pcts],
                'Buffer above baseline (wks)': [f"{b:.1f}" for b in buffers],
            })
            st.dataframe(buf_df, use_container_width=True, hide_index=True)

            # ============================================================
            # COMPLETION PROBABILITY LOOKUP
            # ============================================================
            st.markdown('<div class="section-header">Completion Probability Lookup</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="insight-box">Enter a target completion time to see the probability '
                        'of finishing the project by that week.</div>', unsafe_allow_html=True)

            target_col1, target_col2 = st.columns([1, 2])
            with target_col1:
                target_week = st.number_input(
                    "Target completion (weeks)",
                    min_value=float(np.floor(np.min(project_durations))),
                    max_value=float(np.ceil(np.max(project_durations))) + 10,
                    value=float(round(p_service, 1)), step=0.5,
                )
            with target_col2:
                prob = np.mean(project_durations <= target_week) * 100
                color = '#69f0ae' if prob >= 90 else '#ffd740' if prob >= 50 else '#ff5252'
                advice = ("Very safe deadline." if prob >= 90
                          else "Moderate confidence. Consider buffer on high-criticality activities."
                          if prob >= 50 else "Low confidence. Add contingency or compress scope.")
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 0.5rem;">
                    <div class="metric-value" style="color: {color};">{prob:.1f}%</div>
                    <div class="metric-label">Probability of finishing by week {target_week:.1f}</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f'<p style="color: {color}; font-size: 0.9rem; margin-top: 0.5rem;">{advice}</p>',
                            unsafe_allow_html=True)

            # ============================================================
            # PERCENTILE LOOKUP TABLE
            # ============================================================
            st.markdown('<div class="section-header">Percentile Lookup Table</div>', unsafe_allow_html=True)
            pct_rows = [{'Service Level': f"{p}%",
                         'Completion Time (wks)': f"{np.percentile(project_durations, p):.1f}"}
                        for p in [50, 60, 70, 75, 80, 85, 90, 95, 97, 99]]
            st.dataframe(pd.DataFrame(pct_rows), use_container_width=True, hide_index=True)

            # ============================================================
            # ACTIVITY RISK PROFILE
            # ============================================================
            st.markdown('<div class="section-header">Activity Risk Profile</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-box">Combined view of duration uncertainty and schedule criticality. '
                        'Use this to prioritize where estimation effort and monitoring matter most.</div>',
                        unsafe_allow_html=True)

            risk_rows = []
            for label in topo:
                info = act_info[label]
                md = pert_mean(info['min'], info['avg'], info['max'])
                unc = info['max'] - info['min']
                cv = (unc / 6.0) / md * 100 if md > 0 else 0
                crit = crit_pct[label]
                if crit >= 70 or (crit >= 30 and cv >= 20):
                    rating = "High"
                elif crit >= 30 or cv >= 15:
                    rating = "Medium"
                else:
                    rating = "Low"
                risk_rows.append({
                    'Label': label, 'Activity': name_map.get(label, label),
                    'Mean Duration (wks)': f"{md:.2f}",
                    'Uncertainty Range (wks)': f"{unc:.1f}",
                    'Coeff. of Variation (%)': f"{cv:.1f}",
                    'Criticality (%)': f"{crit:.1f}",
                    'Risk Rating': rating,
                })
            st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

            # ============================================================
            # SUMMARY STATISTICS
            # ============================================================
            st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Std Deviation', 'Minimum', '25th Percentile',
                              'Median', '75th Percentile', f'{service_level}th Percentile', 'Maximum'],
                'Duration (weeks)': [
                    f"{mean_dur:.2f}", f"{std_dur:.2f}",
                    f"{np.min(project_durations):.2f}",
                    f"{np.percentile(project_durations, 25):.2f}",
                    f"{median_dur:.2f}",
                    f"{np.percentile(project_durations, 75):.2f}",
                    f"{p_service:.2f}",
                    f"{np.max(project_durations):.2f}",
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            # ============================================================
            # PER-ACTIVITY STATS
            # ============================================================
            st.markdown('<div class="section-header">Per-Activity Duration Statistics</div>',
                        unsafe_allow_html=True)
            act_stats = []
            for label in topo:
                d = activity_durations[label]
                act_stats.append({
                    'Label': label, 'Activity': name_map.get(label, label),
                    'Mean': f"{np.mean(d):.2f}", 'Std Dev': f"{np.std(d):.2f}",
                    'Min Sampled': f"{np.min(d):.2f}", 'Max Sampled': f"{np.max(d):.2f}",
                    'Criticality': f"{crit_pct[label]:.1f}%",
                })
            st.dataframe(pd.DataFrame(act_stats), use_container_width=True, hide_index=True)

            # ============================================================
            # EXPORT RESULTS
            # ============================================================
            st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)
            exp1, exp2 = st.columns(2)
            with exp1:
                raw_csv = pd.DataFrame({'Simulated Duration (weeks)': project_durations}).to_csv(index=False)
                st.download_button("Download Raw Simulation Data (CSV)", data=raw_csv,
                                   file_name="simulation_raw_output.csv", mime="text/csv",
                                   use_container_width=True)
            with exp2:
                summary_export = pd.DataFrame([
                    {'Metric': 'Distribution', 'Value': dist_type},
                    {'Metric': 'Iterations', 'Value': num_simulations},
                    {'Metric': 'Seed', 'Value': seed if seed > 0 else 'random'},
                    {'Metric': 'Mean Duration (wks)', 'Value': f"{mean_dur:.2f}"},
                    {'Metric': 'Std Deviation (wks)', 'Value': f"{std_dur:.2f}"},
                    {'Metric': 'PERT Baseline (wks)', 'Value': f"{pert_project_mean:.2f}"},
                    {'Metric': 'Median (wks)', 'Value': f"{median_dur:.2f}"},
                    {'Metric': f'P{service_level} (wks)', 'Value': f"{p_service:.2f}"},
                    {'Metric': 'P80 (wks)', 'Value': f"{np.percentile(project_durations, 80):.2f}"},
                    {'Metric': 'P90 (wks)', 'Value': f"{np.percentile(project_durations, 90):.2f}"},
                    {'Metric': 'P99 (wks)', 'Value': f"{np.percentile(project_durations, 99):.2f}"},
                ])
                summary_csv = summary_export.to_csv(index=False)
                st.download_button("Download Summary Statistics (CSV)", data=summary_csv,
                                   file_name="simulation_summary.csv", mime="text/csv",
                                   use_container_width=True)

            # ============================================================
            # COMPLETION
            # ============================================================
            st.markdown("---")
            st.markdown(f"""
            <div style="text-align:center; padding: 2rem;">
                <h3 style="color: #4fc3f7; margin-top: 0.5rem;">Simulation Complete</h3>
                <p style="color: #aaa; font-size: 1.1rem;">
                    {num_simulations:,} iterations completed using {dist_type} distribution.
                </p>
            </div>
            """, unsafe_allow_html=True)
