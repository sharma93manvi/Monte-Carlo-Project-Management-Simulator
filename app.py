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
from scipy.stats import norm

# --- Page Config ---
st.set_page_config(page_title="Monte Carlo Project Simulator", page_icon="🎲",
                   layout="wide", initial_sidebar_state="expanded")

# --- Session State Init ---
for key, default in [("sim_results", None), ("sim_topo", None), ("sim_act_info", None),
                     ("sim_name_map", None), ("sim_schedule", None), ("sim_crit_pct", None),
                     ("sim_dist_type", None), ("sim_n", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

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
section[data-testid="stFileUploader"] label { font-size: 1rem !important; margin-bottom: 0.5rem !important; }
section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] { padding: 1rem !important; }
section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] span {
    white-space: normal !important; line-height: 1.4 !important;
}
</style>
""", unsafe_allow_html=True)

# Responsive CSS
st.markdown("""
<style>
@media (max-width: 768px) {
    .main-header { padding: 1.5rem 1rem; border-radius: 10px; }
    .main-header h1 { font-size: 1.5rem; }
    .main-header p { font-size: 0.85rem; }
    .metric-card { padding: 1rem 0.8rem; margin-bottom: 0.6rem; }
    .metric-card .metric-value { font-size: 1.4rem; }
    .metric-card .metric-label { font-size: 0.7rem; }
    .section-header { font-size: 1.15rem; }
    .insight-box { padding: 0.75rem 0.9rem; font-size: 0.85rem; }
    div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        min-width: 48% !important; flex: 1 1 48% !important; margin-bottom: 0.5rem;
    }
    div.stButton > button[kind="primary"] { font-size: 1rem; padding: 0.7rem 1rem; }
    div[data-testid="stDataFrame"] { overflow-x: auto !important; }
}
@media (max-width: 480px) {
    .main-header h1 { font-size: 1.25rem; }
    .metric-card .metric-value { font-size: 1.2rem; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        min-width: 100% !important; flex: 1 1 100% !important;
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
)
DIST_DESCRIPTIONS = {
    "Triangular": "Uses min, mode (most likely), and max. Good default when you have three-point estimates.",
    "PERT (Beta)": "Beta distribution weighted toward the mode (lambda=4). Less probability to extremes. Widely used in project management.",
    "Uniform": "Every value between min and max is equally likely. Use when you have no idea about the most likely value.",
    "Normal (truncated)": "Bell curve centered on the average, truncated at min/max. Std dev = (max-min)/6.",
    "Lognormal": "Right-skewed distribution. Good for activities like approvals or procurement with long tails.",
}
st.sidebar.markdown(f'<div class="dist-info"><b>{dist_type}</b><br>{DIST_DESCRIPTIONS[dist_type]}</div>',
                    unsafe_allow_html=True)

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Monte Carlo Simulation", type="primary", use_container_width=True)
if st.session_state.sim_results is not None:
    st.sidebar.success("Results ready — check the tabs")
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit")

# ============================================================
# DEFAULT DATA
# ============================================================
DEFAULT_DATA = {
    'Label': ['A','B','C','D','E','F','G'],
    'Activity': ['Design','Build prototype','Evaluate equipment','Test prototype',
                 'Write equipment report','Write methods report','Write final report'],
    'Predecessors': ['','A','A','B','C,D','C,D','E,F'],
    'Min Duration': [16,3,5,2,4,6,1],
    'Avg Duration': [21,6,7,3,6,8,2],
    'Max Duration': [26,9,9,4,8,10,3],
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_data(df):
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
        for col in ['Min Duration', 'Avg Duration', 'Max Duration']:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                errors.append(f"Activity '{label}': {col} is missing.")
        try:
            lo, avg, hi = float(row['Min Duration']), float(row['Avg Duration']), float(row['Max Duration'])
        except (ValueError, TypeError):
            errors.append(f"Activity '{label}': Duration values must be numbers.")
            continue
        if lo < 0 or avg < 0 or hi < 0:
            errors.append(f"Activity '{label}': Durations cannot be negative.")
        if lo > avg:
            errors.append(f"Activity '{label}': Min ({lo}) > Avg ({avg}).")
        if avg > hi:
            errors.append(f"Activity '{label}': Avg ({avg}) > Max ({hi}).")
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
        for p in [x.strip() for x in preds_raw.split(',') if x.strip()]:
            if p == label:
                errors.append(f"Activity '{label}': Cannot be its own predecessor.")
            elif p not in label_set:
                errors.append(f"Activity '{label}': Predecessor '{p}' does not exist.")
    return errors


def topo_sort(df):
    labels = [str(row['Label']).strip() for _, row in df.iterrows()]
    preds_map = {}
    for _, row in df.iterrows():
        label = str(row['Label']).strip()
        raw = str(row.get('Predecessors', '')).strip()
        preds_map[label] = [p.strip() for p in raw.split(',') if p.strip() and p.strip() != 'nan']
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
        alpha = alpha_num / alpha_den if abs(alpha_den) > 1e-12 else 1 + lam * (mode - lo) / (hi - lo)
        if alpha <= 0:
            alpha = 1 + lam * (mode - lo) / (hi - lo)
        beta_p = alpha * (hi - mu) / (mu - lo) if (mu - lo) > 1e-12 else alpha
        if alpha <= 0 or beta_p <= 0:
            return rng.triangular(lo, mode, hi)
        return lo + (hi - lo) * rng.beta(max(alpha, 0.01), max(beta_p, 0.01))
    elif dist_type == "Uniform":
        return rng.uniform(lo, hi)
    elif dist_type == "Normal (truncated)":
        sigma = (hi - lo) / 6.0
        if sigma <= 0:
            return mode
        for _ in range(1000):
            val = rng.normal(mode, sigma)
            if lo <= val <= hi:
                return val
        return mode
    elif dist_type == "Lognormal":
        mu_ = max(mode, 1e-6)
        std = (hi - lo) / 6.0
        if std <= 0:
            return mode
        sig2 = np.log(1 + (std / mu_) ** 2)
        return np.clip(rng.lognormal(np.log(mu_) - sig2 / 2, np.sqrt(sig2)), lo, hi)
    return rng.triangular(lo, mode, hi)


def pert_mean(lo, mode, hi):
    return (lo + 4 * mode + hi) / 6.0

def pert_variance(lo, hi):
    return ((hi - lo) / 6.0) ** 2


def run_simulation(topo, act_info, rng, dist_type, n_sim):
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
            ES[label] = max((EF[p] for p in act_info[label]['preds']), default=0)
            EF[label] = ES[label] + durations[label]
        proj_dur = max(EF.values())
        project_durations[i] = proj_dur
        LF, LS = {}, {}
        for label in reversed(topo):
            LF[label] = proj_dur if not succs[label] else min(LS[s] for s in succs[label])
            LS[label] = LF[label] - durations[label]
        for label in topo:
            if abs(LS[label] - ES[label]) < 1e-9:
                activity_on_critical[label] += 1
    return project_durations, activity_durations, activity_on_critical


def compute_deterministic_schedule(topo, act_info):
    mean_dur = {l: pert_mean(act_info[l]['min'], act_info[l]['avg'], act_info[l]['max']) for l in topo}
    succs = defaultdict(list)
    for label in topo:
        for p in act_info[label]['preds']:
            succs[p].append(label)
    ES, EF = {}, {}
    for label in topo:
        ES[label] = max((EF[p] for p in act_info[label]['preds']), default=0)
        EF[label] = ES[label] + mean_dur[label]
    pf = max(EF.values())
    LF, LS = {}, {}
    for label in reversed(topo):
        LF[label] = pf if not succs[label] else min(LS[s] for s in succs[label])
        LS[label] = LF[label] - mean_dur[label]
    slack = {l: round(LS[l] - ES[l], 6) for l in topo}
    critical = [l for l in topo if slack[l] < 1e-6]
    return {'ES': ES, 'EF': EF, 'LS': LS, 'LF': LF, 'mean_dur': mean_dur,
            'slack': slack, 'critical': critical, 'project_finish': pf, 'succs': succs}


# ============================================================
# PLOT HELPERS
# ============================================================

def dark_fig(figsize=(12, 5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='white', labelsize=9)
    for s in ['bottom', 'left']:
        ax.spines[s].set_color('#555')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    return fig, ax


def draw_network_graph(act_info, topo, name_map, crit_pct=None):
    depth = {}
    for label in topo:
        preds = act_info[label]['preds']
        depth[label] = 0 if not preds else max(depth[p] for p in preds) + 1
    layers = defaultdict(list)
    for label in topo:
        layers[depth[label]].append(label)
    max_depth = max(depth.values()) if depth else 0
    max_layer = max(len(v) for v in layers.values()) if layers else 1
    pos = {}
    for d in range(max_depth + 1):
        nodes = layers[d]
        n = len(nodes)
        for i, label in enumerate(nodes):
            pos[label] = (d * 2.5, (i - (n - 1) / 2.0) * 2.0)
    fw = max(8, (max_depth + 1) * 3)
    fh = max(4, max_layer * 2)
    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    for label in topo:
        x1, y1 = pos[label]
        for pred in act_info[label]['preds']:
            x0, y0 = pos[pred]
            ax.annotate("", xy=(x1 - 0.55, y1), xytext=(x0 + 0.55, y0),
                        arrowprops=dict(arrowstyle="-|>", color='#4fc3f7', lw=1.8,
                                        connectionstyle="arc3,rad=0.1"))
    for label in topo:
        x, y = pos[label]
        c = crit_pct.get(label, 0) if crit_pct else 0
        nc = '#ff5252' if c > 70 else '#ffd740' if c > 40 else '#4fc3f7'
        ax.add_patch(plt.Circle((x, y), 0.5, color=nc, ec='white', lw=2, zorder=3))
        ax.text(x, y + 0.05, label, ha='center', va='center', fontsize=13,
                fontweight='bold', color='white', zorder=4)
        sn = name_map.get(label, label)
        ax.text(x, y - 0.75, sn[:16] + '..' if len(sn) > 18 else sn,
                ha='center', va='top', fontsize=8, color='#aaa', zorder=4)
        info = act_info[label]
        ax.text(x, y + 0.72, f"[{info['min']:.0f},{info['avg']:.0f},{info['max']:.0f}]",
                ha='center', va='bottom', fontsize=7, color='#888', zorder=4)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
    ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Project Network Diagram', color='white', fontsize=14, fontweight='bold', pad=15)
    if crit_pct:
        ax.legend(handles=[
            mpatches.Patch(facecolor='#ff5252', edgecolor='white', label='High (>70%)'),
            mpatches.Patch(facecolor='#ffd740', edgecolor='white', label='Medium (40-70%)'),
            mpatches.Patch(facecolor='#4fc3f7', edgecolor='white', label='Low (<40%)'),
        ], loc='lower right', fontsize=8, facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')
    plt.tight_layout()
    return fig


def draw_gantt(topo, act_info, schedule, name_map, crit_pct=None):
    ES, EF = schedule['ES'], schedule['EF']
    slack, mean_dur, pf = schedule['slack'], schedule['mean_dur'], schedule['project_finish']
    sorted_labels = sorted(topo, key=lambda l: (ES.get(l, 0), l))
    n = len(sorted_labels)
    fig, ax = dark_fig((13, max(4, n * 0.65)))
    for i, lbl in enumerate(sorted_labels):
        es, dur, fl = ES[lbl], mean_dur[lbl], max(0.0, slack.get(lbl, 0))
        crit = (crit_pct or {}).get(lbl, 0)
        color = '#ff5252' if crit > 70 else '#ffd740' if crit > 40 else '#4fc3f7'
        if fl > 0.05:
            ax.barh(i, fl, left=EF[lbl], color=color, edgecolor=color,
                    height=0.45, alpha=0.15, zorder=2, linestyle='--')
        ax.barh(i, dur, left=es, color=color, edgecolor='white', height=0.55, alpha=0.88, zorder=3)
        if dur >= 1.0:
            ax.text(es + dur / 2, i, f"{dur:.1f}w", ha='center', va='center',
                    fontsize=8.5, color='white', fontweight='bold', zorder=4)
        if fl > 0.8:
            ax.text(EF[lbl] + fl / 2, i, f"+{fl:.1f}w", ha='center', va='center',
                    fontsize=7, color=color, alpha=0.75, zorder=4)
        if crit_pct is not None:
            ax.text(pf * 1.02, i, f"{crit:.0f}%", ha='left', va='center',
                    fontsize=7.5, color=color, fontweight='bold', zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{l}  {name_map.get(l, l)}" for l in sorted_labels], fontsize=9.5, color='white')
    ax.invert_yaxis()
    ax.set_xlabel('Week from Project Start', color='white', fontsize=11)
    ax.set_xlim(-0.5, pf * 1.12)
    ax.set_title('Gantt Chart (bar=work | dashed=float/slack)', color='white', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.15, color='white')
    ax.legend(handles=[
        mpatches.Patch(color='#ff5252', label='High (>70%)'),
        mpatches.Patch(color='#ffd740', label='Medium (40-70%)'),
        mpatches.Patch(color='#4fc3f7', label='Low (<40%)'),
    ], loc='lower right', fontsize=8, facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')
    plt.tight_layout()
    return fig


# ============================================================
# TABS
# ============================================================
tab_setup, tab_dash, tab_gantt, tab_risk, tab_sched = st.tabs([
    "Project Setup", "Dashboard", "Gantt & Timeline",
    "Risk Analysis", "Schedule Planning"
])

# TAB 1 - PROJECT SETUP
with tab_setup:
    st.markdown('<div class="section-header">Project Activities</div>', unsafe_allow_html=True)
    sub_edit, sub_upload, sub_template = st.tabs(["Edit Table", "Upload CSV/Excel", "Download Template"])
    with sub_edit:
        st.markdown("Edit the table below. Use comma-separated labels for predecessors (e.g., `C,D`).")
        if 'df' not in st.session_state:
            st.session_state.df = pd.DataFrame(DEFAULT_DATA)
        edited_df = st.data_editor(
            st.session_state.df, num_rows="dynamic", use_container_width=True, key="activity_table",
            column_config={
                "Label": st.column_config.TextColumn("Label", width="small"),
                "Activity": st.column_config.TextColumn("Activity Name", width="medium"),
                "Predecessors": st.column_config.TextColumn("Predecessors", width="small"),
                "Min Duration": st.column_config.NumberColumn("Min", min_value=0.0, format="%.1f"),
                "Avg Duration": st.column_config.NumberColumn("Avg (Mode)", min_value=0.0, format="%.1f"),
                "Max Duration": st.column_config.NumberColumn("Max", min_value=0.0, format="%.1f"),
            })
    with sub_upload:
        st.markdown("Upload a CSV or Excel file with columns: `Label`, `Activity`, "
                    "`Predecessors`, `Min Duration`, `Avg Duration`, `Max Duration`")
        st.markdown("")
        st.markdown("**Select your project file (.csv, .xlsx, .xls):**")
        uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
        if uploaded_file is not None:
            try:
                upload_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                req = {'Label', 'Activity', 'Predecessors', 'Min Duration', 'Avg Duration', 'Max Duration'}
                if req.issubset(set(upload_df.columns)):
                    upload_df['Predecessors'] = upload_df['Predecessors'].fillna('')
                    st.session_state.df = upload_df[list(req)].copy()
                    edited_df = st.session_state.df
                    st.success(f"Loaded {len(upload_df)} activities")
                    st.dataframe(upload_df, use_container_width=True, hide_index=True)
                else:
                    st.error(f"Missing columns: {', '.join(req - set(upload_df.columns))}")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    with sub_template:
        st.markdown("Download a template pre-filled with the Computer Design project.")
        tpl = pd.DataFrame(DEFAULT_DATA)
        st.download_button("Download CSV Template", data=tpl.to_csv(index=False),
                           file_name="project_template.csv", mime="text/csv", use_container_width=True)
        st.dataframe(tpl, use_container_width=True, hide_index=True)
    errors = validate_data(edited_df)
    if errors:
        for e in errors:
            st.error(e)
    else:
        t = topo_sort(edited_df)
        if t is None:
            st.error("Cycle detected. Fix predecessors.")
        else:
            st.success("Network valid. Hit **Run Monte Carlo Simulation** in the sidebar.")
    with st.expander("Distribution Reference"):
        for d, desc in DIST_DESCRIPTIONS.items():
            st.markdown(f"- **{d}**: {desc}")
    with st.expander("Export current table as CSV"):
        st.download_button("Download table", data=edited_df.to_csv(index=False).encode(),
                           file_name="project_activities.csv", mime="text/csv")

# RUN SIMULATION
if run_button:
    errors = validate_data(edited_df)
    if errors:
        st.sidebar.error("Fix validation errors in Project Setup first.")
    else:
        topo = topo_sort(edited_df)
        if topo is None:
            st.sidebar.error("Cycle detected. Fix predecessors.")
        else:
            act_info = {}
            for _, row in edited_df.iterrows():
                label = str(row['Label']).strip()
                preds = [p.strip() for p in str(row.get('Predecessors', '')).split(',')
                         if p.strip() and p.strip() != 'nan']
                act_info[label] = {'preds': preds, 'min': float(row['Min Duration']),
                                   'avg': float(row['Avg Duration']), 'max': float(row['Max Duration'])}
            name_map = {str(r['Label']).strip(): str(r['Activity']).strip() for _, r in edited_df.iterrows()}
            rng = np.random.default_rng(seed if seed > 0 else None)
            with st.spinner("Running simulation..."):
                pd_arr, ad, ac = run_simulation(topo, act_info, rng, dist_type, num_simulations)
            sched = compute_deterministic_schedule(topo, act_info)
            cp = {l: ac[l] / num_simulations * 100 for l in topo}
            st.session_state.sim_results = {'project_durations': pd_arr, 'activity_durations': ad, 'activity_on_critical': ac}
            st.session_state.sim_topo = topo
            st.session_state.sim_act_info = act_info
            st.session_state.sim_name_map = name_map
            st.session_state.sim_schedule = sched
            st.session_state.sim_crit_pct = cp
            st.session_state.sim_dist_type = dist_type
            st.session_state.sim_n = num_simulations
            st.rerun()

sim = st.session_state.sim_results


# TAB 2 - DASHBOARD
with tab_dash:
    if sim is None:
        st.info("Run the simulation from the sidebar to see the dashboard.")
    else:
        proj_dur = sim['project_durations']
        topo = st.session_state.sim_topo
        act_info = st.session_state.sim_act_info
        name_map = st.session_state.sim_name_map
        schedule = st.session_state.sim_schedule
        crit_pct = st.session_state.sim_crit_pct
        n_sim = st.session_state.sim_n
        dt = st.session_state.sim_dist_type
        mean_d = float(np.mean(proj_dur))
        std_d = float(np.std(proj_dur))
        med_d = float(np.median(proj_dur))
        p_sl = float(np.percentile(proj_dur, service_level))
        cv = (std_d / mean_d * 100) if mean_d > 0 else 0
        ppf = schedule['project_finish']
        hc = '#69f0ae' if cv < 5 else '#ffd740' if cv < 10 else '#ff5252'
        ht = 'Low Risk' if cv < 5 else 'Moderate Risk' if cv < 10 else 'High Risk'
        st.markdown(f'<div style="text-align:center;margin-bottom:1rem;"><span style="font-size:1.3rem;font-weight:700;color:{hc};">Schedule Risk: {ht}</span><span style="color:#888;font-size:0.95rem;margin-left:1rem;">(CV={cv:.1f}%)</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box">Distribution: <b>{dt}</b> | Iterations: <b>{n_sim:,}</b> | Service Level: <b>{service_level}%</b></div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl in [(m1, mean_d, "Mean (wks)"), (m2, std_d, "Std Dev (wks)"), (m3, p_sl, f"P{service_level} (wks)"), (m4, med_d, "Median (wks)")]:
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{val:.2f}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        cl, cr = st.columns(2)
        with cl:
            st.markdown("#### Top Risk Activities")
            top5 = sorted(crit_pct.items(), key=lambda x: -x[1])[:5]
            st.dataframe(pd.DataFrame([{'Activity': f"{l} - {name_map.get(l,l)}", 'Criticality': f"{v:.0f}%", 'Level': 'High' if v>=70 else ('Medium' if v>=40 else 'Low')} for l, v in top5]), hide_index=True, use_container_width=True)
        with cr:
            st.markdown("#### Schedule Scenarios")
            st.dataframe(pd.DataFrame([
                {'Scenario': 'PERT Baseline', 'Duration': f"{ppf:.1f}", 'Buffer': '-'},
                {'Scenario': 'Simulated Mean', 'Duration': f"{mean_d:.1f}", 'Buffer': f"+{mean_d-ppf:.1f}"},
                {'Scenario': '80% Confidence', 'Duration': f"{np.percentile(proj_dur,80):.1f}", 'Buffer': f"+{np.percentile(proj_dur,80)-ppf:.1f}"},
                {'Scenario': '90% Confidence', 'Duration': f"{np.percentile(proj_dur,90):.1f}", 'Buffer': f"+{np.percentile(proj_dur,90)-ppf:.1f}"},
                {'Scenario': '95% Confidence', 'Duration': f"{np.percentile(proj_dur,95):.1f}", 'Buffer': f"+{np.percentile(proj_dur,95)-ppf:.1f}"},
            ]), hide_index=True, use_container_width=True)
        st.markdown("---")
        st.markdown("#### PERT Analytical vs Monte Carlo")
        cp = schedule['critical']
        cp_std = np.sqrt(sum(pert_variance(act_info[l]['min'], act_info[l]['max']) for l in cp))
        z = norm.ppf(service_level / 100)
        pert_sl = ppf + z * cp_std
        st.dataframe(pd.DataFrame({'Method': ['PERT (Analytical)', 'Monte Carlo'], 'Mean (wks)': [f"{ppf:.2f}", f"{mean_d:.2f}"], f'P{service_level} (wks)': [f"{pert_sl:.2f}", f"{p_sl:.2f}"], 'Std Dev (wks)': [f"{cp_std:.2f}", f"{std_d:.2f}"]}), hide_index=True, use_container_width=True)
        cp_str = " -> ".join([f"{l} ({name_map.get(l,l)})" for l in cp])
        st.markdown(f'<div class="insight-box">Critical Path: <b>{cp_str}</b></div>', unsafe_allow_html=True)
        st.markdown("---")
        fig, ax = dark_fig((12, 5))
        _, bins, patches = ax.hist(proj_dur, bins=60, edgecolor='none', alpha=0.85)
        cm = plt.cm.get_cmap('cool')
        bc = 0.5*(bins[:-1]+bins[1:])
        cn = plt.Normalize(bc.min(), bc.max())
        for c, p in zip(bc, patches):
            p.set_facecolor(cm(cn(c)))
        ax.axvline(mean_d, color='#ff5252', ls='--', lw=2.5, label=f'Mean={mean_d:.2f}')
        ax.axvline(p_sl, color='#ffd740', ls='--', lw=2.5, label=f'P{service_level}={p_sl:.2f}')
        ax.axvline(med_d, color='#69f0ae', ls=':', lw=2, label=f'Median={med_d:.2f}')
        ax.set_xlabel('Project Duration (weeks)', color='white', fontsize=12)
        ax.set_ylabel('Frequency', color='white', fontsize=12)
        ax.set_title(f'Duration Distribution ({n_sim:,} iterations, {dt})', color='white', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')
        ax.grid(axis='y', alpha=0.15, color='white')
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
        with st.expander("Full Summary Statistics"):
            st.dataframe(pd.DataFrame({'Statistic': ['Mean','Std Dev','Min','P25','Median','P75',f'P{service_level}','Max'], 'Weeks': [f"{mean_d:.2f}",f"{std_d:.2f}",f"{np.min(proj_dur):.2f}",f"{np.percentile(proj_dur,25):.2f}",f"{med_d:.2f}",f"{np.percentile(proj_dur,75):.2f}",f"{p_sl:.2f}",f"{np.max(proj_dur):.2f}"]}), hide_index=True, use_container_width=True)


# TAB 3 - GANTT & TIMELINE
with tab_gantt:
    if sim is None:
        st.info("Run the simulation from the sidebar to see the Gantt chart.")
    else:
        topo = st.session_state.sim_topo
        act_info = st.session_state.sim_act_info
        name_map = st.session_state.sim_name_map
        schedule = st.session_state.sim_schedule
        crit_pct = st.session_state.sim_crit_pct
        st.markdown('<div class="section-header">Gantt Chart</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-box">Bars = mean duration (ES to EF). Dashed = float/slack. Color = criticality.</div>', unsafe_allow_html=True)
        fg = draw_gantt(topo, act_info, schedule, name_map, crit_pct)
        st.pyplot(fg); plt.close(fg)
        st.markdown("---")
        st.markdown('<div class="section-header">Schedule Table</div>', unsafe_allow_html=True)
        rows = []
        for lbl in topo:
            rows.append({'Label': lbl, 'Activity': name_map.get(lbl, lbl), 'PERT Mean': f"{schedule['mean_dur'][lbl]:.2f}", 'ES': f"{schedule['ES'][lbl]:.2f}", 'EF': f"{schedule['EF'][lbl]:.2f}", 'LS': f"{schedule['LS'][lbl]:.2f}", 'LF': f"{schedule['LF'][lbl]:.2f}", 'Float': f"{schedule['slack'][lbl]:.2f}", 'Critical': 'Yes' if lbl in schedule['critical'] else '', 'Crit%': f"{crit_pct[lbl]:.1f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.markdown("---")
        st.markdown('<div class="section-header">Network Diagram</div>', unsafe_allow_html=True)
        fn = draw_network_graph(act_info, topo, name_map, crit_pct)
        st.pyplot(fn); plt.close(fn)

# TAB 4 - RISK ANALYSIS
with tab_risk:
    if sim is None:
        st.info("Run the simulation from the sidebar to see risk analysis.")
    else:
        proj_dur = sim['project_durations']
        act_dur = sim['activity_durations']
        topo = st.session_state.sim_topo
        act_info = st.session_state.sim_act_info
        name_map = st.session_state.sim_name_map
        crit_pct = st.session_state.sim_crit_pct
        st.markdown('<div class="section-header">Activity Criticality Index</div>', unsafe_allow_html=True)
        crit_df = pd.DataFrame([{'Label': l, 'Activity': name_map.get(l,l), 'Crit': crit_pct[l]} for l in topo]).sort_values('Crit', ascending=True)
        fig3, ax3 = dark_fig((10, max(3, len(topo)*0.6)))
        colors = ['#ff5252' if v>70 else '#ffd740' if v>40 else '#69f0ae' for v in crit_df['Crit']]
        bars = ax3.barh([f"{r['Label']}: {r['Activity']}" for _,r in crit_df.iterrows()], crit_df['Crit'], color=colors, edgecolor='none', height=0.6)
        for bar, val in zip(bars, crit_df['Crit']):
            ax3.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', color='white', fontsize=10)
        ax3.set_xlabel('Criticality Index (%)', color='white', fontsize=12)
        ax3.set_title('Critical Path Frequency', color='white', fontsize=14, fontweight='bold')
        ax3.set_xlim(0, 110); ax3.grid(axis='x', alpha=0.15, color='white')
        plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)
        st.markdown("---")
        st.markdown('<div class="section-header">Tornado Chart</div>', unsafe_allow_html=True)
        corrs = []
        for label in topo:
            c, _ = sp_stats.spearmanr(act_dur[label], proj_dur)
            corrs.append({'Label': label, 'Activity': name_map.get(label,label), 'Corr': c if not np.isnan(c) else 0.0})
        corr_df = pd.DataFrame(corrs).sort_values('Corr', key=abs, ascending=True)
        fig4, ax4 = dark_fig((10, max(3, len(topo)*0.6)))
        bcolors = ['#ff5252' if v>0 else '#4fc3f7' for v in corr_df['Corr']]
        ax4.barh([f"{r['Label']}: {r['Activity']}" for _,r in corr_df.iterrows()], corr_df['Corr'], color=bcolors, edgecolor='none', height=0.6)
        for i, (_,r) in enumerate(corr_df.iterrows()):
            off = 0.02 if r['Corr']>=0 else -0.02
            ax4.text(r['Corr']+off, i, f"{r['Corr']:.3f}", va='center', ha='left' if r['Corr']>=0 else 'right', color='white', fontsize=9)
        ax4.axvline(0, color='#555', lw=1)
        ax4.set_xlabel('Spearman Correlation', color='white', fontsize=12)
        ax4.set_title('Duration Sensitivity', color='white', fontsize=14, fontweight='bold')
        ax4.grid(axis='x', alpha=0.15, color='white')
        plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)
        st.markdown("---")
        st.markdown('<div class="section-header">Activity Duration Box Plots</div>', unsafe_allow_html=True)
        fig5, ax5 = dark_fig((12, max(4, len(topo)*0.55)))
        ax5.boxplot([act_dur[l] for l in topo], vert=False, labels=[f"{l}: {name_map.get(l,l)}" for l in topo], patch_artist=True, widths=0.5, boxprops=dict(facecolor='#1a1a2e', edgecolor='#4fc3f7', lw=1.5), whiskerprops=dict(color='#4fc3f7', lw=1.2), capprops=dict(color='#4fc3f7', lw=1.2), medianprops=dict(color='#ff5252', lw=2), flierprops=dict(marker='o', markerfacecolor='#ffd740', markeredgecolor='none', markersize=3, alpha=0.5))
        ax5.set_xlabel('Duration (weeks)', color='white', fontsize=12)
        ax5.set_title('Sampled Duration Distributions', color='white', fontsize=14, fontweight='bold')
        ax5.tick_params(axis='y', colors='white'); ax5.grid(axis='x', alpha=0.15, color='white')
        plt.tight_layout(); st.pyplot(fig5); plt.close(fig5)
        st.markdown("---")
        st.markdown('<div class="section-header">Activity Risk Profile</div>', unsafe_allow_html=True)
        rr = []
        for label in topo:
            info = act_info[label]
            md = pert_mean(info['min'], info['avg'], info['max'])
            unc = info['max'] - info['min']
            cv = (unc/6.0)/md*100 if md>0 else 0
            crit = crit_pct[label]
            rating = 'High' if crit>=70 or (crit>=30 and cv>=20) else 'Medium' if crit>=30 or cv>=15 else 'Low'
            rr.append({'Label': label, 'Activity': name_map.get(label,label), 'Mean': f"{md:.2f}", 'Range': f"{unc:.1f}", 'CV%': f"{cv:.1f}", 'Crit%': f"{crit:.1f}", 'Risk': rating})
        st.dataframe(pd.DataFrame(rr), hide_index=True, use_container_width=True)


# TAB 5 - SCHEDULE PLANNING
with tab_sched:
    if sim is None:
        st.info("Run the simulation from the sidebar to see schedule planning.")
    else:
        proj_dur = sim['project_durations']
        schedule = st.session_state.sim_schedule
        n_sim = st.session_state.sim_n
        dt = st.session_state.sim_dist_type
        ppf = schedule['project_finish']
        mean_d = float(np.mean(proj_dur))
        p_sl = float(np.percentile(proj_dur, service_level))
        st.markdown('<div class="section-header">Cumulative Probability (S-Curve)</div>', unsafe_allow_html=True)
        fig2, ax2 = dark_fig((12, 5))
        sd = np.sort(proj_dur)
        cum = np.arange(1, len(sd)+1)/len(sd)*100
        ax2.plot(sd, cum, color='#4fc3f7', lw=2.5)
        ax2.axhline(service_level, color='#ffd740', ls='--', lw=1.5, label=f'{service_level}% -> {p_sl:.2f}')
        ax2.axvline(p_sl, color='#ffd740', ls='--', lw=1.5)
        ax2.axhline(50, color='#69f0ae', ls=':', lw=1.2, label=f'50% -> {np.median(proj_dur):.2f}')
        ax2.fill_between(sd, cum, alpha=0.08, color='#4fc3f7')
        ax2.set_xlabel('Project Duration (weeks)', color='white', fontsize=12)
        ax2.set_ylabel('Cumulative Probability (%)', color='white', fontsize=12)
        ax2.set_title('Cumulative Distribution Function', color='white', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax2.grid(alpha=0.15, color='white')
        plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)
        st.markdown("---")
        st.markdown('<div class="section-header">Schedule Contingency Buffer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box">Baseline (PERT) = <b>{ppf:.1f} weeks</b>. Buffer needed at each confidence level.</div>', unsafe_allow_html=True)
        pcts = [50,60,70,75,80,85,90,95,97,99]
        buffers = [max(0.0, float(np.percentile(proj_dur, p)) - ppf) for p in pcts]
        fig6, ax6 = dark_fig((10, 4.5))
        bcc = ['#ff5252' if p>=90 else '#ffd740' if p>=75 else '#4fc3f7' for p in pcts]
        bb = ax6.bar([f"{p}%" for p in pcts], buffers, color=bcc, edgecolor='none', width=0.6)
        for bar, val in zip(bb, buffers):
            ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f"{val:.1f}w", ha='center', va='bottom', color='white', fontsize=9)
        ax6.set_xlabel('Confidence Level', color='white', fontsize=12)
        ax6.set_ylabel('Buffer (weeks)', color='white', fontsize=12)
        ax6.set_title(f'Contingency Buffer (above {ppf:.1f} wk baseline)', color='white', fontsize=13, fontweight='bold')
        ax6.grid(axis='y', alpha=0.15, color='white')
        plt.tight_layout(); st.pyplot(fig6); plt.close(fig6)
        st.dataframe(pd.DataFrame({'Confidence': [f"{p}%" for p in pcts], 'Completion (wks)': [f"{np.percentile(proj_dur,p):.1f}" for p in pcts], 'Buffer (wks)': [f"{b:.1f}" for b in buffers]}), hide_index=True, use_container_width=True)
        st.markdown("---")
        st.markdown('<div class="section-header">Completion Probability Lookup</div>', unsafe_allow_html=True)
        tc1, tc2 = st.columns([1, 2])
        with tc1:
            tw = st.number_input("Target completion (weeks)", min_value=float(np.floor(np.min(proj_dur))), max_value=float(np.ceil(np.max(proj_dur)))+10, value=float(round(p_sl, 1)), step=0.5)
        with tc2:
            prob = np.mean(proj_dur <= tw) * 100
            clr = '#69f0ae' if prob>=90 else '#ffd740' if prob>=50 else '#ff5252'
            adv = "Very safe deadline." if prob>=90 else "Moderate confidence. Consider buffer." if prob>=50 else "Low confidence. Add contingency."
            st.markdown(f'<div class="metric-card" style="margin-top:0.5rem;"><div class="metric-value" style="color:{clr};">{prob:.1f}%</div><div class="metric-label">P(finish by week {tw:.1f})</div></div>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:{clr};font-size:0.9rem;margin-top:0.5rem;">{adv}</p>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="section-header">Percentile Lookup Table</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{'Service Level': f"{p}%", 'Completion (wks)': f"{np.percentile(proj_dur,p):.1f}"} for p in [50,60,70,75,80,85,90,95,97,99]]), hide_index=True, use_container_width=True)
        st.markdown("---")
        st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        with e1:
            st.download_button("Download Raw Simulation (CSV)", data=pd.DataFrame({'Duration (wks)': proj_dur}).to_csv(index=False), file_name="simulation_raw.csv", mime="text/csv", use_container_width=True)
        with e2:
            st.download_button("Download Summary (CSV)", data=pd.DataFrame([{'Metric': 'Distribution', 'Value': dt}, {'Metric': 'Iterations', 'Value': n_sim}, {'Metric': 'Mean (wks)', 'Value': f"{mean_d:.2f}"}, {'Metric': 'Std Dev (wks)', 'Value': f"{np.std(proj_dur):.2f}"}, {'Metric': 'Baseline (wks)', 'Value': f"{ppf:.2f}"}, {'Metric': f'P{service_level} (wks)', 'Value': f"{p_sl:.2f}"}]).to_csv(index=False), file_name="simulation_summary.csv", mime="text/csv", use_container_width=True)
