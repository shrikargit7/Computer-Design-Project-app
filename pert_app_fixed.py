import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from scipy import stats

st.set_page_config(page_title="PERT Project Analyzer", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 8px; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("📊 PERT Project Duration Analyzer")
st.caption("Analyze any project using PERT methodology + Monte Carlo simulation.")

# ── Default Data ──────────────────────────────────────────────────────────────
DEFAULT_DATA = [
    {"Activity": "Design",                 "Label": "A", "Predecessors": "",    "Min": 16, "Avg": 21, "Max": 26},
    {"Activity": "Build prototype",        "Label": "B", "Predecessors": "A",   "Min":  3, "Avg":  6, "Max":  9},
    {"Activity": "Evaluate equipment",     "Label": "C", "Predecessors": "A",   "Min":  5, "Avg":  7, "Max":  9},
    {"Activity": "Test prototype",         "Label": "D", "Predecessors": "B",   "Min":  2, "Avg":  3, "Max":  4},
    {"Activity": "Write equipment report", "Label": "E", "Predecessors": "C,D", "Min":  4, "Avg":  6, "Max":  8},
    {"Activity": "Write methods report",   "Label": "F", "Predecessors": "C,D", "Min":  6, "Avg":  8, "Max": 10},
    {"Activity": "Write final report",     "Label": "G", "Predecessors": "E,F", "Min":  1, "Avg":  2, "Max":  3},
]

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(DEFAULT_DATA)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    n_sim = st.slider(
        "Monte Carlo Simulations", 1_000, 50_000, 10_000, 1_000,
        help=(
            "How many random scenarios to run. More simulations = more accurate "
            "probability estimates, but slower computation. "
            "1,000 is fast; 10,000 is a reliable default; 50,000 is very precise."
        ),
    )
    svc_lvl = st.slider(
        "Service Level (%)", 80, 99, 95, 1,
        help=(
            "The confidence level for your completion deadline. "
            "95% means there is a 95% chance the project finishes ON or BEFORE the "
            "forecasted date — only a 5% chance it runs over. "
            "Higher values give safer (longer) deadlines."
        ),
    )
    rng_seed = st.number_input(
        "Random Seed", value=42, step=1,
        help=(
            "Controls the starting point of the random number generator. "
            "The same seed always produces identical simulation results, "
            "making your analysis fully reproducible. "
            "Change it to verify that results are stable across different runs."
        ),
    )
    st.divider()
    if st.button("🔄 Reset to Sample Project", use_container_width=True):
        st.session_state.df = pd.DataFrame(DEFAULT_DATA)
        st.rerun()
    st.markdown("### How to use")
    st.markdown("""
1. Edit or add activities in the table  
2. Set predecessors as comma-separated labels (e.g. `A,B`)  
3. Leave predecessors blank for start activities  
4. Adjust simulation settings  
5. Click **Run Analysis**
    """)

# ── PERT Helpers ──────────────────────────────────────────────────────────────
def pert_te(a, m, b):   return (a + 4*m + b) / 6
def pert_var(a, b):     return ((b - a) / 6) ** 2
def pert_std(a, b):     return (b - a) / 6

def beta_pert_samples(a, m, b, size):
    te = pert_te(a, m, b)
    if b == a:
        return np.full(size, a)
    alpha  = 6 * (te - a) / (b - a)
    beta_p = 6 * (b - te) / (b - a)
    if alpha <= 0 or beta_p <= 0:
        return np.full(size, te)
    # FIX: must scale by (b - a) to map beta[0,1] → [a, b]
    return a + stats.beta.rvs(alpha, beta_p, size=size) * (b - a)

# ── Graph / Path Helpers ──────────────────────────────────────────────────────
def build_graph(df):
    G = nx.DiGraph()
    labels = set(df["Label"].astype(str))
    for lbl in labels:
        G.add_node(lbl)
    for _, row in df.iterrows():
        preds = str(row["Predecessors"]).strip()
        if preds and preds not in ("-", "nan"):
            for p in preds.split(","):
                p = p.strip()
                if p in labels:
                    G.add_edge(p, row["Label"])
    return G

def all_paths(G):
    starts = [n for n in G.nodes() if G.in_degree(n)  == 0]
    ends   = [n for n in G.nodes() if G.out_degree(n) == 0]
    paths  = []
    for s in starts:
        for e in ends:
            try:
                paths.extend(nx.all_simple_paths(G, s, e))
            except nx.NetworkXNoPath:
                pass
    return paths

def critical_path(df):
    G  = build_graph(df)
    te = {r["Label"]: pert_te(r["Min"], r["Avg"], r["Max"]) for _, r in df.iterrows()}
    paths = all_paths(G)
    if not paths:
        return [], 0.0, {}, te
    durations = {tuple(p): sum(te[n] for n in p) for p in paths}
    cp = list(max(durations, key=durations.get))
    return cp, durations[tuple(cp)], durations, te

def run_mc(df, n, seed):
    np.random.seed(seed)
    paths = all_paths(build_graph(df))
    smps  = {r["Label"]: beta_pert_samples(r["Min"], r["Avg"], r["Max"], n)
             for _, r in df.iterrows()}
    proj  = np.zeros(n)
    for path in paths:
        proj = np.maximum(proj, sum(smps[node] for node in path))
    return proj

# ── Activity Input ────────────────────────────────────────────────────────────
st.header("📋 Project Activities")
st.caption("Add / remove rows dynamically. Predecessors: comma-separated labels, blank = start activity.")

edited = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Activity":     st.column_config.TextColumn("Activity Name",          width="medium"),
        "Label":        st.column_config.TextColumn("Label",                  width="small"),
        "Predecessors": st.column_config.TextColumn("Predecessors (e.g. A,B)", width="medium"),
        "Min":          st.column_config.NumberColumn("Min Duration (a)", min_value=0, step=1),
        "Avg":          st.column_config.NumberColumn("Avg Duration (m)", min_value=0, step=1),
        "Max":          st.column_config.NumberColumn("Max Duration (b)", min_value=0, step=1),
    },
    hide_index=True,
    key="editor",
)
st.session_state.df = edited

# ── Run Button ────────────────────────────────────────────────────────────────
run = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

if run:
    df = edited.copy()
    for col in ["Min", "Avg", "Max"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Label", "Min", "Avg", "Max"])
    df["Label"] = df["Label"].astype(str).str.strip()

    if df.empty:
        st.error("No valid activities found. Please check your table.")
        st.stop()

    # PERT calculations
    df["te"] = df.apply(lambda r: round(pert_te(r["Min"], r["Avg"], r["Max"]), 3), axis=1)
    df["σ²"] = df.apply(lambda r: round(pert_var(r["Min"], r["Max"]), 3), axis=1)
    df["σ"]  = df.apply(lambda r: round(pert_std(r["Min"], r["Max"]), 3), axis=1)

    cp, cp_dur, all_dur, te_dict = critical_path(df)
    var_dict = dict(zip(df["Label"], df["σ²"]))
    cp_var   = sum(var_dict.get(n, 0) for n in cp)
    cp_sigma = np.sqrt(cp_var)

    # Monte Carlo
    with st.spinner(f"Running {n_sim:,} simulations…"):
        mc = run_mc(df, n_sim, int(rng_seed))

    mc_mean = mc.mean()
    mc_sl   = np.percentile(mc, svc_lvl)
    pert_sl = stats.norm.ppf(svc_lvl / 100, cp_dur, cp_sigma)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.divider()
    st.header("📈 Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Duration (PERT)", f"{cp_dur:.1f} wks")
    c2.metric("Mean Duration (MC)",   f"{mc_mean:.1f} wks")
    c3.metric(f"{svc_lvl}% Service Level (MC)", f"{mc_sl:.1f} wks")
    c4.metric("Critical Path σ (PERT)", f"{cp_sigma:.2f} wks")

    # ── Critical Path ─────────────────────────────────────────────────────────
    st.subheader("🔴 Critical Path")
    if cp:
        st.success(f"**{'  →  '.join(cp)}**  |  Expected Duration: **{cp_dur:.1f} weeks**")
    else:
        st.warning("No path found. Check predecessor labels.")

    with st.expander("📍 All Paths & Durations"):
        paths_df = pd.DataFrame(
            [{"Path": " → ".join(p), "Duration (wks)": round(d, 2),
              "On Critical Path": list(p) == cp}
             for p, d in sorted(all_dur.items(), key=lambda x: -x[1])]
        )
        st.dataframe(paths_df, use_container_width=True, hide_index=True)

    # ── PERT Table ────────────────────────────────────────────────────────────
    st.subheader("📋 PERT Activity Summary")
    cols = [c for c in ["Activity", "Label", "Min", "Avg", "Max", "te", "σ²", "σ"] if c in df.columns]
    st.dataframe(
        df[cols].rename(columns={"te": "Expected (te)", "σ²": "Variance (σ²)", "σ": "Std Dev (σ)"}),
        use_container_width=True, hide_index=True
    )

    # ── Histogram ─────────────────────────────────────────────────────────────
    st.subheader("📊 Monte Carlo Duration Distribution")
    xr = np.linspace(mc.min(), mc.max(), 400)
    xs = xr[xr <= mc_sl]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=mc, nbinsx=60, histnorm="probability density",
        marker_color="#4C78A8", opacity=0.75, name="MC Durations"
    ))
    fig.add_trace(go.Scatter(
        x=xr, y=stats.norm.pdf(xr, mc_mean, mc.std()),
        mode="lines", line=dict(color="#F58518", width=2.5), name="Normal fit"
    ))
    fig.add_vline(x=mc_mean, line_dash="dash", line_color="green", line_width=2,
                  annotation_text=f"Mean {mc_mean:.1f}w", annotation_position="top right")
    fig.add_vline(x=mc_sl, line_dash="dot", line_color="red", line_width=2.5,
                  annotation_text=f"{svc_lvl}%ile {mc_sl:.1f}w", annotation_position="top left")
    fig.add_traces(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([stats.norm.pdf(xs, mc_mean, mc.std()), np.zeros(len(xs))]),
        fill="toself", fillcolor="rgba(255,0,0,0.08)",
        line=dict(width=0), showlegend=False
    ))
    fig.update_layout(
        title=f"Project Duration Distribution — {n_sim:,} Monte Carlo Simulations",
        xaxis_title="Duration (weeks)", yaxis_title="Probability Density",
        template="plotly_white", height=460, legend=dict(x=0.02, y=0.95)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Service Level Summary ─────────────────────────────────────────────────
    st.subheader(f"🎯 {svc_lvl}% Service Level Summary")
    ca, cb = st.columns(2)
    ca.info(
        f"**PERT Normal Approximation**\n\n"
        f"To achieve {svc_lvl}% confidence, complete by **{pert_sl:.1f} weeks**\n\n"
        f"*(μ = {cp_dur:.1f}, σ = {cp_sigma:.2f}, critical path only)*"
    )
    cb.info(
        f"**Monte Carlo Simulation**\n\n"
        f"To achieve {svc_lvl}% confidence, complete by **{mc_sl:.1f} weeks**\n\n"
        f"*(based on {n_sim:,} simulations, full network)*"
    )
    st.caption(
        "💡 PERT approximation uses only critical-path variance; "
        "Monte Carlo captures uncertainty across **all** paths, so it's typically more conservative."
    )

else:
    st.info("👆 Edit your activities above and click **Run Analysis** to see the full results.")
    prev = edited.copy()
    try:
        for col in ["Min", "Avg", "Max"]:
            prev[col] = pd.to_numeric(prev[col], errors="coerce")
        prev["Expected (te)"] = prev.apply(
            lambda r: round(pert_te(r["Min"], r["Avg"], r["Max"]), 2)
            if pd.notna(r["Min"]) else None, axis=1)
        st.dataframe(
            prev[["Activity", "Label", "Predecessors", "Min", "Avg", "Max", "Expected (te)"]],
            use_container_width=True, hide_index=True
        )
    except Exception:
        st.dataframe(prev, use_container_width=True, hide_index=True)
