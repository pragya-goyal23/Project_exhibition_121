from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

BASE = Path(__file__).parent
DEFAULT_THRESHOLD = 0.30

st.set_page_config(page_title="Room Energy Audit", page_icon="⚡", layout="wide")

#vocabulary
STATUS = {
    "NORMAL": ("Normal", "✅", "#2e7d32"),
    "UNUSUAL": ("Unusual", "🟡", "#f9a825"),
    "POTENTIALLY_ACTIONABLE": ("Worth checking", "🔴", "#d32f2f"),
    "INSUFFICIENT_EVIDENCE": ("Not enough data", "⚪", "#757575"),
}
ROUTE = {
    "DECIDE": "Clear enough to act on",
    "REVIEW": "Needs a person to look",
    "ABSTAIN": "Can't tell",
}
NEXT_STEP = {
    "NORMAL": "Nothing to do.",
    "UNUSUAL": "No immediate escalation; continue monitoring.",
    "POTENTIALLY_ACTIONABLE": "Inspect equipment and operating schedule.",
    "INSUFFICIENT_EVIDENCE": "Obtain or validate the missing data (for example occupancy).",
}
PRIORITY = {"POTENTIALLY_ACTIONABLE": 0, "UNUSUAL": 1, "INSUFFICIENT_EVIDENCE": 2, "NORMAL": 3}
SAMPLES = {
    "Illustrative day (project data)": "energy_data_graph_based.csv",
    "Random synthetic day": "energy_data_random.csv",
}
REQUIRED = ["Timestamp", "Power_kW", "Expected_Power_kW"]


def status_text(code, tech=False):
    name, emoji, _ = STATUS[code]
    return f"{emoji} {name}" + (f" ({code})" if tech else "")


#data + rules
@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


def parse_times(ts):
    s = ts.astype(str).str.strip()
    if s.str.match(r"^\d{1,2}:\d{2}(:\d{2})?$").all():  
        return pd.to_datetime("2000-01-01 " + s, errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def normalise(raw):
    df = raw.copy().rename(columns={"Equipment": "Equipment_Status"})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))
    if "Room_ID" not in df.columns:
        df["Room_ID"] = "Room"
    df["Time"] = parse_times(df["Timestamp"])
    if df["Time"].isna().any():
        raise ValueError("Some Timestamp values could not be read (use HH:MM or YYYY-MM-DD HH:MM:SS).")
    for c in ["Power_kW", "Expected_Power_kW"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Power_kW", "Expected_Power_kW"]).copy()
    df["Energy_kWh"] = pd.to_numeric(df["Energy_kWh"], errors="coerce").fillna(df["Power_kW"]) \
        if "Energy_kWh" in df.columns else df["Power_kW"]
    for c in ["Occupancy", "Scheduled"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else float("nan")
    if "Equipment_Status" not in df.columns:
        df["Equipment_Status"] = "Unknown"
    if "Data_Quality" not in df.columns:
        df["Data_Quality"] = "Good"
    df["Quality_OK"] = df["Data_Quality"].astype(str).str.upper().eq("GOOD") \
        & df["Occupancy"].notna() & df["Scheduled"].notna()
    return df


def decide(r):
    """Same rules as decision_states.py, plus a plain-English reason."""
    if not r.Quality_OK:
        return "INSUFFICIENT_EVIDENCE", "Data quality is not good or occupancy/schedule is missing, so the system does not judge this reading."
    if not r.Anomaly:
        return "NORMAL", "Power is within the allowed range around the expected value."
    if r.Occupancy > 0 or r.Scheduled == 1:
        return "UNUSUAL", "Power is off from expected, but the room is occupied or scheduled, so this may be legitimate use."
    if r.Actionability_Score >= 0.80:
        return "POTENTIALLY_ACTIONABLE", "Power is off from expected, the room is empty and unscheduled, and the evidence is strong."
    return "UNUSUAL", "Power is off from expected, but there is not enough supporting evidence to call it actionable."


def confidence(r):
    """Same values as confidence_routing.py."""
    if not r.Quality_OK:
        return 0.25
    return {"NORMAL": 0.90, "UNUSUAL": 0.70, "POTENTIALLY_ACTIONABLE": 0.85}.get(r.Decision_State, 0.30)


def route(c):
    return "DECIDE" if c >= 0.80 else "REVIEW" if c >= 0.50 else "ABSTAIN"


def num(v):
    if pd.isna(v):
        return "unknown"
    return int(v) if float(v).is_integer() else v


def analyse(raw, thr):
    df = normalise(raw)
    frames = []
    for _, g in df.groupby("Room_ID", sort=False):
        g = g.sort_values("Time", kind="stable").copy()
        g["Residual_kW"] = g["Power_kW"] - g["Expected_Power_kW"]
        g["Anomaly"] = g["Residual_kW"].abs() > thr
        g["Context_Mismatch"] = (g["Occupancy"] == 0) & (g["Scheduled"] == 0)
        g["Persistent"] = g["Anomaly"] | g["Anomaly"].shift(1, fill_value=False)
        g["Pts_Anomaly"] = g["Anomaly"] * 0.40
        g["Pts_Context"] = g["Context_Mismatch"] * 0.30
        g["Pts_Persistent"] = g["Persistent"] * 0.20
        g["Pts_Quality"] = g["Quality_OK"] * 0.10
        g["Actionability_Score"] = (
            g[["Pts_Anomaly", "Pts_Context", "Pts_Persistent", "Pts_Quality"]].sum(axis=1).clip(0, 1).round(2)
        )
        frames.append(g)
    df = pd.concat(frames).reset_index(drop=True)
    res = df.apply(decide, axis=1, result_type="expand")
    df["Decision_State"], df["Reason"] = res[0], res[1]
    df["Confidence"] = df.apply(confidence, axis=1)
    df["Route"] = df["Confidence"].apply(route)
    df["Measurement_ID"] = [f"MEAS_{i + 1:03d}" for i in range(len(df))]
    df["Baseline_ID"], df["Rule_Version"] = "BASE_V1", "RULE_V1"
    df["Evidence_Bundle"] = df.apply(
        lambda r: (
            f"Observed={r.Power_kW}kW | Expected={r.Expected_Power_kW}kW | Residual={r.Residual_kW:.2f}kW | "
            f"Occupancy={num(r.Occupancy)} | Scheduled={num(r.Scheduled)} | "
            f"Equipment={r.Equipment_Status} | DataQuality={r.Data_Quality}"
        ),
        axis=1,
    )
    fmt = "%H:%M" if df["Time"].dt.normalize().nunique() == 1 else "%d %b %H:%M"
    df["Label"] = df["Time"].dt.strftime(fmt)
    return df


def describe(r):
    occ = "occupancy unknown" if pd.isna(r.Occupancy) else \
        (f"room occupied ({num(r.Occupancy)})" if r.Occupancy > 0 else "room empty")
    sch = "schedule unknown" if pd.isna(r.Scheduled) else ("scheduled use" if r.Scheduled == 1 else "no scheduled use")
    return (f"Used {r.Power_kW:.2f} kW; about {r.Expected_Power_kW:.2f} kW was expected ({r.Residual_kW:+.2f} kW). "
            f"{occ.capitalize()}, {sch}, equipment {r.Equipment_Status}.")


#sidebar
st.sidebar.title("⚡ Room Energy Audit")
available = {k: v for k, v in SAMPLES.items() if (BASE / v).exists()}
source_kind = st.sidebar.radio("Data", ["Project sample data", "Upload my own CSV"])
source_file = None
if source_kind == "Project sample data":
    if not available:
        st.error("No project CSV files found next to dash.py. Copy them into the same folder.")
        st.stop()
    sample_name = st.sidebar.selectbox("Sample", list(available))
    source_file = available[sample_name]
    raw = read_csv(BASE / source_file)
    data_label = "synthetic sample data"
else:
    up = st.sidebar.file_uploader("CSV file", type="csv")
    st.sidebar.caption("Needs: " + ", ".join(REQUIRED) + ". Optional: Room_ID, Occupancy, Scheduled, "
                       "Equipment_Status, Data_Quality, Energy_kWh.")
    if up is None:
        st.info("Upload a CSV in the sidebar, or switch to the project sample data.")
        st.stop()
    raw = pd.read_csv(up)
    data_label = "your uploaded data"

if "Room_ID" in raw.columns and raw["Room_ID"].nunique() > 1:
    room = st.sidebar.selectbox("Room", sorted(raw["Room_ID"].astype(str).unique()))
    raw = raw[raw["Room_ID"].astype(str) == room]
elif "Room_ID" in raw.columns:
    room = str(raw["Room_ID"].iloc[0])
else:
    room = "Room"

with st.sidebar.expander("⚙️ Advanced (technical users)"):
    threshold = st.slider("Anomaly threshold (kW)", 0.05, 1.00, DEFAULT_THRESHOLD, 0.05,
                          help="A reading is 'unusual' when it differs from expected by more than this. Project default: 0.30.")
    tech = st.toggle("Show technical wording and details", value=False)

try:
    df = analyse(raw, threshold)
except ValueError as e:
    st.error(str(e))
    st.stop()
if df.empty:
    st.warning("No usable readings in this data.")
    st.stop()

#numbers
step_h = df["Time"].diff().dt.total_seconds().median() / 3600 if len(df) > 1 else 1.0
step_h = step_h if pd.notna(step_h) and step_h > 0 else 1.0
used = df["Energy_kWh"].sum()
expected_kwh = (df["Expected_Power_kW"] * step_h).sum()
extra = (df["Residual_kW"].clip(lower=0) * step_h).sum()
empty = df[df["Context_Mismatch"]]
standby = empty["Energy_kWh"].sum()
counts = df["Decision_State"].value_counts()
n_check, n_unusual, n_nodata = (int(counts.get(k, 0)) for k in ["POTENTIALLY_ACTIONABLE", "UNUSUAL", "INSUFFICIENT_EVIDENCE"])

queue = df[df["Decision_State"] != "NORMAL"].copy()
queue["_p"] = queue["Decision_State"].map(PRIORITY)
queue = queue.sort_values(["_p", "Actionability_Score", "Residual_kW"], ascending=[True, False, False],
                          key=lambda s: s.abs() if s.name == "Residual_kW" else s)

#header
st.title(f"Room energy check: {room}")
st.info(f"Demo using {data_label}. A flag means **worth reviewing**, not a confirmed fault.", icon="ℹ️")

tab_over, tab_power, tab_queue, tab_explain, tab_evidence, tab_about = st.tabs(
    ["🏠 Overview", "📈 Power & Energy", "📋 Review Queue", "🔍 Explain a Reading", "🧾 Evidence & Audit", "ℹ️ About"]
)

#OVERVIEW
with tab_over:
    if n_check:
        times = ", ".join(queue[queue.Decision_State == "POTENTIALLY_ACTIONABLE"]["Label"].head(6))
        st.error(f"🔴 **{n_check} reading(s) worth checking** at: {times}. Please look at the Review Queue.")
    elif n_unusual or n_nodata:
        st.warning(f"🟡 Nothing urgent. {n_unusual} unusual reading(s) that may be normal activity, "
                   f"{n_nodata} with not enough data.")
    else:
        st.success("✅ Everything looks normal.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Energy used", f"{used:.2f} kWh")
    c2.metric("Energy expected", f"{expected_kwh:.2f} kWh")
    c3.metric("Extra energy", f"{extra:.2f} kWh", help="Total amount by which usage went above expected.")
    c4.metric("Used while room empty", f"{standby:.2f} kWh",
              help="Energy used when nobody was in the room and nothing was scheduled (standby / phantom load).")

    st.subheader("What should be done?")
    r1, r2, r3 = st.columns(3)
    rc = df["Route"].value_counts()
    r1.metric("✅ " + ROUTE["DECIDE"], int(rc.get("DECIDE", 0)), help="Route: DECIDE (confidence 0.80 or higher)")
    r2.metric("🟡 " + ROUTE["REVIEW"], int(rc.get("REVIEW", 0)), help="Route: REVIEW (confidence 0.50 to 0.79)")
    r3.metric("⚪ " + ROUTE["ABSTAIN"], int(rc.get("ABSTAIN", 0)), help="Route: ABSTAIN (confidence below 0.50)")

    if len(queue):
        top = queue.iloc[0]
        with st.container(border=True):
            st.markdown(f"**Most important: {top.Label}, {status_text(top.Decision_State, tech)}**")
            st.write(describe(top))
            st.caption("Suggested next step: " + NEXT_STEP[top.Decision_State])

    fig = go.Figure(go.Bar(
        y=[status_text(k, tech) for k in STATUS], x=[int(counts.get(k, 0)) for k in STATUS],
        orientation="h", marker_color=[STATUS[k][2] for k in STATUS]))
    fig.update_layout(height=220, margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Number of readings",
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig)

#POWER & ENERGY
with tab_power:
    st.subheader("Actual vs expected power")
    st.caption("The grey band is the normal range. Coloured dots are readings the system wants you to notice.")

    has_ctx = [(lab, col) for lab, col in [("Someone in room", "Occupancy"), ("Scheduled", "Scheduled")]
               if df[col].notna().any()]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=df.Time, y=df.Expected_Power_kW + threshold, line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.Time, y=(df.Expected_Power_kW - threshold).clip(lower=0), line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(120,120,120,0.15)", name="Normal range",
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.Time, y=df.Expected_Power_kW, mode="lines", name="Expected power",
                             line=dict(color="#888", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.Time, y=df.Power_kW, mode="lines+markers", name="Actual power",
                             line=dict(color="#005580", width=2)), row=1, col=1)
    for code in ["UNUSUAL", "POTENTIALLY_ACTIONABLE", "INSUFFICIENT_EVIDENCE"]:
        sub = df[df.Decision_State == code]
        if len(sub):
            fig.add_trace(go.Scatter(x=sub.Time, y=sub.Power_kW, mode="markers", name=STATUS[code][0],
                                     marker=dict(size=14, color=STATUS[code][2], line=dict(width=2, color="white"))),
                          row=1, col=1)
    rows, z = [], []
    for lab, col in has_ctx:
        rows.append(lab)
        z.append((df[col].fillna(0) > 0).astype(int).tolist())
    if df["Equipment_Status"].astype(str).str.upper().isin(["ON", "OFF"]).any():
        rows.append("Equipment on")
        z.append(df["Equipment_Status"].astype(str).str.upper().eq("ON").astype(int).tolist())
    if rows:
        fig.add_trace(go.Heatmap(x=df.Time, y=rows, z=z, showscale=False, zmin=0, zmax=1, xgap=2, ygap=2,
                                 colorscale=[[0, "#eceff1"], [1, "#455a64"]], hoverinfo="x+y"), row=2, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_layout(height=520, margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig)
    if rows:
        st.caption("Bottom strip: dark = yes, light = no. It shows why a high reading may be perfectly normal.")

    st.subheader("Energy use by hour")
    width = step_h * 3600 * 1000 * 0.8
    bar = go.Figure(go.Bar(x=df.Time, y=df.Energy_kWh, width=width,
                           marker_color=df.Decision_State.map(lambda k: STATUS[k][2]),
                           customdata=df.Decision_State.map(lambda k: STATUS[k][0]),
                           hovertemplate="%{x|%H:%M}: %{y:.2f} kWh (%{customdata})<extra></extra>"))
    bar.update_layout(height=300, margin=dict(t=10, b=10), yaxis_title="Energy (kWh)")
    st.plotly_chart(bar)
    st.caption("Bar colour matches status: green normal, yellow unusual, red worth checking, grey not enough data.")

    st.subheader("Standby energy (room empty)")
    share = 100 * standby / used if used else 0
    hrs = len(empty)
    st.write(f"During **{hrs}** reading(s) the room was empty and unscheduled. It used **{standby:.2f} kWh** "
             f"then, which is **{share:.0f}%** of the total.")
    if hrs:
        st.caption(f"Average power in those readings: {empty.Power_kW.mean():.2f} kW "
                   f"(expected {empty.Expected_Power_kW.mean():.2f} kW).")

    with st.expander("Difference from expected power (technical)", expanded=tech):
        rf = go.Figure(go.Bar(x=df.Time, y=df.Residual_kW, width=width,
                              marker_color=df.Anomaly.map({True: "#d32f2f", False: "#90a4ae"})))
        for y in (threshold, -threshold):
            rf.add_hline(y=y, line_dash="dash", line_color="#d32f2f")
        rf.update_layout(height=280, margin=dict(t=10, b=10), yaxis_title="Residual (kW)")
        st.plotly_chart(rf)
        st.caption(f"Residual = actual minus expected. Readings beyond ±{threshold:.2f} kW are flagged as anomalies.")

#REVIEW QUEUE
with tab_queue:
    st.subheader("Readings to look at")
    only_check = st.checkbox("Only show 'Worth checking'", value=False)
    q = queue[queue.Decision_State == "POTENTIALLY_ACTIONABLE"] if only_check else queue
    if q.empty:
        st.success("Nothing in the queue. All readings are normal.")
    else:
        st.caption("Most important first.")
        for _, r in q.head(15).iterrows():
            with st.container(border=True):
                a, b = st.columns([1, 4])
                a.markdown(f"### {r.Label}")
                a.write(status_text(r.Decision_State, tech))
                b.write(describe(r))
                b.markdown(f"**Next step:** {NEXT_STEP[r.Decision_State]}")
                b.caption(ROUTE[r.Route] + (f" · confidence {r.Confidence:.2f} · score {r.Actionability_Score:.2f}" if tech else ""))
        if len(q) > 15:
            st.caption(f"Showing 15 of {len(q)}. The full list is in the table below.")
        table = pd.DataFrame({
            "Time": q.Label, "Room": q.Room_ID, "Status": [status_text(c, tech) for c in q.Decision_State],
            "Power (kW)": q.Power_kW, "Expected (kW)": q.Expected_Power_kW,
            "What to do": q.Route.map(ROUTE), "Next step": q.Decision_State.map(NEXT_STEP),
            "Score": q.Actionability_Score, "Confidence": q.Confidence,
        })
        with st.expander("Full table"):
            st.dataframe(table, hide_index=True)
        st.download_button("⬇️ Download review queue (CSV)", table.to_csv(index=False), "review_queue.csv", "text/csv")

#EXPLAIN
with tab_explain:
    st.subheader("Why did this reading get its status?")
    default = int(queue.index[0]) if len(queue) else 0
    idx = st.selectbox("Pick a reading", list(df.index), index=list(df.index).index(default),
                       format_func=lambda i: f"{df.Label[i]}  |  {status_text(df.Decision_State[i])}",
                       key=f"explain_{room}_{threshold}_{source_file}")
    r = df.loc[idx]
    st.markdown(f"### {status_text(r.Decision_State, tech)}")
    st.write(describe(r))
    st.write("**Why:** " + r.Reason)
    e1, e2, e3 = st.columns(3)
    e1.metric("What to do", ROUTE[r.Route])
    e2.metric("Confidence", f"{r.Confidence:.0%}")
    e3.metric("Next step", NEXT_STEP[r.Decision_State].split(";")[0].rstrip("."))
    st.caption("This is a flag for a person to review, not a confirmed fault.")

    check = pd.DataFrame([
        ["Power far from expected?", "Yes" if r.Anomaly else "No", f"{r.Residual_kW:+.2f} kW vs ±{threshold:.2f} allowed"],
        ["Someone in the room?", "unknown" if pd.isna(r.Occupancy) else ("Yes" if r.Occupancy > 0 else "No"), ""],
        ["Scheduled to be in use?", "unknown" if pd.isna(r.Scheduled) else ("Yes" if r.Scheduled == 1 else "No"), ""],
        ["Equipment", str(r.Equipment_Status), ""],
        ["Data reliable?", "Yes" if r.Quality_OK else "No", str(r.Data_Quality)],
    ], columns=["Check", "Answer", "Detail"])
    st.dataframe(check, hide_index=True)

    with st.expander("Technical details", expanded=tech):
        score = pd.DataFrame([
            ["Anomaly (power beyond threshold)", bool(r.Anomaly), r.Pts_Anomaly],
            ["Context mismatch (empty and unscheduled)", bool(r.Context_Mismatch), r.Pts_Context],
            ["Persistent (this or previous reading anomalous)", bool(r.Persistent), r.Pts_Persistent],
            ["Good data quality", bool(r.Quality_OK), r.Pts_Quality],
        ], columns=["Factor", "Applies", "Points"])
        st.dataframe(score, hide_index=True)
        st.write(f"**Actionability score:** {r.Actionability_Score:.2f} (needs 0.80 or more, in an empty and unscheduled room, to be 'Potentially actionable')")
        st.write(f"**Decision state:** `{r.Decision_State}`  ·  **Route:** `{r.Route}`  ·  **Confidence:** {r.Confidence:.2f}")
        st.write(f"**Evidence:** `{r.Evidence_Bundle}`")
        st.caption(f"{r.Measurement_ID} · {r.Baseline_ID} · {r.Rule_Version}")

#EVIDENCE & AUDIT
with tab_evidence:
    st.subheader("Evidence log")
    st.caption("Every reading is stored with the facts behind its status so it can be checked later.")
    flagged_only = st.checkbox("Only flagged readings", value=True)
    ev = df[df.Decision_State != "NORMAL"] if flagged_only else df
    evt = pd.DataFrame({
        "Measurement": ev.Measurement_ID, "Time": ev.Label,
        "Status": [status_text(c, tech) for c in ev.Decision_State],
        "Route": ev.Route if tech else ev.Route.map(ROUTE),
        "Evidence": ev.Evidence_Bundle, "Baseline": ev.Baseline_ID, "Rule": ev.Rule_Version,
    })
    if not tech:
        evt = evt.drop(columns=["Baseline", "Rule"])
    st.dataframe(evt, hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Evidence log (CSV)", evt.to_csv(index=False), "evidence_log.csv", "text/csv")
    d2.download_button("⬇️ Full results (CSV)", df.drop(columns=["Time"]).to_csv(index=False), "full_results.csv", "text/csv")

    saved = BASE / "confidence_routing_output.csv"
    if source_file == "energy_data_graph_based.csv" and abs(threshold - DEFAULT_THRESHOLD) < 1e-9 and saved.exists():
        sv = read_csv(saved)
        ok = len(sv) == len(df) and (sv.Decision_State.values == df.Decision_State.values).all() \
            and (sv.Route.values == df.Route.values).all()
        with st.expander("Check against saved project output", expanded=tech):
            (st.success if ok else st.warning)(
                "Dashboard results match confidence_routing_output.csv." if ok else
                "Dashboard results differ from confidence_routing_output.csv. Re-run the project scripts.")

    st.subheader("Example audit cases")
    ap = BASE / "sample_audit_cases.csv"
    if ap.exists():
        for _, c in read_csv(ap).iterrows():
            with st.container(border=True):
                st.markdown(f"**{c.Case_ID}: {c.Context}**")
                st.write(f"{c.Diagnosis}. Suggested action: {c.Recommendation}.")
                st.caption(f"Evidence: {c.Evidence} · Expected state: {c.Expected_State}")
    else:
        st.info("sample_audit_cases.csv not found.")
    st.caption("Confirmed actionable (Case D) needs independent evidence, such as a fault record. "
               "Power data alone can never confirm a fault, which is why the system only says 'worth checking'.")

#ABOUT
with tab_about:
    st.subheader("How it works")
    st.markdown(
        """
1. **Compare** actual power with the power expected for that room and hour.
2. **Add context:** is anyone in the room, is it scheduled, is the equipment on, is the data reliable?
3. **Score** how likely the reading is to be wasted energy, then give it a status:
   Normal, Unusual, Worth checking, or Not enough data.
4. **Decide how sure we are:** clear enough to act on, needs a person to look, or can't tell.
5. **Keep the evidence** behind every status so it can be audited.
"""
    )
    st.warning("This is a proof of concept on synthetic data. It flags readings for review and never confirms a fault.")
    with st.expander("Technical summary", expanded=tech):
        st.markdown(
            f"""
- **Anomaly:** |actual − expected| > **{threshold:.2f} kW** (project default 0.30).
- **Actionability score:** +0.40 anomaly, +0.30 empty and unscheduled, +0.20 persistent, +0.10 good data.
- **States:** `NORMAL`, `UNUSUAL`, `POTENTIALLY_ACTIONABLE` (score ≥ 0.80, empty, unscheduled), `INSUFFICIENT_EVIDENCE`.
- **Confidence:** 0.90 normal, 0.70 unusual, 0.85 potentially actionable, 0.25 poor data. **Routes:** ≥ 0.80 DECIDE, ≥ 0.50 REVIEW, else ABSTAIN.
- **Uploaded data:** if occupancy or schedule is missing, readings are treated as insufficient evidence.
"""
        )
