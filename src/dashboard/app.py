"""Step 6 - Live Streamlit dashboard.

Run:  streamlit run src/dashboard/app.py

Three tabs: Live monitor (auto-refreshing), Model performance, API performance.
Live sections use st.fragment(run_every=...), so only they re-render - not the whole page."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # so `src` imports work

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config, db  # noqa: E402

st.set_page_config(page_title="FraudPulse", page_icon="🛡️", layout="wide")

DB = config.db_path()
BLUE, RED, GREY = "#2a6fdb", "#d64545", "#9aa0a6"
CHART_MARGIN = dict(l=10, r=10, t=50, b=10)

# ---------------- sidebar ----------------
st.sidebar.title("🛡️ FraudPulse")
st.sidebar.caption("Real-time fraud scoring with explainable AI alerts")
refresh = st.sidebar.slider("Refresh every (s)", 1, 10, 2)
bucket = st.sidebar.selectbox("Fraud-rate time bucket", ["10s", "30s", "1min", "5min"], index=1)
feed_rows = st.sidebar.slider("Live feed rows", 10, 100, 20)
alerts_shown = st.sidebar.slider("Alerts shown", 2, 10, 4)
st.sidebar.divider()
st.sidebar.caption("Start the API and the simulator, then watch the Live monitor update.")

st.title("FraudPulse")
st.caption("Real-time credit-card fraud detection · XGBoost + FastAPI + SHAP-grounded LLM explanations")


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:.1%}"


def alert_card(r) -> None:
    truth = {1.0: "✅ confirmed fraud", 0.0: "⚠️ false positive"}.get(r.actual_label, "")
    with st.container(border=True):
        st.markdown(f"**{r.transaction_id}** · ${r.amount:,.2f} · "
                    f"fraud probability **{r.fraud_probability:.1%}** · {truth}")
        if r.explanation:
            st.write(r.explanation)
        else:
            st.caption("⏳ Explanation pending (rate-limited queue)…")
        feats = json.loads(r.top_features or "[]")
        src = {"llm": "Groq LLM", "template": "template"}.get(r.explanation_source, "—")
        st.caption(f"Source: {src} · Top SHAP drivers: " + ", ".join(
            f"{f['feature']} ({f['contribution']:+.2f})" for f in feats))


tab_live, tab_model, tab_api = st.tabs(["📡 Live monitor", "📈 Model performance",
                                        "⚡ API performance"])

# ======================= LIVE MONITOR =======================
with tab_live:
    @st.fragment(run_every=refresh)
    def live_section():
        df = db.read_df(DB, "SELECT * FROM transactions ORDER BY id DESC LIMIT 20000")
        if df.empty:
            st.info("No transactions yet. Run: "
                    "`python -m src.stream.simulator --fraud-ratio 0.02 --reset-db`")
            return
        df["scored_at"] = pd.to_datetime(df["scored_at"], utc=True, format="ISO8601")

        flagged = df[df.is_flagged == 1]
        labelled = flagged.dropna(subset=["actual_label"])
        live_precision = labelled.actual_label.mean() if len(labelled) else float("nan")
        actual = df.dropna(subset=["actual_label"])
        n_fraud = int((actual.actual_label == 1).sum())
        caught = int(((actual.actual_label == 1) & (actual.is_flagged == 1)).sum())
        live_recall = caught / n_fraud if n_fraud else float("nan")

        # KPIs: two rows of three so labels never wrap
        r1 = st.columns(3)
        r1[0].metric("Transactions scored", f"{len(df):,}")
        r1[1].metric("Alerts raised", f"{len(flagged):,}")
        r1[2].metric("Alert rate", pct(len(flagged) / len(df)))
        r2 = st.columns(3)
        r2[0].metric("Live precision", pct(live_precision),
                     help="Share of alerts that were truly fraud (labels from the replayed data)")
        r2[1].metric("Live recall", pct(live_recall),
                     help="Share of actual frauds that raised an alert")
        r2[2].metric("p95 round-trip", f"{df.roundtrip_ms.quantile(0.95):.1f} ms",
                     help="Client-observed latency of POST /score")

        # Fraud rate over time
        ts = (df.set_index("scored_at").sort_index()
              .resample(bucket).agg(total=("is_flagged", "size"), flagged=("is_flagged", "sum"),
                                    actual=("actual_label", "sum")))
        ts = ts[ts.total > 0]
        ts["flag_rate"] = ts.flagged / ts.total * 100
        ts["actual_rate"] = ts.actual / ts.total * 100
        fig = go.Figure()
        fig.add_scatter(x=ts.index, y=ts.actual_rate, name="Actual fraud (labels)", mode="lines",
                        line=dict(color=GREY, width=6), opacity=0.45)
        fig.add_scatter(x=ts.index, y=ts.flag_rate, name="Alerts raised by model",
                        mode="lines+markers", line=dict(color=RED, width=2))
        fig.update_layout(title=f"Fraud rate over time (% of transactions per {bucket})",
                          height=300, margin=CHART_MARGIN, yaxis_title="%",
                          yaxis=dict(rangemode="tozero"),
                          legend=dict(orientation="h", y=1.15, x=1, xanchor="right"))
        st.plotly_chart(fig, width="stretch")

        left, right = st.columns([1, 1.15], gap="large")
        with left:
            st.subheader("Live transaction feed")
            feed = df.head(feed_rows)[["scored_at", "transaction_id", "amount",
                                       "fraud_probability", "is_flagged"]].copy()
            feed["scored_at"] = feed.scored_at.dt.strftime("%H:%M:%S")
            feed["is_flagged"] = feed.is_flagged.map({1: "🚨 Alert", 0: "OK"})
            st.dataframe(feed, hide_index=True, width="stretch", height=460,
                         column_config={
                             "scored_at": "Time",
                             "transaction_id": "Transaction",
                             "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                             "fraud_probability": st.column_config.ProgressColumn(
                                 "Fraud prob.", min_value=0, max_value=1, format="%.3f"),
                             "is_flagged": "Status"})
        with right:
            st.subheader(f"🚨 Fraud alerts ({len(flagged)})")
            if flagged.empty:
                st.write("No alerts yet.")
            for _, r in flagged.head(alerts_shown).iterrows():
                alert_card(r)
            older = flagged.iloc[alerts_shown:]
            if len(older):
                with st.expander(f"Show {len(older)} older alerts"):
                    for _, r in older.head(50).iterrows():
                        alert_card(r)

    live_section()

# ======================= MODEL PERFORMANCE =======================
with tab_model:
    metrics = load_json(config.model_dir() / "metrics.json")
    if not metrics:
        st.info("Train the model first: `python -m src.train`")
    else:
        xg, lr = metrics["models"]["xgboost"], metrics["models"]["logistic_regression"]
        c = st.columns(4)
        c[0].metric("PR-AUC", f"{xg['pr_auc']:.3f}", f"{xg['pr_auc'] - lr['pr_auc']:+.3f} vs baseline")
        c[1].metric("Precision", pct(xg["precision"]))
        c[2].metric("Recall", pct(xg["recall"]))
        c[3].metric("F1", f"{xg['f1']:.3f}")
        st.caption(f"XGBoost on the held-out test set: {metrics['test_rows']:,} transactions, "
                   f"{metrics['test_frauds']} frauds. Predicting 'always legit' would score "
                   f"{metrics['naive_always_legit_accuracy']:.2%} accuracy and catch nothing, "
                   f"so PR-AUC, precision and recall are the headline metrics.")

        names = {"logistic_regression": "Logistic Regression (baseline)", "xgboost": "XGBoost"}
        table = pd.DataFrame([{
            "Model": names.get(k, k), "Threshold": f"{v['threshold']:.4f}",
            "Precision": pct(v["precision"]), "Recall": pct(v["recall"]),
            "F1": f"{v['f1']:.3f}", "PR-AUC": f"{v['pr_auc']:.3f}", "ROC-AUC": f"{v['roc_auc']:.3f}",
            "Caught / missed": f"{v['tp']} / {v['fn']}", "False alarms": v["fp"],
        } for k, v in metrics["models"].items()])
        st.dataframe(table, hide_index=True, width="stretch")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            fig = go.Figure()
            for name, color in (("xgboost", BLUE), ("logistic_regression", GREY)):
                pr = metrics["pr_curves"][name]
                fig.add_scatter(x=pr["recall"], y=pr["precision"], mode="lines",
                                name=f"{names[name]} · PR-AUC {metrics['models'][name]['pr_auc']:.3f}",
                                line=dict(color=color, width=2.5))
            fig.add_scatter(x=[xg["recall"]], y=[xg["precision"]], mode="markers",
                            name="XGBoost at chosen threshold",
                            marker=dict(color=RED, size=11, symbol="diamond"))
            fig.update_layout(title="Precision–Recall curve (test set)", height=400,
                              xaxis=dict(title="Recall", range=[0, 1.02]),
                              yaxis=dict(title="Precision", range=[0, 1.05]),
                              margin=CHART_MARGIN, legend=dict(x=0.02, y=0.05))
            st.plotly_chart(fig, width="stretch")
        with c2:
            fi = metrics.get("feature_importance", {})
            if fi:
                items = sorted(fi.items(), key=lambda kv: kv[1])
                fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items],
                                       orientation="h", marker_color=BLUE))
                fig.update_layout(title="Top features (XGBoost gain)", height=400,
                                  margin=CHART_MARGIN, xaxis_title="Gain")
                st.plotly_chart(fig, width="stretch")
        st.caption("V1–V28 are anonymized PCA components in the public dataset, so features are "
                   "referred to as 'behaviour signals'.")

# ======================= API PERFORMANCE =======================
with tab_api:
    @st.fragment(run_every=10)   # picks up new load-test runs without a page reload
    def loadtest_section():
        lt = load_json(config.results_dir() / "loadtest_latest.json")
        if not lt:
            st.info("Run `python -m src.loadtest` to measure throughput and latency.")
            return
        L = lt["latency_ms"]
        c = st.columns(4)
        c[0].metric("Throughput", f"{lt['throughput_rps']:.0f} req/s")
        c[1].metric("p50 latency", f"{L['p50']:.1f} ms")
        c[2].metric("p95 latency", f"{L['p95']:.1f} ms")
        c[3].metric("Errors", f"{lt['errors']} / {lt['requests']:,}")
        st.caption(f"Latest run: {lt['requests']:,} requests · {lt['concurrency']} concurrent "
                   f"clients · {lt.get('workers', 1)} uvicorn worker(s) · client and server on the "
                   f"same machine")

        hist_path = config.results_dir() / "loadtest_history.jsonl"
        if hist_path.exists():
            rows = [json.loads(x) for x in hist_path.read_text().splitlines() if x.strip()]
            hist = pd.DataFrame([{
                "Time (UTC)": r["timestamp"].replace("T", " ")[:19],
                "Clients": r["concurrency"], "Workers": r.get("workers"),  # older runs: blank
                "Requests": r["requests"], "Req/s": r["throughput_rps"],
                "p50 ms": r["latency_ms"]["p50"], "p95 ms": r["latency_ms"]["p95"],
                "p99 ms": r["latency_ms"]["p99"], "Errors": r["errors"]} for r in rows])
            hist["Workers"] = hist["Workers"].astype("Int64")   # whole numbers, blanks allowed
            st.subheader("All load-test runs")
            st.dataframe(hist.iloc[::-1], hide_index=True, width="stretch")

    loadtest_section()
