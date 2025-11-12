# app_streamlit.py
# Single-file Streamlit app: EDA, Optimization, Inline ML (demo), and an AI-style chatbot (local retrieval)
# Put this file at the repo root. Use requirements.txt with streamlit,pandas,numpy,plotly,scikit-learn,joblib

import os
import io
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# plotting
import plotly.express as px

# ML
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib

# ------------------------ Page config & styles ------------------------
st.set_page_config(page_title="Week 9 - EC2 & S3 EDA + Optimization + Chatbot", layout="wide")

EC2_COLOR = "#1f77b4"  # blue-ish
S3_COLOR = "#ff7f0e"   # orange-ish
ACCENT = "#2dd4bf"

st.markdown(
    f"""
    <style>
    .block-container{{padding-top:1rem; padding-left:1.2rem; padding-right:1.2rem;}}
    .ec2-title h3, .ec2-title h4 {{ color: {EC2_COLOR}; margin: 0; }}
    .s3-title h3, .s3-title h4  {{ color: {S3_COLOR}; margin: 0; }}
    .metric {{"font-size": 26px; color: #fff;}}
    .small-muted {{ color: #aaa; font-size:12px; }}
    .chat-user {{ background:#0f172a; padding:8px; border-radius:8px; margin:6px 0; }}
    .chat-bot  {{ background:#072a2a; color:#e6fffa; padding:8px; border-radius:8px; margin:6px 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------ Column maps / helpers ------------------------
EC2_MAP = {
    "ResourceId": "InstanceId",
    "InstanceType": "InstanceType",
    "Region": "Region",
    "State": "State",
    "CPUUtilization": "CPUUtilization",
    "MemoryUtilization": "MemoryUtilization",
    "NetworkIn_Bps": "NetworkIn_Bps",
    "NetworkOut_Bps": "NetworkOut_Bps",
    "CostUSD": "CostPerHourUSD",
    "Tags": "Tags",
    "CreationDate": "LaunchTime",
    "ResourceType": "ResourceType",
}
S3_MAP = {
    "BucketName": "BucketName",
    "Region": "Region",
    "StorageClass": "StorageClass",
    "ObjectCount": "ObjectCount",
    "TotalSizeGB": "TotalSizeGB",
    "CostUSD": "MonthlyCostUSD",
    "VersionEnabled": "VersioningEnabled",
    "Encryption": "Encryption",
    "CreationDate": "CreatedDate",
    "Tags": "Tags",
}

def safe_read_csv(file_or_path, colmap, is_ec2):
    """Accept file-like (upload) or path (string); standardize columns."""
    if file_or_path is None:
        return pd.DataFrame()
    if hasattr(file_or_path, "read"):
        df = pd.read_csv(file_or_path)
    else:
        df = pd.read_csv(file_or_path)
    # rename known columns
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})
    # numeric conversions
    if is_ec2:
        for c in ["CPUUtilization","MemoryUtilization","NetworkIn_Bps","NetworkOut_Bps","CostPerHourUSD"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    else:
        for c in ["ObjectCount","TotalSizeGB","MonthlyCostUSD"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def df_info_text(df: pd.DataFrame):
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

def find_util_col_s3(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in {
        "utilizationpercent","utilization_%","usedpercent","usagepercent",
        "storageutilization","storage_utilization","utilization","used_pct","used_percent"
    }]
    return candidates[0] if candidates else None

# ------------------------ Sidebar: uploads & filters ------------------------
with st.sidebar:
    st.header("Data sources & filters")
    st.caption("Upload CSVs or commit data/ec2.csv & data/s3.csv in the repo.")
    ec2_upload = st.file_uploader("Upload EC2 CSV", type=["csv"], key="ec2_up")
    s3_upload  = st.file_uploader("Upload S3 CSV", type=["csv"], key="s3_up")
    # fallback to repo files if present
    ec2_path = ec2_upload if ec2_upload is not None else ("data/ec2.csv" if os.path.exists("data/ec2.csv") else None)
    s3_path  = s3_upload if s3_upload is not None else ("data/s3.csv"  if os.path.exists("data/s3.csv")  else None)
    st.markdown("---")
    st.subheader("View filters (left)")
    # thresholds will be available also in Optimization tab
    st.write("Note: Filters apply to all tabs")
    # placeholders for slider defaults; actual lists computed after load

# validate files exist
if ec2_path is None or s3_path is None:
    st.sidebar.error("Upload both EC2 and S3 CSVs here or add them to repo at /data/ec2.csv and /data/s3.csv")
    st.stop()

# load data
ec2_df = safe_read_csv(ec2_path, EC2_MAP, is_ec2=True)
s3_df  = safe_read_csv(s3_path, S3_MAP, is_ec2=False)

if ec2_df.empty or s3_df.empty:
    st.error("Loaded CSV appears empty or invalid. Check columns/encoding.")
    st.stop()

# basic cleaning / default columns
ec2_df = ec2_df.fillna({"InstanceType":"Unknown","Region":"Unknown","Tags":"Unknown"})
s3_df  = s3_df.fillna({"StorageClass":"Unknown","Region":"Unknown","Encryption":"Unknown","VersioningEnabled":"Unknown"})


# get filter options
ec2_regions = sorted(ec2_df["Region"].dropna().unique().tolist()) if "Region" in ec2_df.columns else []
s3_regions  = sorted(s3_df["Region"].dropna().unique().tolist()) if "Region" in s3_df.columns else []
itypes      = sorted(ec2_df["InstanceType"].dropna().unique().tolist()) if "InstanceType" in ec2_df.columns else []
sclasses    = sorted(s3_df["StorageClass"].dropna().unique().tolist()) if "StorageClass" in s3_df.columns else []

# filter inputs (left)
with st.sidebar.expander("Filter controls", expanded=True):
    picked_ec2_regions = st.multiselect("EC2 regions", ec2_regions, default=ec2_regions)
    picked_s3_regions  = st.multiselect("S3 regions", s3_regions, default=s3_regions)
    picked_types = st.multiselect("Instance types", itypes, default=itypes[:10] if len(itypes)>10 else itypes)
    picked_sclasses = st.multiselect("Storage classes", sclasses, default=sclasses)
    # cost sliders
    if "CostPerHourUSD" in ec2_df.columns:
        minc = float(ec2_df["CostPerHourUSD"].min()); maxc = float(ec2_df["CostPerHourUSD"].max())
        ec2_cost_range = st.slider("EC2 cost/hr range", minc, maxc, (minc, maxc))
    else:
        ec2_cost_range = None
    if "MonthlyCostUSD" in s3_df.columns:
        mins = float(s3_df["MonthlyCostUSD"].min()); maxs = float(s3_df["MonthlyCostUSD"].max())
        s3_cost_range = st.slider("S3 monthly cost range", mins, maxs, (mins, maxs))
    else:
        s3_cost_range = None

# apply filters function
def apply_filters(df_ec2, df_s3):
    e = df_ec2.copy()
    s = df_s3.copy()
    if picked_ec2_regions:
        e = e[e["Region"].isin(picked_ec2_regions)]
    if picked_types:
        e = e[e["InstanceType"].isin(picked_types)]
    if ec2_cost_range is not None and "CostPerHourUSD" in e.columns:
        lo, hi = ec2_cost_range; e = e[(e["CostPerHourUSD"] >= lo) & (e["CostPerHourUSD"] <= hi)]
    if picked_s3_regions:
        s = s[s["Region"].isin(picked_s3_regions)]
    if picked_sclasses:
        s = s[s["StorageClass"].isin(picked_sclasses)]
    if s3_cost_range is not None and "MonthlyCostUSD" in s.columns:
        lo2, hi2 = s3_cost_range; s = s[(s["MonthlyCostUSD"] >= lo2) & (s["MonthlyCostUSD"] <= hi2)]
    return e, s

ec2_f, s3_f = apply_filters(ec2_df, s3_df)

# ------------------------ Top KPIs ------------------------
st.title("Week 9 — EC2 & S3 EDA, Optimization & Chatbot")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.subheader("EC2 rows")
    st.metric("", f"{len(ec2_f):,}")
with k2:
    st.subheader("S3 rows")
    st.metric("", f"{len(s3_f):,}")
with k3:
    st.subheader("Avg EC2 $/hr")
    avg_ec2 = ec2_f["CostPerHourUSD"].mean() if "CostPerHourUSD" in ec2_f.columns and len(ec2_f) else 0.0
    st.metric("", f"${avg_ec2:,.3f}")
with k4:
    st.subheader("Total S3 $/mo")
    tot_s3 = s3_f["MonthlyCostUSD"].sum() if "MonthlyCostUSD" in s3_f.columns and len(s3_f) else 0.0
    st.metric("", f"${tot_s3:,.2f}")

st.markdown("---")

# ------------------------ Tabs ------------------------
tab_overview, tab_ec2, tab_s3, tab_insights, tab_opt, tab_ml, tab_chat, tab_downloads = st.tabs(
    ["Overview","EC2","S3","Insights","Optimization","ML (train in-app)","Chatbot","Downloads"]
)

# ---------- Overview ----------
with tab_overview:
    st.header("Overview")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='ec2-title'><h3>EC2</h3></div>", unsafe_allow_html=True)
        st.code(df_info_text(ec2_f))
        st.write("Shape:", ec2_f.shape)
        st.write("Summary (describe):")
        st.dataframe(ec2_f.describe(include='all', datetime_is_numeric=False), use_container_width=True)
    with c2:
        st.markdown("<div class='s3-title'><h3>S3</h3></div>", unsafe_allow_html=True)
        st.code(df_info_text(s3_f))
        st.write("Shape:", s3_f.shape)
        st.write("Summary (describe):")
        st.dataframe(s3_f.describe(include='all', datetime_is_numeric=False), use_container_width=True)

    st.markdown("#### Sample rows")
    h1, h2 = st.columns(2)
    with h1:
        st.dataframe(ec2_f.head(10), use_container_width=True)
    with h2:
        st.dataframe(s3_f.head(10), use_container_width=True)

# ---------- EC2 tab ----------
with tab_ec2:
    st.markdown("<div class='ec2-title'><h3>EC2 Visuals & Table</h3></div>", unsafe_allow_html=True)
    if "CPUUtilization" in ec2_f.columns and len(ec2_f):
        fig = px.histogram(ec2_f, x="CPUUtilization", nbins=30, title="EC2 CPU Utilization", color_discrete_sequence=[EC2_COLOR])
        st.plotly_chart(fig, use_container_width=True)
    if {"CPUUtilization","CostPerHourUSD"}.issubset(ec2_f.columns) and len(ec2_f):
        fig2 = px.scatter(ec2_f, x="CPUUtilization", y="CostPerHourUSD", title="EC2: CPU vs Cost",
                          hover_data=["InstanceId","InstanceType","Region"], color_discrete_sequence=[EC2_COLOR])
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("#### EC2 table (filtered)")
    st.dataframe(ec2_f, use_container_width=True)

# ---------- S3 tab ----------
with tab_s3:
    st.markdown("<div class='s3-title'><h3>S3 Visuals & Table</h3></div>", unsafe_allow_html=True)
    if {"Region","TotalSizeGB"}.issubset(s3_f.columns) and len(s3_f):
        by_region = s3_f.groupby("Region")["TotalSizeGB"].sum().reset_index().sort_values("TotalSizeGB", ascending=False)
        fig = px.bar(by_region, x="Region", y="TotalSizeGB", title="Total S3 Storage by Region", color_discrete_sequence=[S3_COLOR])
        st.plotly_chart(fig, use_container_width=True)
    if {"TotalSizeGB","MonthlyCostUSD"}.issubset(s3_f.columns) and len(s3_f):
        size_col = "ObjectCount" if "ObjectCount" in s3_f.columns else "TotalSizeGB"
        fig3 = px.scatter(s3_f, x="TotalSizeGB", y="MonthlyCostUSD", size=size_col, title="S3 Cost vs Storage (bubble≈objects)",
                          hover_data=["BucketName","Region","StorageClass"], color_discrete_sequence=[S3_COLOR])
        st.plotly_chart(fig3, use_container_width=True)
    st.markdown("#### S3 table (filtered)")
    st.dataframe(s3_f, use_container_width=True)

# ---------- Insights ----------
with tab_insights:
    st.header("Insights: Top-5 and Regional Metrics")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 5 Most Expensive EC2")
        if "CostPerHourUSD" in ec2_f.columns and len(ec2_f):
            st.dataframe(ec2_f.sort_values("CostPerHourUSD", ascending=False).head(5), use_container_width=True)
    with c2:
        st.subheader("Top 5 Largest S3")
        if "TotalSizeGB" in s3_f.columns and len(s3_f):
            st.dataframe(s3_f.sort_values("TotalSizeGB", ascending=False).head(5), use_container_width=True)
    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Avg EC2 cost per region")
        if {"Region","CostPerHourUSD"}.issubset(ec2_f.columns):
            st.dataframe(ec2_f.groupby("Region")["CostPerHourUSD"].mean().sort_values(ascending=False).to_frame("AvgCostPerHr"), use_container_width=True)
    with c4:
        st.subheader("Total S3 storage per region")
        if {"Region","TotalSizeGB"}.issubset(s3_f.columns):
            st.dataframe(s3_f.groupby("Region")["TotalSizeGB"].sum().sort_values(ascending=False).to_frame("TotalSizeGB"), use_container_width=True)

# ---------- Optimization tab ----------
with tab_opt:
    st.header("Optimization (interactive thresholds & color sections)")

    # Layout: colored headers
    st.markdown("<div class='ec2-title'><h4>EC2 (blue section)</h4></div>", unsafe_allow_html=True)
    st.caption("Only under-utilized EC2 (CPU below threshold) shown here.")

    col1, col2 = st.columns([2,1])
    with col1:
        cpu_thresh = st.slider("EC2 under-utilized CPU threshold (%)", 1, 50, 20)
        st.markdown("**Show instances with CPU less than:** " + str(cpu_thresh) + "%")
    with col2:
        # hours per month impacted (used to compute monthly saving if you want partial savings)
        hours_month = st.slider("EC2 hours per month affected", 0, 730, 730)
        st.caption("If you stop instance, hours per month you save (default 730 hrs)")

    # filter EC2 under threshold
    ec2_under = ec2_f.copy()
    if "CPUUtilization" in ec2_under.columns:
        ec2_under = ec2_under[ec2_under["CPUUtilization"] < cpu_thresh].copy()
    else:
        ec2_under = ec2_under.iloc[0:0].copy()

    # compute savings per instance (assume full elimination by default)
    if "CostPerHourUSD" in ec2_under.columns:
        ec2_under["SavingPerHourUSD"] = ec2_under["CostPerHourUSD"]  # full stop -> saving equals cost/hr
        ec2_under["SavingPerMonthUSD"] = ec2_under["SavingPerHourUSD"] * hours_month
    else:
        ec2_under["SavingPerHourUSD"] = 0.0
        ec2_under["SavingPerMonthUSD"] = 0.0

    st.markdown("#### EC2 candidates (CPU < threshold)")
    st.dataframe(ec2_under[["InstanceId","Region","InstanceType","CPUUtilization","CostPerHourUSD","SavingPerHourUSD","SavingPerMonthUSD"]].sort_values("SavingPerMonthUSD", ascending=False), use_container_width=True)

    # EC2 graph
    if not ec2_under.empty:
        fig_ec2 = px.scatter(ec2_under, x="CPUUtilization", y="CostPerHourUSD", size="SavingPerMonthUSD",
                             hover_data=["InstanceId","InstanceType","Region"], title="EC2 under-utilized: CPU vs cost (size ≈ monthly saving)",
                             color_discrete_sequence=[EC2_COLOR])
        st.plotly_chart(fig_ec2, use_container_width=True)

    # EC2 totals
    total_ec2_saving = float(ec2_under["SavingPerMonthUSD"].sum()) if not ec2_under.empty else 0.0
    st.metric("EC2 total saving / month (candidates)", f"${total_ec2_saving:,.2f}")

    st.markdown("---")
    st.markdown("<div class='s3-title'><h4>S3 (orange section)</h4></div>", unsafe_allow_html=True)
    st.caption("Under-utilized S3 buckets are those under a utilization threshold (< %).")

    # S3 threshold controls
    colA, colB, colC = st.columns([1.5,1.5,1])
    with colA:
        s3_util_thresh = st.slider("S3 utilization threshold (%)", 1, 50, 20)
    with colB:
        s3_month_cost_thresh = st.number_input("Minimum monthly cost to consider (USD)", value=float(0))
    with colC:
        lifecycle_saving_pct = st.slider("Estimated lifecycle move saving (%)", 0, 100, 40)

    # determine utilization percent column or estimate
    util_col = find_util_col_s3(s3_f)
    s3_eval = s3_f.copy()
    if util_col:
        s3_eval["UtilizationPercent"] = pd.to_numeric(s3_eval[util_col], errors="coerce")
        # if values look like 0-1, convert
        if s3_eval["UtilizationPercent"].dropna().between(0,1).mean() > 0.6:
            s3_eval["UtilizationPercent"] = s3_eval["UtilizationPercent"] * 100.0
    else:
        # estimate per-region 95th percentile total size and compute utilization percent
        if "TotalSizeGB" not in s3_eval.columns:
            s3_eval["UtilizationPercent"] = 0
        else:
            if "Region" not in s3_eval.columns:
                s3_eval["Region"] = "ALL"
            p95 = s3_eval.groupby("Region")["TotalSizeGB"].quantile(0.95).rename("SizeP95").reset_index()
            s3_eval = s3_eval.merge(p95, on="Region", how="left")
            s3_eval["UtilizationPercent"] = s3_eval.apply(lambda r: (r["TotalSizeGB"] / r["SizeP95"] * 100.0) if r["SizeP95"] > 0 else 0, axis=1)

    # underutilized selection
    s3_under = s3_eval[(s3_eval["UtilizationPercent"] < s3_util_thresh)].copy()
    if "MonthlyCostUSD" in s3_under.columns:
        # compute saving as monthly cost * lifecycle_saving_pct/100 (if moving to IA/Glacier), or full cost if you delete (we assume lifecycle)
        s3_under["SavingPerMonthUSD"] = s3_under["MonthlyCostUSD"] * (lifecycle_saving_pct / 100.0)
        # allow cost threshold
        if s3_month_cost_thresh and s3_month_cost_thresh > 0:
            s3_under = s3_under[s3_under["MonthlyCostUSD"] >= s3_month_cost_thresh]
    else:
        s3_under["SavingPerMonthUSD"] = 0.0

    st.markdown("#### S3 candidates (under-utilized)")
    cols_show = ["BucketName","Region","StorageClass","TotalSizeGB","UtilizationPercent","MonthlyCostUSD","SavingPerMonthUSD"]
    cols_show = [c for c in cols_show if c in s3_under.columns]
    st.dataframe(s3_under[cols_show].sort_values("SavingPerMonthUSD", ascending=False), use_container_width=True)

    # S3 graph
    if not s3_under.empty and "MonthlyCostUSD" in s3_under.columns:
        fig_s3 = px.scatter(s3_under, x="TotalSizeGB", y="MonthlyCostUSD", size="SavingPerMonthUSD",
                            hover_data=["BucketName","Region","StorageClass"], title="S3 under-utilized: Size vs Cost (size ≈ saving)",
                            color_discrete_sequence=[S3_COLOR])
        st.plotly_chart(fig_s3, use_container_width=True)

    total_s3_saving = float(s3_under["SavingPerMonthUSD"].sum()) if not s3_under.empty else 0.0
    st.metric("S3 total saving / month (candidates)", f"${total_s3_saving:,.2f}")

    st.markdown("---")
    grand_total = total_ec2_saving + total_s3_saving
    st.subheader("Grand Total Monthly Savings (EC2 + S3 candidates)")
    st.metric("Grand total saving per month", f"${grand_total:,.2f}")

    # Recommendations text (auto-generated)
    st.markdown("### Recommendations (auto-generated)")
    recs = []
    for _, r in ec2_under.sort_values("SavingPerMonthUSD", ascending=False).head(30).iterrows():
        inst = r.get("InstanceId","<id>")
        cpu = r.get("CPUUtilization", np.nan)
        cost = r.get("CostPerHourUSD", 0.0)
        recs.append(f"EC2 {inst} ({r.get('Region','')}) CPU={cpu:.1f}% Cost/hr=${cost:.3f} -> Suggest: STOP or DOWNSIZE or SCHEDULE stop-start; est saving/mo ${r.get('SavingPerMonthUSD',0):.2f}")
    for _, r in s3_under.sort_values("SavingPerMonthUSD", ascending=False).head(30).iterrows():
        b = r.get("BucketName","<name>")
        util = r.get("UtilizationPercent", np.nan)
        cost = r.get("MonthlyCostUSD",0.0)
        recs.append(f"S3 {b} ({r.get('Region','')}) util={util:.1f}% Cost/mo=${cost:.2f} -> Suggest: LIFECYCLE to IA/GLACIER, EXPIRATION of old versions; est saving/mo ${r.get('SavingPerMonthUSD',0):.2f}")
    if recs:
        st.write("• " + "  \n• ".join(recs))
    else:
        st.info("No recommendations generated (no candidates).")

# ---------- ML tab: Inline training (demo) ----------
with tab_ml:
    st.header("ML: Inline demo training (EC2 idle classifier)")

    st.markdown("This trains a small demo model inside the app using the current filtered EC2 table. **This is for demo only**. For production, train offline and commit model artifact.")
    colA, colB = st.columns([2,1])
    with colA:
        test_size = st.slider("Validation split (%)", 5, 40, 20)
        random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
    with colB:
        run_train = st.button("Train model now (demo)")

    model_obj = None
    if run_train:
        if ec2_f.empty:
            st.error("No EC2 rows after filtering.")
        else:
            # create a very simple training set: label = CPU < cpu_thresh (the same rule) -> used as pseudo-label for demo
            df_train = ec2_f.copy()
            if "CPUUtilization" not in df_train.columns:
                st.error("EC2 dataset lacks CPUUtilization column for ML demo.")
            else:
                df_train["label_idle"] = (df_train["CPUUtilization"] < cpu_thresh).astype(int)
                # simple features: CPU, cost, net in/out
                features = []
                df_train["CPUUtilization"] = df_train["CPUUtilization"].fillna(0)
                features.append("CPUUtilization")
                if "CostPerHourUSD" in df_train.columns:
                    df_train["CostPerHourUSD"] = df_train["CostPerHourUSD"].fillna(0); features.append("CostPerHourUSD")
                for col in ["NetworkIn_Bps","NetworkOut_Bps","MemoryUtilization"]:
                    if col in df_train.columns:
                        df_train[col] = df_train[col].fillna(0); features.append(col)
                X = df_train[features].fillna(0)
                y = df_train["label_idle"]
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(test_size/100.0), random_state=int(random_state))
                clf = GradientBoostingClassifier()
                clf.fit(X_train, y_train)
                try:
                    prob = clf.predict_proba(X_val)[:,1]
                    auc = None
                    # compute simple accuracy/precision etc
                    acc = clf.score(X_val, y_val)
                    st.success(f"Trained demo model — accuracy (val) = {acc:.3f}")
                except Exception:
                    st.success("Trained demo model (no probability available)")
                model_obj = clf
                # cache model in session state
                st.session_state["demo_model"] = clf

    # If model present in session state, run inference and show results
    if "demo_model" in st.session_state:
        clf = st.session_state["demo_model"]
        df_live = ec2_f.copy()
        # prepare same features
        feats = ["CPUUtilization"]
        if "CostPerHourUSD" in df_live.columns:
            feats.append("CostPerHourUSD")
        for col in ["NetworkIn_Bps","NetworkOut_Bps","MemoryUtilization"]:
            if col in df_live.columns:
                feats.append(col)
        X_live = df_live[feats].fillna(0)
        try:
            prob_live = clf.predict_proba(X_live)[:,1]
            df_live["IdleProb"] = prob_live
            df_live["SavingPerMonthUSD"] = df_live.get("CostPerHourUSD", 0) * hours_month * df_live["IdleProb"]  # weighted saving estimate
            st.markdown("### Predicted idle probability and weighted saving")
            st.dataframe(df_live[["InstanceId","Region","InstanceType","CPUUtilization","IdleProb","SavingPerMonthUSD"]].sort_values("IdleProb", ascending=False).head(50), use_container_width=True)
            # summary
            est_total = float(df_live["SavingPerMonthUSD"].sum())
            st.metric("Estimated weighted saving / month (ML)", f"${est_total:,.2f}")
        except Exception as e:
            st.error(f"Inference error: {e}")

# ---------- Chatbot (local, retrieval + rule-based) ----------
with tab_chat:
    st.header("AI-style Chatbot (local)")
    st.markdown("Ask natural questions like:  \n- `show idle ec2`  \n- `top s3 buckets`  \n- `what is the grand total saving`  \nResponse is generated from current data (no external API).")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def bot_reply(user_text: str) -> Tuple[str, Optional[pd.DataFrame]]:
        q = user_text.lower().strip()
        # quick heuristics
        if any(w in q for w in ["idle ec2", "under-utilized ec2", "underutilized ec2", "cpu <"]):
            if ec2_under.empty:
                return "No under-utilized EC2 found in current filtered view.", None
            df = ec2_under.sort_values("SavingPerMonthUSD", ascending=False).copy()
            return "Here are the top EC2 under-utilized candidates (by estimated monthly saving):", df[["InstanceId","Region","InstanceType","CPUUtilization","SavingPerMonthUSD"]].head(20)
        if any(w in q for w in ["s3 under", "under-utilized s3", "underutilized s3", "cold buckets", "low util"]):
            if s3_under.empty:
                return "No under-utilized S3 buckets found in current filtered view.", None
            df = s3_under.sort_values("SavingPerMonthUSD", ascending=False).copy()
            return "Top under-utilized S3 buckets (by estimated monthly saving):", df[["BucketName","Region","StorageClass","UtilizationPercent","SavingPerMonthUSD"]].head(20)
        if "grand total" in q or "total saving" in q or "savings" in q:
            return f"Grand total monthly saving (current view): ${grand_total:,.2f}", None
        if q.startswith("top "):
            # e.g., "top 5 expensive ec2" or "top 10 buckets"
            try:
                parts = q.split()
                n = int(parts[1])
            except Exception:
                n = 5
            if "expensive ec2" in q or "most expensive ec2" in q:
                if "CostPerHourUSD" in ec2_f.columns:
                    return f"Top {n} most expensive EC2 instances:", ec2_f.sort_values("CostPerHourUSD", ascending=False).head(n)[["InstanceId","Region","InstanceType","CostPerHourUSD"]]
                else:
                    return "EC2 cost column missing.", None
            if "largest s3" in q or "biggest s3" in q:
                if "TotalSizeGB" in s3_f.columns:
                    return f"Top {n} largest S3 buckets by size:", s3_f.sort_values("TotalSizeGB", ascending=False).head(n)[["BucketName","Region","TotalSizeGB","MonthlyCostUSD"]]
                else:
                    return "S3 size column missing.", None
        # fallback help
        return ("I can show under-utilized EC2/S3, top lists, or grand total. Try queries like: "
                "'show idle ec2', 'top 5 most expensive ec2', 'under-utilized s3', 'grand total saving'."), None

    # chat UI
    col_in, col_btn = st.columns([8,1])
    with col_in:
        user_q = st.text_input("Ask a question (e.g., 'show idle ec2')", key="chat_input")
    with col_btn:
        submit = st.button("Ask")

    if submit and user_q:
        st.session_state.chat_history.append(("user", user_q))
        bot_text, bot_df = bot_reply(user_q)
        st.session_state.chat_history.append(("bot", bot_text))
        if bot_df is not None:
            # store last df for user to view/download
            st.session_state["last_df"] = bot_df

    # display history
    for who, text in st.session_state.chat_history:
        if who == "user":
            st.markdown(f"<div class='chat-user'><b>You:</b> {text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'><b>Bot:</b> {text}</div>", unsafe_allow_html=True)

    # show last returned table if exists
    if "last_df" in st.session_state:
        st.markdown("#### Result table")
        st.dataframe(st.session_state["last_df"], use_container_width=True)
        csv = st.session_state["last_df"].to_csv(index=False).encode("utf-8")
        st.download_button("Download result CSV", csv, file_name="chat_result.csv", mime="text/csv")

# ---------- Downloads ----------
with tab_downloads:
    st.header("Exports")
    if not ec2_under.empty:
        tmp = ec2_under.copy()
        st.download_button("Download EC2 underutilized CSV", tmp.to_csv(index=False).encode("utf-8"), file_name="ec2_underutilized.csv")
    if not s3_under.empty:
        tmp2 = s3_under.copy()
        st.download_button("Download S3 underutilized CSV", tmp2.to_csv(index=False).encode("utf-8"), file_name="s3_underutilized.csv")
    st.download_button("Download EC2 filtered", ec2_f.to_csv(index=False).encode("utf-8"), file_name="ec2_filtered.csv")
    st.download_button("Download S3 filtered", s3_f.to_csv(index=False).encode("utf-8"), file_name="s3_filtered.csv")

# ------------------------ End ------------------------
