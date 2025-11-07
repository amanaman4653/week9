import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ==================== Page / Theme ====================
st.set_page_config(page_title="Week 9 – EC2 & S3 EDA", layout="wide")

EC2_COLOR = "#2E86DE"   # blue
S3_COLOR  = "#E67E22"   # orange
BG_CARD   = "rgba(255,255,255,0.06)"

st.markdown(f"""
<style>
.block-container {{ padding-top:1rem; padding-bottom:0.5rem; }}
.ec2-title h3, .ec2-title h4 {{ color:{EC2_COLOR}!important; margin:0.2rem 0; }}
.s3-title  h3, .s3-title  h4 {{ color:{S3_COLOR}!important;  margin:0.2rem 0; }}
.metric-box [data-testid="stMetricValue"] {{ font-size: 28px; }}
hr {{ margin: 0.8rem 0; }}
</style>
""", unsafe_allow_html=True)

# ==================== Column Mapping ====================
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

# ==================== Helpers ====================
def resolve_path(p: str) -> str:
    """
    Try the provided path; if it doesn't exist, try the same filename
    in repo root and in data/ . Returns a string path (may not exist).
    """
    cand = Path(p)
    if cand.exists():
        return str(cand)
    name = Path(p).name
    for alt in [Path(name), Path("data") / name]:
        if alt.exists():
            return str(alt)
    return str(cand)

@st.cache_data
def load_and_standardize(path: str, colmap: dict, is_ec2: bool):
    df = pd.read_csv(resolve_path(path))
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})
    if is_ec2:
        for c in ["CPUUtilization","MemoryUtilization","NetworkIn_Bps","NetworkOut_Bps","CostPerHourUSD"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    else:
        for c in ["ObjectCount","TotalSizeGB","MonthlyCostUSD"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def df_info_as_text(df: pd.DataFrame) -> str:
    buf = io.StringIO(); df.info(buf=buf); return buf.getvalue()

def format_dollars(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return x

def apply_filters(ec2: pd.DataFrame, s3: pd.DataFrame,
                  ec2_regions, s3_regions, itypes, sclasses,
                  ec2_cost_range, s3_cost_range):
    e = ec2.copy(); s = s3.copy()
    if "Region" in e.columns and ec2_regions: e = e[e["Region"].isin(ec2_regions)]
    if "InstanceType" in e.columns and itypes: e = e[e["InstanceType"].isin(itypes)]
    if ec2_cost_range and "CostPerHourUSD" in e.columns:
        lo, hi = ec2_cost_range; e = e[(e["CostPerHourUSD"] >= lo) & (e["CostPerHourUSD"] <= hi)]
    if "Region" in s.columns and s3_regions: s = s[s["Region"].isin(s3_regions)]
    if "StorageClass" in s.columns and sclasses: s = s[s["StorageClass"].isin(sclasses)]
    if s3_cost_range and "MonthlyCostUSD" in s.columns:
        lo2, hi2 = s3_cost_range; s = s[(s["MonthlyCostUSD"] >= lo2) & (s["MonthlyCostUSD"] <= hi2)]
    return e, s

def only_underutilized_ec2(df: pd.DataFrame, threshold=20.0) -> pd.DataFrame:
    if "CPUUtilization" not in df.columns: return df.iloc[0:0].copy()
    return df[df["CPUUtilization"] < threshold].copy()

def find_utilization_percent_column(df: pd.DataFrame):
    candidates = [c for c in df.columns if str(c).lower() in
                  {"utilizationpercent","utilization_%","usedpercent","usagepercent","storageutilization","storage_utilization","utilization"}]
    return candidates[0] if candidates else None

def s3_underutilized_20pct(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    col = find_utilization_percent_column(df)
    d = df.copy()
    if col:
        util = pd.to_numeric(d[col], errors="coerce")
        if util.dropna().between(0, 1).mean() > 0.6: util = util * 100.0
        d["UtilizationPercent"] = util
    else:
        if "TotalSizeGB" not in d.columns: return d.iloc[0:0].copy()
        if "Region" not in d.columns: d["Region"] = "ALL"
        p95 = d.groupby("Region")["TotalSizeGB"].quantile(0.95).rename("SizeP95").reset_index()
        d = d.merge(p95, on="Region", how="left")
        d["UtilizationPercent"] = np.where(d["SizeP95"] > 0, (d["TotalSizeGB"] / d["SizeP95"]) * 100.0, 0.0)
    return d[d["UtilizationPercent"] < 20.0].copy()

def build_recommendations(ec2_under: pd.DataFrame, s3_under: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in ec2_under.iterrows():
        iid = r.get("InstanceId", "unknown"); reg = r.get("Region", "unknown")
        it  = r.get("InstanceType", "unknown"); cpu = r.get("CPUUtilization", np.nan)
        cph = r.get("CostPerHourUSD", 0.0); save = cph * 730.0
        action = "Terminate / schedule stop" if pd.notna(cpu) and cpu < 10 else "Right-size to smaller family"
        rows.append({"Resource":"EC2","IdOrName":str(iid),"Region":reg,
                     "Details":f"{it}, CPU {cpu:.2f}%, ${cph:.3f}/hr",
                     "Recommendation":f"{action}; reclaim ~{format_dollars(save)} per month",
                     "MonthlySavingUSD":round(save,2)})
    for _, r in s3_under.iterrows():
        b = r.get("BucketName","unknown"); reg = r.get("Region","unknown")
        sc = r.get("StorageClass","unknown"); util = r.get("UtilizationPercent", np.nan)
        cost = r.get("MonthlyCostUSD", 0.0)
        reco = ("Move cold objects to IA/Glacier, enable lifecycle & expire noncurrent versions"
                if str(sc).upper() == "STANDARD"
                else "Review retention; move to cheaper class; expire noncurrent versions")
        rows.append({"Resource":"S3","IdOrName":str(b),"Region":reg,
                     "Details":f"{sc}, Util {util:.2f}%, Cost/mo {format_dollars(cost)}",
                     "Recommendation":f"{reco}; reclaim ~{format_dollars(cost)} per month (if eliminated)",
                     "MonthlySavingUSD":round(float(cost),2)})
    return pd.DataFrame(rows)

# ==================== Sidebar (paths & filters) ====================
with st.sidebar:
    st.header("Data Sources")
    # Default to root filenames; app will also auto-find them in data/
    ec2_path = st.text_input("EC2 CSV path", "ec2.csv")
    s3_path  = st.text_input("S3 CSV path",  "s3.csv")
    st.caption("Place files beside this app (ec2.csv, s3.csv) or in /data. The app will find them.")

# Load
try:
    ec2_raw = load_and_standardize(ec2_path, EC2_MAP, is_ec2=True)
    s3_raw  = load_and_standardize(s3_path,  S3_MAP,  is_ec2=False)
except Exception as e:
    st.error(f"Failed to load CSVs: {e}")
    st.stop()

# Basic cleaning
ec2 = ec2_raw.fillna({"InstanceType":"Unknown","Region":"Unknown","Tags":"Unknown"})
s3  = s3_raw.fillna({"StorageClass":"Unknown","Region":"Unknown",
                     "Encryption":"Unknown","VersioningEnabled":"Unknown"})

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    ec2_regions_all = sorted(ec2["Region"].dropna().unique().tolist()) if "Region" in ec2.columns else []
    s3_regions_all  = sorted(s3["Region"].dropna().unique().tolist()) if "Region" in s3.columns else []
    itypes_all      = sorted(ec2["InstanceType"].dropna().unique().tolist()) if "InstanceType" in ec2.columns else []
    sclasses_all    = sorted(s3["StorageClass"].dropna().unique().tolist()) if "StorageClass" in s3.columns else []
    pick_ec2_regions = st.multiselect("EC2 Regions", ec2_regions_all, default=ec2_regions_all)
    pick_s3_regions  = st.multiselect("S3 Regions",  s3_regions_all,  default=s3_regions_all)
    pick_itypes      = st.multiselect("Instance Types", itypes_all, default=itypes_all[:10] if len(itypes_all)>10 else itypes_all)
    pick_sclasses    = st.multiselect("Storage Classes", sclasses_all, default=sclasses_all)
    if "CostPerHourUSD" in ec2.columns and len(ec2):
        lo, hi = float(ec2["CostPerHourUSD"].min()), float(ec2["CostPerHourUSD"].max())
        ec2_cost_range = st.slider("EC2 Cost/hour filter", lo, hi, (lo, hi))
    else:
        ec2_cost_range = None
    if "MonthlyCostUSD" in s3.columns and len(s3):
        lo2, hi2 = float(s3["MonthlyCostUSD"].min()), float(s3["MonthlyCostUSD"].max())
        s3_cost_range = st.slider("S3 Monthly cost filter", lo2, hi2, (lo2, hi2))
    else:
        s3_cost_range = None

# Apply filters globally
ec2_f, s3_f = apply_filters(ec2, s3, pick_ec2_regions, pick_s3_regions, pick_itypes, pick_sclasses, ec2_cost_range, s3_cost_range)

# ==================== Title & KPIs ====================
st.title("Week 9 – EC2 & S3 Exploratory Data Analysis")

k1,k2,k3,k4 = st.columns(4)
with k1:
    with st.container(border=True):
        st.markdown("**EC2 rows**")
        st.markdown(f"<div class='metric-box'><span>{len(ec2_f):,}</span></div>", unsafe_allow_html=True)
with k2:
    with st.container(border=True):
        st.markdown("**S3 rows**")
        st.markdown(f"<div class='metric-box'><span>{len(s3_f):,}</span></div>", unsafe_allow_html=True)
with k3:
    with st.container(border=True):
        avg = ec2_f["CostPerHourUSD"].mean() if "CostPerHourUSD" in ec2_f.columns and len(ec2_f) else np.nan
        st.markdown("**Avg EC2 $/hr**")
        st.markdown(f"<div class='metric-box'><span>{format_dollars(avg)}</span></div>", unsafe_allow_html=True)
with k4:
    with st.container(border=True):
        total = s3_f["MonthlyCostUSD"].sum() if "MonthlyCostUSD" in s3_f.columns and len(s3_f) else np.nan
        st.markdown("**Total S3 $/mo**")
        st.markdown(f"<div class='metric-box'><span>{format_dollars(total)}</span></div>", unsafe_allow_html=True)

st.markdown("---")

# ==================== Tabs ====================
tab_overview, tab_ec2, tab_s3, tab_insights, tab_opt, tab_downloads = st.tabs(
    ["Overview", "EC2", "S3", "Insights", "Optimization", "Downloads"]
)

# -------- OVERVIEW --------
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='ec2-title'><h3>EC2 – Overview</h3></div>", unsafe_allow_html=True)
        st.code(df_info_as_text(ec2_f))
        st.write("**shape**", ec2_f.shape)
        st.write("**describe()**"); st.dataframe(ec2_f.describe(include='all'), use_container_width=True)
    with c2:
        st.markdown("<div class='s3-title'><h3>S3 – Overview</h3></div>", unsafe_allow_html=True)
        st.code(df_info_as_text(s3_f))
        st.write("**shape**", s3_f.shape)
        st.write("**describe()**"); st.dataframe(s3_f.describe(include='all'), use_container_width=True)
    st.markdown("**Head (first 10 rows)**")
    h1,h2 = st.columns(2)
    with h1: st.dataframe(ec2_f.head(10), use_container_width=True)
    with h2: st.dataframe(s3_f.head(10),  use_container_width=True)

# -------- EC2 --------
with tab_ec2:
    st.markdown("<div class='ec2-title'><h3>EC2 – Visuals</h3></div>", unsafe_allow_html=True)
    if "CPUUtilization" in ec2_f.columns and len(ec2_f):
        fig = px.histogram(ec2_f, x="CPUUtilization", nbins=30, title="EC2 CPU Utilization",
                           color_discrete_sequence=[EC2_COLOR])
        st.plotly_chart(fig, use_container_width=True)
    if {"CPUUtilization","CostPerHourUSD"}.issubset(ec2_f.columns) and len(ec2_f):
        fig = px.scatter(ec2_f, x="CPUUtilization", y="CostPerHourUSD", title="EC2: CPU vs Cost",
                         color_discrete_sequence=[EC2_COLOR],
                         hover_data=["InstanceId","Region","InstanceType"])
        st.plotly_chart(fig, use_container_width=True)

# -------- S3 --------
with tab_s3:
    st.markdown("<div class='s3-title'><h3>S3 – Visuals</h3></div>", unsafe_allow_html=True)
    if {"Region","TotalSizeGB"}.issubset(s3_f.columns) and len(s3_f):
        series = s3_f.groupby("Region")["TotalSizeGB"].sum().reset_index().sort_values("TotalSizeGB", ascending=False)
        fig = px.bar(series, x="Region", y="TotalSizeGB", title="S3 Total Storage by Region",
                     color_discrete_sequence=[S3_COLOR])
        st.plotly_chart(fig, use_container_width=True)
    if {"TotalSizeGB","MonthlyCostUSD"}.issubset(s3_f.columns) and len(s3_f):
        size_col = "ObjectCount" if "ObjectCount" in s3_f.columns else "TotalSizeGB"
        fig = px.scatter(s3_f, x="TotalSizeGB", y="MonthlyCostUSD",
                         size=size_col, size_max=40,
                         title="S3: Monthly Cost vs Total Storage (bubble≈objects)",
                         color_discrete_sequence=[S3_COLOR],
                         hover_data=["BucketName","Region","StorageClass"])
        st.plotly_chart(fig, use_container_width=True)

# -------- OPTIMIZATION --------
with tab_opt:
    st.subheader("Optimization – Fixed 20% Rules")

    # ===== EC2: CPU < 20% (Elimination saving) =====
    st.markdown("<div class='ec2-title'><h4>EC2 – CPU < 20% (Eliminate)</h4></div>", unsafe_allow_html=True)
    ec2_under = only_underutilized_ec2(ec2_f, threshold=20.0)
    if not ec2_under.empty and "CostPerHourUSD" in ec2_under.columns:
        ec2_under = ec2_under.copy()
        ec2_under["SavingPerMonthUSD"] = ec2_under["CostPerHourUSD"] * 730.0
        st.dataframe(
            ec2_under[["InstanceId","Region","InstanceType","CPUUtilization","CostPerHourUSD","SavingPerMonthUSD"]],
            use_container_width=True
        )
        fig = px.scatter(ec2_under, x="CPUUtilization", y="CostPerHourUSD",
                         title="Under-utilized EC2: CPU vs Cost (bubble≈monthly saving)",
                         color_discrete_sequence=[EC2_COLOR],
                         size="SavingPerMonthUSD", size_max=45,
                         hover_data=["InstanceId","Region","InstanceType","SavingPerMonthUSD"])
        st.plotly_chart(fig, use_container_width=True)
        total_ec2_month = float(ec2_under["SavingPerMonthUSD"].sum())
        st.metric("EC2 total saving / month", f"${total_ec2_month:,.2f}")
    else:
        st.info("No EC2 instances with CPU < 20% after filters.")
        total_ec2_month = 0.0

    st.markdown("---")

    # ===== S3: Utilization < 20% (Elimination saving) =====
    st.markdown("<div class='s3-title'><h4>S3 – Storage Utilization < 20% (Eliminate)</h4></div>", unsafe_allow_html=True)
    s3_under = s3_underutilized_20pct(s3_f)

    util_col = find_utilization_percent_column(s3_f)
    if util_col:
        st.caption(f"Utilization derived from column **{util_col}** (< 20%).")
    else:
        st.caption("Utilization estimated as TotalSizeGB / 95th-percentile size within each region (< 20%).")

    if not s3_under.empty and "MonthlyCostUSD" in s3_under.columns:
        s3_under = s3_under.copy()
        s3_under["SavingPerMonthUSD"] = s3_under["MonthlyCostUSD"]  # eliminate
        show_cols = ["BucketName","Region","StorageClass","TotalSizeGB","MonthlyCostUSD","SavingPerMonthUSD"]
        if "UtilizationPercent" in s3_under.columns:
            show_cols.insert(3, "UtilizationPercent")
        st.dataframe(s3_under[show_cols], use_container_width=True)

        fig = px.scatter(s3_under, x="TotalSizeGB", y="MonthlyCostUSD",
                         title="Under-utilized S3: Cost vs Size (bubble≈monthly saving)",
                         color_discrete_sequence=[S3_COLOR],
                         size="SavingPerMonthUSD", size_max=45,
                         hover_data=["BucketName","Region","StorageClass","SavingPerMonthUSD"])
        st.plotly_chart(fig, use_container_width=True)
        total_s3_month = float(s3_under["SavingPerMonthUSD"].sum())
        st.metric("S3 total saving / month", f"${total_s3_month:,.2f}")
    else:
        st.info("No S3 buckets under 20% utilization after filters.")
        total_s3_month = 0.0

    st.markdown("---")
    grand_total = total_ec2_month + total_s3_month
    st.markdown("## Grand Total (Monthly) — Eliminating under-utilized EC2 & S3")
    st.metric("Grand total saving per month", f"${grand_total:,.2f}")

    # ===== RECOMMENDATIONS =====
    st.markdown("## Recommendations")
    rec_df = build_recommendations(ec2_under if 'ec2_under' in locals() else pd.DataFrame(),
                                   s3_under if 's3_under' in locals() else pd.DataFrame())
    if rec_df.empty:
        st.info("No recommendations — no under-utilized resources matched the 20% rules.")
    else:
        bullets = [f"- **{r['Resource']}** {r['IdOrName']} ({r['Region']}): {r['Recommendation']}"
                   for _, r in rec_df.iterrows()]
        st.markdown("\n".join(bullets))
        st.markdown("### Recommendation Table")
        st.dataframe(rec_df, use_container_width=True)
        st.download_button(
            "Download recommendations (CSV)",
            rec_df.to_csv(index=False).encode("utf-8"),
            file_name="recommendations.csv",
            mime="text/csv",
        )

# -------- DOWNLOADS --------
with tab_downloads:
    st.subheader("Exports")
    ec2_u = only_underutilized_ec2(ec2_f, 20.0)
    if not ec2_u.empty and "CostPerHourUSD" in ec2_u.columns:
        tmp = ec2_u.copy(); tmp["SavingPerMonthUSD"] = tmp["CostPerHourUSD"] * 730.0
        st.download_button("Download EC2 under-utilized (CPU<20%)",
                           tmp.to_csv(index=False).encode("utf-8"),
                           file_name="ec2_underutilized.csv", mime="text/csv")
    s3_u = s3_underutilized_20pct(s3_f)
    if not s3_u.empty and "MonthlyCostUSD" in s3_u.columns:
        tmp2 = s3_u.copy(); tmp2["SavingPerMonthUSD"] = tmp2["MonthlyCostUSD"]
        st.download_button("Download S3 under-utilized (<20% utilization)",
                           tmp2.to_csv(index=False).encode("utf-8"),
                           file_name="s3_underutilized.csv", mime="text/csv")
    if 'rec_df' in locals() and not rec_df.empty:
        st.download_button("Download all recommendations (CSV)",
                           rec_df.to_csv(index=False).encode("utf-8"),
                           file_name="recommendations.csv", mime="text/csv")
    st.download_button("Download EC2 (filtered)", ec2_f.to_csv(index=False).encode("utf-8"),
                       file_name="ec2_filtered.csv", mime="text/csv")
    st.download_button("Download S3 (filtered)", s3_f.to_csv(index=False).encode("utf-8"),
                       file_name="s3_filtered.csv", mime="text/csv")
