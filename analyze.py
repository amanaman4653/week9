#!/usr/bin/env python
import argparse, os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FIGS_DIR = "figs"

# Expected columns. If your CSV uses slightly different names,
# adjust the values on the right side only.
COLUMN_MAP_EC2 = {
    # yours -> expected
    "ResourceId": "InstanceId",
    "InstanceType": "InstanceType",
    "Region": "Region",
    "State": "State",
    "CPUUtilization": "CPUUtilization",
    "MemoryUtilization": "MemoryUtilization",
    "NetworkIn_Bps": "NetworkIn_Bps",
    "NetworkOut_Bps": "NetworkOut_Bps",

    # your file has CostUSD (not hourly); we’ll treat it as CostPerHourUSD for the assignment visuals
    "CostUSD": "CostPerHourUSD",

    "Tags": "Tags",
    "CreationDate": "LaunchTime",  # rename so it shows up in the summary
    # keep these if present
    "ResourceType": "ResourceType",
}

COLUMN_MAP_S3 = {
    "BucketName": "BucketName",
    "Region": "Region",
    "StorageClass": "StorageClass",
    "ObjectCount": "ObjectCount",
    "TotalSizeGB": "TotalSizeGB",

    # your file has CostUSD; map to MonthlyCostUSD for charts
    "CostUSD": "MonthlyCostUSD",

    "VersionEnabled": "VersioningEnabled",
    "Encryption": "Encryption",
    "CreationDate": "CreatedDate",
    "Tags": "Tags",
}


def load_csv(path, colmap):
    df = pd.read_csv(path)
    # rename if user used slightly different labels
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})
    return df

def coerce_types_ec2(df):
    for c in ["CPUUtilization","MemoryUtilization","NetworkIn_Bps","NetworkOut_Bps","CostPerHourUSD"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "LaunchTime" in df.columns:
        df["LaunchTime"] = pd.to_datetime(df["LaunchTime"], errors="coerce")
    return df

def coerce_types_s3(df):
    for c in ["ObjectCount","TotalSizeGB","MonthlyCostUSD"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "CreatedDate" in df.columns:
        df["CreatedDate"] = pd.to_datetime(df["CreatedDate"], errors="coerce")
    if "VersioningEnabled" in df.columns:
        df["VersioningEnabled"] = df["VersioningEnabled"].astype(str)
    return df

def impute_missing(df):
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna("Unknown")
    return df

def flag_outliers_iqr(df, col, newcol):
    if col not in df.columns:
        df[newcol] = False
        return df
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    df[newcol] = (df[col] < low) | (df[col] > high)
    return df

def plot_hist(series, title, out_png, xlabel):
    plt.figure()
    series.dropna().plot(kind="hist", bins=30)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_scatter(x, y, title, out_png, xlabel, ylabel):
    plt.figure()
    plt.scatter(x, y, alpha=0.7)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_bar(series, title, out_png, xlabel, ylabel):
    plt.figure()
    series.plot(kind="bar")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def summarize(df, name):
    lines = []
    lines.append(f"### {name} shape: {df.shape}\n")
    lines.append(df.describe(include='all').to_markdown()
)
    lines.append("")
    return "\n".join(lines)

def optimization_hints(ec2, s3):
    hints = []

    if {"CPUUtilization","CostPerHourUSD"}.issubset(ec2.columns):
        under = ec2[(ec2["CPUUtilization"] < 15) & (ec2["CostPerHourUSD"] >= ec2["CostPerHourUSD"].median())]
        if len(under)>0:
            hints.append({
                "title": "Right-size or schedule EC2",
                "detail": f"{len(under)} instances have <15% CPU but above-median cost.",
                "examples": under.head(5)[["InstanceType","Region","CPUUtilization","CostPerHourUSD"]].to_dict(orient="records")
            })

    if "State" in ec2.columns:
        stale = ec2[ec2["State"].isin(["stopped","terminated"])]
        if len(stale)>0:
            hints.append({"title":"Cleanup stopped/terminated EC2","detail":f"{len(stale)} resources appear stale."})

    if {"StorageClass","TotalSizeGB"}.issubset(s3.columns):
        std = s3[s3["StorageClass"].astype(str).str.upper()=="STANDARD"]
        large_std = std[std["TotalSizeGB"]>std["TotalSizeGB"].median()]
        if len(large_std)>0:
            hints.append({"title":"S3 lifecycle to IA/Glacier","detail":f"{len(large_std)} large STANDARD buckets."})

    if "VersioningEnabled" in s3.columns and "MonthlyCostUSD" in s3.columns:
        costly = s3[(s3["VersioningEnabled"].str.lower()=="true") & (s3["MonthlyCostUSD"]>s3["MonthlyCostUSD"].median())]
        if len(costly)>0:
            hints.append({"title":"Expire noncurrent S3 versions","detail":f"{len(costly)} cost-heavy versioned buckets."})
    return hints

def main(ec2_path, s3_path):
    os.makedirs(FIGS_DIR, exist_ok=True)

    ec2 = coerce_types_ec2(load_csv(ec2_path, COLUMN_MAP_EC2))
    s3  = coerce_types_s3(load_csv(s3_path, COLUMN_MAP_S3))

    print("EC2 columns:", list(ec2.columns))
    print("S3 columns:", list(s3.columns))

    ec2 = impute_missing(ec2)
    s3  = impute_missing(s3)

    ec2 = flag_outliers_iqr(ec2, "CPUUtilization", "is_outlier_cpu")
    ec2 = flag_outliers_iqr(ec2, "CostPerHourUSD", "is_outlier_cost")
    s3  = flag_outliers_iqr(s3, "TotalSizeGB", "is_outlier_size")
    s3  = flag_outliers_iqr(s3, "MonthlyCostUSD", "is_outlier_monthly_cost")

    if "CPUUtilization" in ec2.columns:
        plot_hist(ec2["CPUUtilization"], "EC2 CPU Utilization", f"{FIGS_DIR}/ec2_cpu_hist.png", "CPU %")
    if {"CPUUtilization","CostPerHourUSD"}.issubset(ec2.columns):
        plot_scatter(ec2["CPUUtilization"], ec2["CostPerHourUSD"], "EC2 CPU vs Cost", f"{FIGS_DIR}/ec2_cpu_vs_cost_scatter.png", "CPU %", "Cost/Hour (USD)")
    if {"Region","TotalSizeGB"}.issubset(s3.columns):
        storage_by_region = s3.groupby("Region")["TotalSizeGB"].sum().sort_values(ascending=False)
        plot_bar(storage_by_region, "S3 Total Storage by Region", f"{FIGS_DIR}/s3_storage_by_region_bar.png", "Region", "Total Size (GB)")
    if {"TotalSizeGB","MonthlyCostUSD"}.issubset(s3.columns):
        plot_scatter(s3["TotalSizeGB"], s3["MonthlyCostUSD"], "S3 Cost vs Storage", f"{FIGS_DIR}/s3_cost_vs_storage_scatter.png", "Total Size (GB)", "Monthly Cost (USD)")

    top5_ec2 = ec2.sort_values("CostPerHourUSD", ascending=False).head(5) if "CostPerHourUSD" in ec2.columns else pd.DataFrame()
    top5_s3  = s3.sort_values("TotalSizeGB", ascending=False).head(5) if "TotalSizeGB" in s3.columns else pd.DataFrame()

    avg_ec2_cost_by_region = ec2.groupby("Region")["CostPerHourUSD"].mean().sort_values(ascending=False) if {"Region","CostPerHourUSD"}.issubset(ec2.columns) else pd.Series(dtype=float)
    total_s3_storage_by_region = s3.groupby("Region")["TotalSizeGB"].sum().sort_values(ascending=False) if {"Region","TotalSizeGB"}.issubset(s3.columns) else pd.Series(dtype=float)

    hints = optimization_hints(ec2, s3)

    parts = [
        "# EDA Summary",
        summarize(ec2, "EC2"),
        summarize(s3, "S3"),
        "## Top 5 Most Expensive EC2",
        top5_ec2.to_markdown(index=False) if not top5_ec2.empty else "_N/A_",
        "## Top 5 Largest S3 Buckets",
        top5_s3.to_markdown(index=False) if not top5_s3.empty else "_N/A_",
        "## Average EC2 Cost per Region",
        avg_ec2_cost_by_region.to_frame("AvgCostPerHourUSD").to_markdown() if len(avg_ec2_cost_by_region)>0 else "_N/A_",
        "## Total S3 Storage per Region",
        total_s3_storage_by_region.to_frame("TotalSizeGB").to_markdown() if len(total_s3_storage_by_region)>0 else "_N/A_",
        "## Optimization Suggestions",
        json.dumps(hints, indent=2) if hints else "_No strong signals detected_",
        "_Figures saved in figs/_"
    ]
    with open("summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("Done. See figs/ and summary.md")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ec2", required=True)
    p.add_argument("--s3", required=True)
    args = p.parse_args()
    main(args.ec2, args.s3)
