#!/usr/bin/env python3
"""Autonomous downstream RNA-seq workflow for trisomic vs disomic comparison.

The workflow starts from an nf-core/rnaseq-style count matrix and sample metadata.
It performs deterministic analyses and writes a compact interpretation report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def import_scientific_stack() -> None:
    global np, pd, plt, sns, stats, linkage, leaves_list, pdist, PCA, StandardScaler, multipletests
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from statsmodels.stats.multitest import multipletests
    except ImportError as exc:
        raise SystemExit(
            "Missing scientific Python dependencies. Install with "
            "`pip install -r requirements.txt`."
        ) from exc


@dataclass
class Design:
    sample_id: str
    condition: str
    case_level: str
    control_level: str
    batch_columns: List[str]
    covariates: List[str]
    pair_column: Optional[str]


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "YAML config requires pyyaml. Use analysis_config.json or install with `pip install pyyaml`."
        ) from exc
    return yaml.safe_load(path.read_text())


def load_counts(path: Path) -> pd.DataFrame:
    counts = read_table(path)
    first_col = counts.columns[0]
    if first_col.lower() in {"gene", "gene_id", "id", "transcript", "transcript_id", "unnamed: 0"}:
        counts = counts.set_index(first_col)
    counts.index = counts.index.astype(str)
    counts = counts.apply(pd.to_numeric, errors="coerce").fillna(0)
    counts = counts.loc[counts.sum(axis=1) > 0]
    return counts.round().astype(int)


def align_counts_metadata(counts: pd.DataFrame, metadata: pd.DataFrame, design: Design) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing_cols = [c for c in [design.sample_id, design.condition] if c not in metadata.columns]
    if missing_cols:
        raise ValueError(f"Metadata is missing required columns: {missing_cols}")
    metadata = metadata.copy()
    metadata[design.sample_id] = metadata[design.sample_id].astype(str)
    metadata = metadata.set_index(design.sample_id, drop=False)

    shared = [sample for sample in counts.columns.astype(str) if sample in metadata.index]
    if len(shared) < 2:
        raise ValueError("Fewer than two count columns match metadata sample IDs.")
    counts = counts.loc[:, shared]
    metadata = metadata.loc[shared].copy()

    present_levels = set(metadata[design.condition].astype(str))
    required = {design.case_level, design.control_level}
    if not required.issubset(present_levels):
        raise ValueError(f"Condition column must contain {required}; observed {present_levels}.")
    return counts, metadata


def aggregate_transcripts(counts: pd.DataFrame, tx2gene_path: Optional[Path]) -> pd.DataFrame:
    if tx2gene_path is None or not tx2gene_path.exists():
        return counts
    tx2gene = read_table(tx2gene_path)
    tx_col = "transcript_id" if "transcript_id" in tx2gene.columns else tx2gene.columns[0]
    gene_col = "gene_id" if "gene_id" in tx2gene.columns else tx2gene.columns[1]
    mapper = tx2gene[[tx_col, gene_col]].dropna().drop_duplicates().set_index(tx_col)[gene_col]
    mapped = counts.join(mapper.rename("gene_id"), how="inner")
    if mapped.empty:
        return counts
    return mapped.groupby("gene_id").sum()


def filter_counts(counts: pd.DataFrame, min_total: int, min_samples: int) -> pd.DataFrame:
    keep = (counts.sum(axis=1) >= min_total) & ((counts > 0).sum(axis=1) >= min_samples)
    return counts.loc[keep].copy()


def size_factors_median_ratio(counts: pd.DataFrame) -> pd.Series:
    positive = counts.replace(0, np.nan)
    geo_means = np.exp(np.log(positive).mean(axis=1))
    valid = np.isfinite(geo_means) & (geo_means > 0)
    ratios = counts.loc[valid].div(geo_means[valid], axis=0)
    size_factors = ratios.replace(0, np.nan).median(axis=0)
    return size_factors.fillna(size_factors.median())


def normalized_expression(counts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    lib = counts.sum(axis=0)
    cpm = counts.div(lib, axis=1) * 1_000_000
    size_factors = size_factors_median_ratio(counts)
    norm = counts.div(size_factors, axis=1)
    log_norm = np.log2(norm + 1)
    return cpm, log_norm, size_factors


def differential_expression_welch(
    log_expr: pd.DataFrame,
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    design: Design,
) -> pd.DataFrame:
    condition = metadata[design.condition].astype(str)
    case_samples = metadata.index[condition == design.case_level].tolist()
    control_samples = metadata.index[condition == design.control_level].tolist()
    rows = []
    base_mean = counts.mean(axis=1)
    for gene, values in log_expr.iterrows():
        case = values[case_samples].astype(float)
        ctrl = values[control_samples].astype(float)
        stat, pval = stats.ttest_ind(case, ctrl, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "gene_id": gene,
                "baseMean": float(base_mean.loc[gene]),
                "log2FoldChange": float(case.mean() - ctrl.mean()),
                "stat": float(stat) if np.isfinite(stat) else np.nan,
                "pvalue": float(pval) if np.isfinite(pval) else np.nan,
            }
        )
    res = pd.DataFrame(rows).set_index("gene_id")
    ok = res["pvalue"].notna()
    res["padj"] = np.nan
    if ok.any():
        res.loc[ok, "padj"] = multipletests(res.loc[ok, "pvalue"], method="fdr_bh")[1]
    return res.sort_values(["padj", "pvalue"], na_position="last")


def add_annotation(res: pd.DataFrame, annotation_path: Optional[Path]) -> pd.DataFrame:
    if annotation_path is None or not annotation_path.exists():
        return res
    ann = read_table(annotation_path)
    key = "gene_id" if "gene_id" in ann.columns else ann.columns[0]
    ann[key] = ann[key].astype(str)
    ann = ann.drop_duplicates(key).set_index(key)
    out = res.join(ann, how="left")
    if "gene_name" in out.columns:
        cols = ["gene_name"] + [c for c in out.columns if c != "gene_name"]
        out = out[cols]
    return out


def plot_qc(counts: pd.DataFrame, log_expr: pd.DataFrame, metadata: pd.DataFrame, design: Design, outdir: Path) -> Dict[str, str]:
    paths = {}
    outdir.mkdir(parents=True, exist_ok=True)
    qc = pd.DataFrame(
        {
            "sample": counts.columns,
            "library_size": counts.sum(axis=0).values,
            "detected_features": (counts > 0).sum(axis=0).values,
            design.condition: metadata[design.condition].astype(str).values,
        }
    )
    qc.to_csv(outdir / "sample_qc_metrics.csv", index=False)

    plt.figure(figsize=(9, 4))
    sns.barplot(data=qc, x="sample", y="library_size", hue=design.condition)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = outdir / "library_sizes.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths["library_sizes"] = str(path)

    pca_data = log_expr.loc[log_expr.var(axis=1).sort_values(ascending=False).head(min(5000, log_expr.shape[0])).index].T
    pca_scaled = StandardScaler().fit_transform(pca_data)
    pca = PCA(n_components=min(3, pca_scaled.shape[0], pca_scaled.shape[1]))
    pcs = pca.fit_transform(pca_scaled)
    pc_df = pd.DataFrame(pcs[:, :2], columns=["PC1", "PC2"], index=log_expr.columns).join(metadata)
    pc_df.to_csv(outdir / "pca_sample_scores.csv")
    plt.figure(figsize=(6.5, 5.5))
    sns.scatterplot(data=pc_df, x="PC1", y="PC2", hue=design.condition, s=90)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.tight_layout()
    path = outdir / "pca.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths["pca"] = str(path)

    corr = log_expr.corr(method="spearman")
    order = leaves_list(linkage(pdist(corr), method="average")) if corr.shape[0] > 2 else np.arange(corr.shape[0])
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr.iloc[order, order], cmap="vlag", vmin=0.5, vmax=1.0)
    plt.tight_layout()
    path = outdir / "sample_correlation_heatmap.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths["sample_correlation"] = str(path)
    return paths


def plot_de(res: pd.DataFrame, outdir: Path, fdr: float, lfc: float) -> Dict[str, str]:
    paths = {}
    plot_df = res.copy()
    plot_df["minus_log10_padj"] = -np.log10(plot_df["padj"].clip(lower=np.nextafter(0, 1)))
    plot_df["significant"] = (plot_df["padj"] <= fdr) & (plot_df["log2FoldChange"].abs() >= lfc)
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=plot_df,
        x="log2FoldChange",
        y="minus_log10_padj",
        hue="significant",
        palette={False: "#7f8c8d", True: "#c0392b"},
        s=14,
        linewidth=0,
    )
    plt.axvline(lfc, color="#34495e", lw=1, ls="--")
    plt.axvline(-lfc, color="#34495e", lw=1, ls="--")
    plt.axhline(-math.log10(fdr), color="#34495e", lw=1, ls="--")
    plt.tight_layout()
    path = outdir / "volcano.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths["volcano"] = str(path)

    top = res.head(50).copy()
    cols = [c for c in ["gene_name", "baseMean", "log2FoldChange", "pvalue", "padj", "chromosome", "biotype"] if c in top.columns]
    top[cols].to_csv(outdir / "top_50_differential_features.csv")
    return paths


def read_gmt(path: Path) -> Dict[str, List[str]]:
    gene_sets: Dict[str, List[str]] = {}
    if not path.exists():
        return gene_sets
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                gene_sets[parts[0]] = sorted(set(parts[2:]))
    return gene_sets


def overrepresentation(
    res: pd.DataFrame,
    gene_sets: Dict[str, List[str]],
    fdr: float,
    lfc: float,
    outdir: Path,
) -> pd.DataFrame:
    if not gene_sets:
        return pd.DataFrame()
    gene_symbol_col = "gene_name" if "gene_name" in res.columns else None
    universe = set(res[gene_symbol_col].dropna().astype(str)) if gene_symbol_col else set(res.index.astype(str))
    sig_mask = (res["padj"] <= fdr) & (res["log2FoldChange"].abs() >= lfc)
    sig = set(res.loc[sig_mask, gene_symbol_col].dropna().astype(str)) if gene_symbol_col else set(res.loc[sig_mask].index.astype(str))
    rows = []
    for name, genes in gene_sets.items():
        gs = set(genes) & universe
        if len(gs) < 5 or not sig:
            continue
        a = len(sig & gs)
        b = len(sig - gs)
        c = len(gs - sig)
        d = len(universe - sig - gs)
        odds, pval = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append({"term": name, "overlap": a, "set_size": len(gs), "odds_ratio": odds, "pvalue": pval})
    ora = pd.DataFrame(rows)
    if ora.empty:
        return ora
    ora["padj"] = multipletests(ora["pvalue"], method="fdr_bh")[1]
    ora = ora.sort_values(["padj", "pvalue"])
    ora.to_csv(outdir / "pathway_overrepresentation.csv", index=False)
    return ora


def chromosome_dosage(res: pd.DataFrame, log_expr: pd.DataFrame, metadata: pd.DataFrame, design: Design, chromosome: str, outdir: Path) -> Dict[str, object]:
    if "chromosome" not in res.columns:
        return {"available": False, "reason": "gene annotation has no chromosome column"}
    chr_series = res["chromosome"].astype(str).str.replace("chr", "", regex=False)
    chr_mask = chr_series == str(chromosome).replace("chr", "")
    non_chr_mask = ~chr_mask & chr_series.notna()
    case_samples = metadata.index[metadata[design.condition].astype(str) == design.case_level].tolist()
    control_samples = metadata.index[metadata[design.condition].astype(str) == design.control_level].tolist()
    chr_genes = res.index[chr_mask].intersection(log_expr.index)
    non_chr_genes = res.index[non_chr_mask].intersection(log_expr.index)
    if len(chr_genes) < 5:
        return {"available": False, "reason": f"fewer than five chromosome {chromosome} features"}

    score = pd.DataFrame(
        {
            "sample": log_expr.columns,
            "chr_score": log_expr.loc[chr_genes].mean(axis=0).values,
            "genome_score": log_expr.loc[non_chr_genes].mean(axis=0).values if len(non_chr_genes) else np.nan,
            design.condition: metadata[design.condition].astype(str).values,
        }
    )
    score["chr_minus_genome"] = score["chr_score"] - score["genome_score"]
    score.to_csv(outdir / f"chromosome_{chromosome}_dosage_scores.csv", index=False)
    plt.figure(figsize=(5.5, 4.5))
    sns.boxplot(data=score, x=design.condition, y="chr_minus_genome", color="#d5dbdb")
    sns.stripplot(data=score, x=design.condition, y="chr_minus_genome", hue=design.condition, s=8)
    plt.tight_layout()
    plt.savefig(outdir / f"chromosome_{chromosome}_dosage_score.png", dpi=180)
    plt.close()

    chr_lfc = res.loc[chr_mask, "log2FoldChange"].dropna()
    non_chr_lfc = res.loc[non_chr_mask, "log2FoldChange"].dropna()
    mw = stats.mannwhitneyu(chr_lfc, non_chr_lfc, alternative="two-sided") if len(non_chr_lfc) else None
    ttest = stats.ttest_ind(
        score.loc[score[design.condition] == design.case_level, "chr_minus_genome"],
        score.loc[score[design.condition] == design.control_level, "chr_minus_genome"],
        equal_var=False,
    )
    return {
        "available": True,
        "n_chr_features": int(chr_mask.sum()),
        "median_chr_log2fc": float(chr_lfc.median()),
        "median_non_chr_log2fc": float(non_chr_lfc.median()) if len(non_chr_lfc) else None,
        "chr_lfc_mannwhitney_pvalue": float(mw.pvalue) if mw else None,
        "sample_score_ttest_pvalue": float(ttest.pvalue) if np.isfinite(ttest.pvalue) else None,
    }


def score_marker_sets(log_expr: pd.DataFrame, metadata: pd.DataFrame, design: Design, marker_sets: Dict[str, List[str]], outdir: Path) -> pd.DataFrame:
    if not marker_sets:
        return pd.DataFrame()
    symbol_index = pd.Index(log_expr.index.astype(str))
    rows = []
    for set_name, genes in marker_sets.items():
        present = [g for g in genes if g in symbol_index]
        if not present:
            continue
        z = pd.DataFrame(StandardScaler().fit_transform(log_expr.loc[present].T), index=log_expr.columns, columns=present)
        score = z.mean(axis=1)
        for sample, value in score.items():
            rows.append({"sample": sample, "marker_set": set_name, "score": value, design.condition: metadata.loc[sample, design.condition]})
    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores
    scores.to_csv(outdir / "marker_set_scores.csv", index=False)
    plt.figure(figsize=(8, 4.8))
    sns.boxplot(data=scores, x="marker_set", y="score", hue=design.condition)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "marker_set_scores.png", dpi=180)
    plt.close()
    return scores


def write_report(
    config: dict,
    counts: pd.DataFrame,
    filtered: pd.DataFrame,
    metadata: pd.DataFrame,
    design: Design,
    res: pd.DataFrame,
    chr_summary: Dict[str, object],
    ora: pd.DataFrame,
    outdir: Path,
) -> None:
    fdr = config["analysis"]["fdr_threshold"]
    lfc = config["analysis"]["log2fc_threshold"]
    sig = res[(res["padj"] <= fdr) & (res["log2FoldChange"].abs() >= lfc)]
    up = sig[sig["log2FoldChange"] > 0]
    down = sig[sig["log2FoldChange"] < 0]
    condition_counts = metadata[design.condition].astype(str).value_counts().to_dict()

    lines = [
        f"# RNA-seq Autonomous Analysis Report: {config['project']['name']}",
        "",
        "## Input Audit",
        f"- Samples analyzed: {metadata.shape[0]}",
        f"- Condition counts: {condition_counts}",
        f"- Features before filtering: {counts.shape[0]}",
        f"- Features after filtering: {filtered.shape[0]}",
        "",
        "## Differential Expression",
        f"- Significant features at FDR <= {fdr} and |log2FC| >= {lfc}: {sig.shape[0]}",
        f"- Higher in {design.case_level}: {up.shape[0]}",
        f"- Lower in {design.case_level}: {down.shape[0]}",
    ]

    if not sig.empty:
        label_col = "gene_name" if "gene_name" in sig.columns else None
        top_up = up.sort_values("log2FoldChange", ascending=False).head(10)
        top_down = down.sort_values("log2FoldChange").head(10)
        lines += ["", f"Top features higher in {design.case_level}:"]
        for gene, row in top_up.iterrows():
            label = row[label_col] if label_col and pd.notna(row[label_col]) else gene
            lines.append(f"- {label}: log2FC {row['log2FoldChange']:.2f}, padj {row['padj']:.2g}")
        lines += ["", f"Top features lower in {design.case_level}:"]
        for gene, row in top_down.iterrows():
            label = row[label_col] if label_col and pd.notna(row[label_col]) else gene
            lines.append(f"- {label}: log2FC {row['log2FoldChange']:.2f}, padj {row['padj']:.2g}")

    lines += ["", "## Chromosome 21 Dosage"]
    if chr_summary.get("available"):
        lines += [
            f"- Chromosome features tested: {chr_summary['n_chr_features']}",
            f"- Median chromosome log2FC: {chr_summary['median_chr_log2fc']:.3f}",
            f"- Median non-chromosome log2FC: {chr_summary['median_non_chr_log2fc']:.3f}",
            f"- Chr-vs-nonchr log2FC Mann-Whitney p-value: {chr_summary['chr_lfc_mannwhitney_pvalue']:.3g}",
            f"- Per-sample chr-minus-genome score p-value: {chr_summary['sample_score_ttest_pvalue']:.3g}",
        ]
    else:
        lines.append(f"- Not run: {chr_summary.get('reason', 'annotation unavailable')}")

    lines += ["", "## Pathway Enrichment"]
    if ora.empty:
        lines.append("- No local GMT enrichment results were produced.")
    else:
        for _, row in ora.head(10).iterrows():
            lines.append(f"- {row['term']}: overlap {int(row['overlap'])}, padj {row['padj']:.2g}")

    lines += [
        "",
        "## Autonomous Next-Step Recommendations",
        "- Inspect PCA and sample-correlation plots for clone, batch, or differentiation-day structure.",
        "- If a paired/isogenic design exists, add the pair column to the design before final inference.",
        "- Confirm chromosome 21 annotation and inspect whether dosage-sensitive HSA21 genes drive the signal.",
        "- Run pathway enrichment with curated local MSigDB/GO GMT files for reproducible interpretation.",
        "- Treat Welch-test differential expression as a Python fallback; use PyDESeq2 or R/DESeq2 for final publication-grade statistics.",
    ]
    (outdir / "autonomous_report.md").write_text("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="analysis_config.json")
    args = parser.parse_args(argv)
    import_scientific_stack()

    config_path = Path(args.config)
    config = load_config(config_path)
    design = Design(
        sample_id=config["columns"]["sample_id"],
        condition=config["columns"]["condition"],
        case_level=str(config["columns"]["case_level"]),
        control_level=str(config["columns"]["control_level"]),
        batch_columns=config["columns"].get("batch_columns") or [],
        covariates=config["columns"].get("covariates") or [],
        pair_column=config["columns"].get("pair_column"),
    )
    outdir = Path(config["analysis"]["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    counts = load_counts(Path(config["input"]["counts_file"]))
    counts = aggregate_transcripts(counts, Path(config["input"]["transcript_to_gene_file"]) if config["input"].get("transcript_to_gene_file") else None)
    metadata = read_table(Path(config["input"]["metadata_file"]))
    counts, metadata = align_counts_metadata(counts, metadata, design)
    filtered = filter_counts(
        counts,
        int(config["analysis"]["min_total_counts"]),
        int(config["analysis"]["min_samples_expressed"]),
    )
    cpm, log_expr, size_factors = normalized_expression(filtered)
    size_factors.to_csv(outdir / "size_factors.csv", header=["size_factor"])
    cpm.to_csv(outdir / "cpm.csv")
    log_expr.to_csv(outdir / "log2_median_ratio_normalized_counts.csv")

    plot_qc(filtered, log_expr, metadata, design, outdir)
    res = differential_expression_welch(log_expr, filtered, metadata, design)
    res = add_annotation(res, Path(config["input"]["gene_annotation_file"]) if config["input"].get("gene_annotation_file") else None)
    res.to_csv(outdir / "differential_expression_welch_fallback.csv")
    plot_de(res, outdir, float(config["analysis"]["fdr_threshold"]), float(config["analysis"]["log2fc_threshold"]))

    chr_summary = chromosome_dosage(
        res,
        log_expr,
        metadata,
        design,
        str(config["analysis"].get("chromosome_of_interest", "21")),
        outdir,
    )
    Path(outdir / "chromosome_dosage_summary.json").write_text(json.dumps(chr_summary, indent=2))

    marker_sets = config["analysis"].get("marker_sets") or {}
    score_marker_sets(log_expr, metadata, design, marker_sets, outdir)

    gmt_path = Path(config["input"]["gene_sets_gmt"]) if config["input"].get("gene_sets_gmt") else None
    gene_sets = read_gmt(gmt_path) if gmt_path else {}
    ora = overrepresentation(
        res,
        gene_sets,
        float(config["analysis"]["fdr_threshold"]),
        float(config["analysis"]["log2fc_threshold"]),
        outdir,
    )

    write_report(config, counts, filtered, metadata, design, res, chr_summary, ora, outdir)
    print(f"Workflow complete. See {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
