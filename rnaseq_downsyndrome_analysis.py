# %% [markdown]
# # Down Syndrome Trisomic vs Disomic RNA-seq Analysis Notebook
#
# This is a hard-coded Python notebook scaffold for downstream analysis of an
# nf-core/rnaseq count matrix and sample metadata. It assumes the upstream
# FASTQ-to-count pipeline has already completed.
#
# Open this file as a Jupyter percent-format notebook in VS Code, JupyterLab
# with Jupytext, or convert it to `.ipynb` with:
#
# ```bash
# jupytext --to ipynb rnaseq_downsyndrome_analysis.py
# ```

# %%
from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

sns.set_theme(style="whitegrid", context="talk")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# %% [markdown]
# ## 1. Hard-coded project configuration
#
# Edit these constants to match your nf-core/rnaseq output files and metadata.

# %%
PROJECT_NAME = "down_syndrome_trisomic_vs_disomic_msc"

COUNTS_FILE = Path("data/raw_counts.tsv")
METADATA_FILE = Path("data/sample_metadata.csv")
GENE_ANNOTATION_FILE = Path("data/gene_annotation.tsv")  # optional but recommended
TRANSCRIPT_TO_GENE_FILE = Path("data/tx2gene.tsv")       # optional
GENE_SETS_GMT = Path("data/gene_sets.gmt")               # optional local GMT

OUTPUT_DIR = Path("results_notebook")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

SAMPLE_ID_COL = "sample"
CONDITION_COL = "karyotype"
CASE_LEVEL = "trisomic"
CONTROL_LEVEL = "disomic"

BATCH_COLS = []          # example: ["batch", "sequencing_lane"]
COVARIATES = []          # example: ["sex", "passage", "RIN"]
PAIR_COL = None          # example: "clone" for paired/isogenic designs

MIN_TOTAL_COUNTS = 10
MIN_SAMPLES_EXPRESSED = 2
TOP_VARIABLE_GENES = 5000
FDR_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 1.0
CHROMOSOME_OF_INTEREST = "21"

MARKER_SETS = {
    "pluripotency": ["POU5F1", "SOX2", "NANOG", "LIN28A", "DPPA4"],
    "mesenchymal_stem_cell": ["THY1", "ENG", "NT5E", "PDGFRA", "VIM", "COL1A1", "COL1A2", "DCN", "LUM"],
    "neural_crest_mesenchymal": ["SOX9", "TWIST1", "SNAI2", "CDH2"],
}

# %% [markdown]
# ## 2. Load counts, metadata, and optional annotations

# %%
def read_table(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def load_counts(path):
    df = read_table(path)
    first_col = df.columns[0]
    if first_col.lower() in {"gene", "gene_id", "id", "transcript", "transcript_id", "unnamed: 0"}:
        df = df.set_index(first_col)
    df.index = df.index.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    df = df.loc[df.sum(axis=1) > 0]
    return df.round().astype(int)


counts_raw = load_counts(COUNTS_FILE)
metadata = read_table(METADATA_FILE)

if TRANSCRIPT_TO_GENE_FILE.exists():
    tx2gene = read_table(TRANSCRIPT_TO_GENE_FILE)
    tx_col = "transcript_id" if "transcript_id" in tx2gene.columns else tx2gene.columns[0]
    gene_col = "gene_id" if "gene_id" in tx2gene.columns else tx2gene.columns[1]
    mapper = tx2gene[[tx_col, gene_col]].dropna().drop_duplicates().set_index(tx_col)[gene_col]
    mapped = counts_raw.join(mapper.rename("gene_id"), how="inner")
    if not mapped.empty:
        counts_raw = mapped.groupby("gene_id").sum()

metadata[SAMPLE_ID_COL] = metadata[SAMPLE_ID_COL].astype(str)
metadata = metadata.set_index(SAMPLE_ID_COL, drop=False)
shared_samples = [s for s in counts_raw.columns.astype(str) if s in metadata.index]
counts_raw = counts_raw.loc[:, shared_samples]
metadata = metadata.loc[shared_samples].copy()

assert CONDITION_COL in metadata.columns, f"Missing metadata condition column: {CONDITION_COL}"
assert {CASE_LEVEL, CONTROL_LEVEL}.issubset(set(metadata[CONDITION_COL].astype(str))), "Condition levels are missing."

counts_raw.shape, metadata.shape

# %%
annotation = None
if GENE_ANNOTATION_FILE.exists():
    annotation = read_table(GENE_ANNOTATION_FILE)
    key = "gene_id" if "gene_id" in annotation.columns else annotation.columns[0]
    annotation[key] = annotation[key].astype(str)
    annotation = annotation.drop_duplicates(key).set_index(key)
    display(annotation.head())

display(metadata.head())
display(counts_raw.iloc[:5, :5])

# %% [markdown]
# ## 3. Input QC and low-count filtering

# %%
sample_qc = pd.DataFrame({
    "sample": counts_raw.columns,
    "library_size": counts_raw.sum(axis=0).values,
    "detected_features": (counts_raw > 0).sum(axis=0).values,
    CONDITION_COL: metadata[CONDITION_COL].astype(str).values,
})
sample_qc.to_csv(OUTPUT_DIR / "sample_qc_metrics.csv", index=False)
display(sample_qc)

plt.figure(figsize=(10, 4))
sns.barplot(data=sample_qc, x="sample", y="library_size", hue=CONDITION_COL)
plt.xticks(rotation=45, ha="right")
plt.title("Library size per sample")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "library_sizes.png", dpi=180)
plt.show()

keep = (counts_raw.sum(axis=1) >= MIN_TOTAL_COUNTS) & ((counts_raw > 0).sum(axis=1) >= MIN_SAMPLES_EXPRESSED)
counts = counts_raw.loc[keep].copy()
print(f"Kept {counts.shape[0]:,} of {counts_raw.shape[0]:,} features after low-count filtering.")

# %% [markdown]
# ## 4. Normalization
#
# This notebook computes CPM and DESeq2-style median-ratio size-factor
# normalization. For publication-grade DE statistics, prefer PyDESeq2 or R/DESeq2
# with the correct paired/batch design. A Welch-test fallback is included so the
# notebook remains runnable in a minimal Python environment.

# %%
def size_factors_median_ratio(counts):
    positive = counts.replace(0, np.nan)
    geo_means = np.exp(np.log(positive).mean(axis=1))
    valid = np.isfinite(geo_means) & (geo_means > 0)
    ratios = counts.loc[valid].div(geo_means[valid], axis=0)
    size_factors = ratios.replace(0, np.nan).median(axis=0)
    return size_factors.fillna(size_factors.median())


library_sizes = counts.sum(axis=0)
cpm = counts.div(library_sizes, axis=1) * 1_000_000
size_factors = size_factors_median_ratio(counts)
norm_counts = counts.div(size_factors, axis=1)
log_expr = np.log2(norm_counts + 1)

cpm.to_csv(OUTPUT_DIR / "cpm.csv")
log_expr.to_csv(OUTPUT_DIR / "log2_median_ratio_normalized_counts.csv")
size_factors.to_csv(OUTPUT_DIR / "size_factors.csv", header=["size_factor"])

display(size_factors.to_frame("size_factor"))

# %% [markdown]
# ## 5. Sample-level structure: PCA, correlation, clustering, batch checks

# %%
top_var = log_expr.var(axis=1).sort_values(ascending=False).head(min(TOP_VARIABLE_GENES, log_expr.shape[0])).index
pca_matrix = log_expr.loc[top_var].T
pca_scaled = StandardScaler().fit_transform(pca_matrix)
pca = PCA(n_components=min(5, pca_scaled.shape[0], pca_scaled.shape[1]))
pcs = pca.fit_transform(pca_scaled)

pca_df = pd.DataFrame(pcs[:, :2], index=pca_matrix.index, columns=["PC1", "PC2"]).join(metadata)
pca_df.to_csv(OUTPUT_DIR / "pca_sample_scores.csv")

plt.figure(figsize=(7, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=CONDITION_COL, s=110)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
plt.title("PCA of normalized expression")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca.png", dpi=180)
plt.show()

corr = log_expr.corr(method="spearman")
order = leaves_list(linkage(pdist(corr), method="average")) if corr.shape[0] > 2 else np.arange(corr.shape[0])
plt.figure(figsize=(7, 6))
sns.heatmap(corr.iloc[order, order], cmap="vlag", vmin=0.5, vmax=1.0)
plt.title("Sample Spearman correlation")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sample_correlation_heatmap.png", dpi=180)
plt.show()

# %%
for col in [CONDITION_COL] + BATCH_COLS + COVARIATES:
    if col in metadata.columns:
        groups = metadata[col].astype(str)
        if groups.nunique() > 1:
            print(f"{col}:")
            print(pca_df.groupby(groups)["PC1"].describe()[["count", "mean", "std"]])

# %% [markdown]
# ## 6. Differential expression
#
# The first cell tries PyDESeq2. If it is not installed, the second cell runs a
# Welch-test fallback on log-normalized expression. The fallback is useful for
# agentic triage but should not replace DESeq2/edgeR/limma-voom for final claims.

# %%
def run_pydeseq2(counts, metadata):
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        return None

    design_factors = [CONDITION_COL] + BATCH_COLS + COVARIATES
    design_factors = [c for c in design_factors if c in metadata.columns]
    dds = DeseqDataSet(
        counts=counts.T.astype(int),
        metadata=metadata.loc[counts.columns],
        design_factors=design_factors,
        refit_cooks=True,
    )
    dds.deseq2()
    stat_res = DeseqStats(dds, contrast=(CONDITION_COL, CASE_LEVEL, CONTROL_LEVEL))
    stat_res.summary()
    out = stat_res.results_df.copy()
    out.index.name = "gene_id"
    return out.sort_values(["padj", "pvalue"], na_position="last")


deseq2_results = run_pydeseq2(counts, metadata)
if deseq2_results is None:
    print("PyDESeq2 is not installed; using Welch-test fallback in the next cell.")
else:
    print("PyDESeq2 completed.")

# %%
def run_welch_fallback(log_expr, counts, metadata):
    condition = metadata[CONDITION_COL].astype(str)
    case_samples = metadata.index[condition == CASE_LEVEL].tolist()
    control_samples = metadata.index[condition == CONTROL_LEVEL].tolist()
    rows = []
    for gene, values in log_expr.iterrows():
        case = values[case_samples].astype(float)
        ctrl = values[control_samples].astype(float)
        stat, pvalue = stats.ttest_ind(case, ctrl, equal_var=False, nan_policy="omit")
        rows.append({
            "gene_id": gene,
            "baseMean": counts.loc[gene].mean(),
            "log2FoldChange": case.mean() - ctrl.mean(),
            "stat": stat,
            "pvalue": pvalue,
        })
    res = pd.DataFrame(rows).set_index("gene_id")
    ok = res["pvalue"].notna()
    res["padj"] = np.nan
    res.loc[ok, "padj"] = multipletests(res.loc[ok, "pvalue"], method="fdr_bh")[1]
    return res.sort_values(["padj", "pvalue"], na_position="last")


de_results = deseq2_results if deseq2_results is not None else run_welch_fallback(log_expr, counts, metadata)

if annotation is not None:
    de_results = de_results.join(annotation, how="left")
    if "gene_name" in de_results.columns:
        cols = ["gene_name"] + [c for c in de_results.columns if c != "gene_name"]
        de_results = de_results[cols]

de_results.to_csv(OUTPUT_DIR / "differential_expression_results.csv")
display(de_results.head(20))

# %%
sig = de_results[(de_results["padj"] <= FDR_THRESHOLD) & (de_results["log2FoldChange"].abs() >= LOG2FC_THRESHOLD)]
print(f"Significant features: {sig.shape[0]:,}")
print(f"Higher in {CASE_LEVEL}: {(sig['log2FoldChange'] > 0).sum():,}")
print(f"Lower in {CASE_LEVEL}: {(sig['log2FoldChange'] < 0).sum():,}")

volcano = de_results.copy()
volcano["minus_log10_padj"] = -np.log10(volcano["padj"].clip(lower=np.nextafter(0, 1)))
volcano["significant"] = (volcano["padj"] <= FDR_THRESHOLD) & (volcano["log2FoldChange"].abs() >= LOG2FC_THRESHOLD)

plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=volcano,
    x="log2FoldChange",
    y="minus_log10_padj",
    hue="significant",
    palette={False: "#7f8c8d", True: "#c0392b"},
    s=16,
    linewidth=0,
)
plt.axvline(LOG2FC_THRESHOLD, color="#34495e", lw=1, ls="--")
plt.axvline(-LOG2FC_THRESHOLD, color="#34495e", lw=1, ls="--")
plt.axhline(-math.log10(FDR_THRESHOLD), color="#34495e", lw=1, ls="--")
plt.title("Differential expression volcano")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "volcano.png", dpi=180)
plt.show()

# %% [markdown]
# ## 7. Chromosome 21 dosage and dosage compensation

# %%
chr_summary = {"available": False}
if "chromosome" in de_results.columns:
    chrom = de_results["chromosome"].astype(str).str.replace("chr", "", regex=False)
    chr_mask = chrom == str(CHROMOSOME_OF_INTEREST).replace("chr", "")
    non_chr_mask = ~chr_mask & chrom.notna()
    chr_genes = de_results.index[chr_mask].intersection(log_expr.index)
    non_chr_genes = de_results.index[non_chr_mask].intersection(log_expr.index)

    if len(chr_genes) >= 5 and len(non_chr_genes) >= 5:
        score = pd.DataFrame({
            "sample": log_expr.columns,
            "chr_score": log_expr.loc[chr_genes].mean(axis=0).values,
            "genome_score": log_expr.loc[non_chr_genes].mean(axis=0).values,
            CONDITION_COL: metadata[CONDITION_COL].astype(str).values,
        })
        score["chr_minus_genome"] = score["chr_score"] - score["genome_score"]
        score.to_csv(OUTPUT_DIR / f"chromosome_{CHROMOSOME_OF_INTEREST}_dosage_scores.csv", index=False)

        plt.figure(figsize=(5.5, 4.5))
        sns.boxplot(data=score, x=CONDITION_COL, y="chr_minus_genome", color="#d5dbdb")
        sns.stripplot(data=score, x=CONDITION_COL, y="chr_minus_genome", hue=CONDITION_COL, s=8)
        plt.title(f"Chr{CHROMOSOME_OF_INTEREST} dosage score")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"chromosome_{CHROMOSOME_OF_INTEREST}_dosage_score.png", dpi=180)
        plt.show()

        mw = stats.mannwhitneyu(
            de_results.loc[chr_mask, "log2FoldChange"].dropna(),
            de_results.loc[non_chr_mask, "log2FoldChange"].dropna(),
            alternative="two-sided",
        )
        chr_summary = {
            "available": True,
            "n_chr_features": int(chr_mask.sum()),
            "median_chr_log2fc": float(de_results.loc[chr_mask, "log2FoldChange"].median()),
            "median_non_chr_log2fc": float(de_results.loc[non_chr_mask, "log2FoldChange"].median()),
            "chr_vs_nonchr_mannwhitney_pvalue": float(mw.pvalue),
        }

chr_summary

# %% [markdown]
# ## 8. Pathway enrichment with a local GMT file
#
# This avoids network-dependent enrichment. Provide MSigDB, GO, Reactome, KEGG,
# Hallmark, or custom Down Syndrome/MSC gene sets as `data/gene_sets.gmt`.

# %%
def read_gmt(path):
    gene_sets = {}
    if not path.exists():
        return gene_sets
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                gene_sets[parts[0]] = sorted(set(parts[2:]))
    return gene_sets


def run_ora(de_results, gene_sets):
    if not gene_sets:
        return pd.DataFrame()
    gene_col = "gene_name" if "gene_name" in de_results.columns else None
    universe = set(de_results[gene_col].dropna().astype(str)) if gene_col else set(de_results.index.astype(str))
    sig_mask = (de_results["padj"] <= FDR_THRESHOLD) & (de_results["log2FoldChange"].abs() >= LOG2FC_THRESHOLD)
    sig_genes = set(de_results.loc[sig_mask, gene_col].dropna().astype(str)) if gene_col else set(de_results.loc[sig_mask].index.astype(str))
    rows = []
    for term, genes in gene_sets.items():
        gs = set(genes) & universe
        if len(gs) < 5 or not sig_genes:
            continue
        a = len(sig_genes & gs)
        b = len(sig_genes - gs)
        c = len(gs - sig_genes)
        d = len(universe - sig_genes - gs)
        odds, pvalue = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append({"term": term, "overlap": a, "set_size": len(gs), "odds_ratio": odds, "pvalue": pvalue})
    ora = pd.DataFrame(rows)
    if ora.empty:
        return ora
    ora["padj"] = multipletests(ora["pvalue"], method="fdr_bh")[1]
    return ora.sort_values(["padj", "pvalue"])


gene_sets = read_gmt(GENE_SETS_GMT)
ora = run_ora(de_results, gene_sets)
if ora.empty:
    print("No ORA results. Add a local GMT file or relax thresholds.")
else:
    ora.to_csv(OUTPUT_DIR / "pathway_overrepresentation.csv", index=False)
    display(ora.head(20))

# %% [markdown]
# ## 9. Marker set scoring for differentiation state

# %%
marker_rows = []
symbol_index = pd.Index(log_expr.index.astype(str))
for set_name, genes in MARKER_SETS.items():
    present = [gene for gene in genes if gene in symbol_index]
    if not present:
        continue
    z = pd.DataFrame(StandardScaler().fit_transform(log_expr.loc[present].T), index=log_expr.columns, columns=present)
    score = z.mean(axis=1)
    for sample, value in score.items():
        marker_rows.append({"sample": sample, "marker_set": set_name, "score": value, CONDITION_COL: metadata.loc[sample, CONDITION_COL]})

marker_scores = pd.DataFrame(marker_rows)
if marker_scores.empty:
    print("No marker genes matched expression row IDs. If rows are Ensembl IDs, provide gene symbols or map gene IDs.")
else:
    marker_scores.to_csv(OUTPUT_DIR / "marker_set_scores.csv", index=False)
    plt.figure(figsize=(9, 4.8))
    sns.boxplot(data=marker_scores, x="marker_set", y="score", hue=CONDITION_COL)
    plt.xticks(rotation=30, ha="right")
    plt.title("Lineage marker set scores")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "marker_set_scores.png", dpi=180)
    plt.show()

# %% [markdown]
# ## 10. Co-expression modules
#
# This is a lightweight WGCNA-like screen based on hierarchical clustering of the
# most variable genes. Use real WGCNA/hdWGCNA for final module analysis.

# %%
n_module_genes = min(2000, log_expr.shape[0])
module_genes = log_expr.var(axis=1).sort_values(ascending=False).head(n_module_genes).index
module_expr = log_expr.loc[module_genes]

gene_corr_dist = pdist(module_expr, metric="correlation")
gene_linkage = linkage(gene_corr_dist, method="average")
module_labels = fcluster(gene_linkage, t=0.7, criterion="distance")

module_table = pd.DataFrame({"gene_id": module_genes, "module": module_labels}).set_index("gene_id")
module_scores = []
for module_id, genes in module_table.groupby("module").groups.items():
    if len(genes) < 10:
        continue
    z = pd.DataFrame(StandardScaler().fit_transform(log_expr.loc[list(genes)].T), index=log_expr.columns, columns=list(genes))
    eigengene = z.mean(axis=1)
    case = eigengene[metadata[CONDITION_COL].astype(str) == CASE_LEVEL]
    ctrl = eigengene[metadata[CONDITION_COL].astype(str) == CONTROL_LEVEL]
    stat, pvalue = stats.ttest_ind(case, ctrl, equal_var=False)
    module_scores.append({
        "module": module_id,
        "n_genes": len(genes),
        "case_minus_control": case.mean() - ctrl.mean(),
        "pvalue": pvalue,
    })

module_scores = pd.DataFrame(module_scores).sort_values("pvalue") if module_scores else pd.DataFrame()
if not module_scores.empty:
    module_scores["padj"] = multipletests(module_scores["pvalue"], method="fdr_bh")[1]
    module_scores.to_csv(OUTPUT_DIR / "coexpression_module_scores.csv", index=False)
    module_table.to_csv(OUTPUT_DIR / "coexpression_module_membership.csv")
    display(module_scores.head(20))

# %% [markdown]
# ## 11. Isoform/transcript usage screen
#
# If your input rows are transcripts and `TRANSCRIPT_TO_GENE_FILE` exists, run
# this before aggregation in a separate copy of the raw transcript matrix. This is
# a triage screen based on transcript proportions, not a replacement for DRIMSeq,
# DEXSeq, satuRn, or IsoformSwitchAnalyzeR.

# %%
if TRANSCRIPT_TO_GENE_FILE.exists():
    tx_counts = load_counts(COUNTS_FILE)
    tx2gene = read_table(TRANSCRIPT_TO_GENE_FILE)
    tx_col = "transcript_id" if "transcript_id" in tx2gene.columns else tx2gene.columns[0]
    gene_col = "gene_id" if "gene_id" in tx2gene.columns else tx2gene.columns[1]
    mapper = tx2gene[[tx_col, gene_col]].dropna().drop_duplicates().set_index(tx_col)[gene_col]
    tx_counts = tx_counts.loc[tx_counts.index.intersection(mapper.index), shared_samples]
    tx_gene = mapper.loc[tx_counts.index]
    gene_totals = tx_counts.groupby(tx_gene).transform("sum")
    tx_prop = (tx_counts / gene_totals.replace(0, np.nan)).fillna(0)
    condition = metadata[CONDITION_COL].astype(str)
    case_samples = metadata.index[condition == CASE_LEVEL].tolist()
    control_samples = metadata.index[condition == CONTROL_LEVEL].tolist()
    prop_delta = tx_prop[case_samples].mean(axis=1) - tx_prop[control_samples].mean(axis=1)
    usage = pd.DataFrame({"gene_id": tx_gene, "delta_transcript_fraction": prop_delta})
    usage = usage.reindex(usage["delta_transcript_fraction"].abs().sort_values(ascending=False).index)
    usage.to_csv(OUTPUT_DIR / "transcript_usage_triage.csv")
    display(usage.head(20))
else:
    print("Transcript usage screen skipped because TRANSCRIPT_TO_GENE_FILE is not present.")

# %% [markdown]
# ## 12. Autonomous interpretation report

# %%
report = []
report.append(f"# RNA-seq Report: {PROJECT_NAME}")
report.append("")
report.append("## Input")
report.append(f"- Samples: {metadata.shape[0]}")
report.append(f"- Raw features: {counts_raw.shape[0]}")
report.append(f"- Filtered features: {counts.shape[0]}")
report.append(f"- Conditions: {metadata[CONDITION_COL].astype(str).value_counts().to_dict()}")
report.append("")
report.append("## Differential Expression")
report.append(f"- Significant features at FDR <= {FDR_THRESHOLD} and |log2FC| >= {LOG2FC_THRESHOLD}: {sig.shape[0]}")
report.append(f"- Higher in {CASE_LEVEL}: {(sig['log2FoldChange'] > 0).sum()}")
report.append(f"- Lower in {CASE_LEVEL}: {(sig['log2FoldChange'] < 0).sum()}")

if not sig.empty:
    label_col = "gene_name" if "gene_name" in sig.columns else None
    report.append("")
    report.append(f"Top features higher in {CASE_LEVEL}:")
    for gene, row in sig.sort_values("log2FoldChange", ascending=False).head(10).iterrows():
        label = row[label_col] if label_col and pd.notna(row[label_col]) else gene
        report.append(f"- {label}: log2FC {row['log2FoldChange']:.2f}, padj {row['padj']:.2g}")
    report.append("")
    report.append(f"Top features lower in {CASE_LEVEL}:")
    for gene, row in sig.sort_values("log2FoldChange").head(10).iterrows():
        label = row[label_col] if label_col and pd.notna(row[label_col]) else gene
        report.append(f"- {label}: log2FC {row['log2FoldChange']:.2f}, padj {row['padj']:.2g}")

report.append("")
report.append("## Chromosome 21")
if chr_summary.get("available"):
    report.append(f"- Chr{CHROMOSOME_OF_INTEREST} features: {chr_summary['n_chr_features']}")
    report.append(f"- Median Chr{CHROMOSOME_OF_INTEREST} log2FC: {chr_summary['median_chr_log2fc']:.3f}")
    report.append(f"- Median non-Chr{CHROMOSOME_OF_INTEREST} log2FC: {chr_summary['median_non_chr_log2fc']:.3f}")
    report.append(f"- Chr-vs-nonchr Mann-Whitney p-value: {chr_summary['chr_vs_nonchr_mannwhitney_pvalue']:.3g}")
else:
    report.append("- Chromosome analysis skipped; provide annotation with a `chromosome` column.")

report.append("")
report.append("## Recommended follow-up")
report.append("- Re-run DE with a paired/clone-aware DESeq2 design if isogenic pairing exists.")
report.append("- Compare results with edgeR quasi-likelihood and limma-voom as sensitivity analyses.")
report.append("- Add curated GMT files for GO, Hallmark, Reactome, interferon, extracellular matrix, cell-cycle, and Down Syndrome gene sets.")
report.append("- Validate chromosome 21 dosage, global transcriptional amplification/compensation, and differentiation marker status before biological interpretation.")

(OUTPUT_DIR / "notebook_autonomous_report.md").write_text("\n".join(report) + "\n")
print("\n".join(report[:40]))

