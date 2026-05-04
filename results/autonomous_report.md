# RNA-seq Autonomous Analysis Report: down_syndrome_trisomic_vs_disomic_msc

## Input Audit
- Samples analyzed: 8
- Condition counts: {'disomic': 4, 'trisomic': 4}
- Features before filtering: 1000
- Features after filtering: 1000

## Differential Expression
- Significant features at FDR <= 0.05 and |log2FC| >= 1.0: 0
- Higher in trisomic: 0
- Lower in trisomic: 0

## Chromosome 21 Dosage
- Chromosome features tested: 80
- Median chromosome log2FC: 0.089
- Median non-chromosome log2FC: -0.008
- Chr-vs-nonchr log2FC Mann-Whitney p-value: 0.0343
- Per-sample chr-minus-genome score p-value: 0.0448

## Pathway Enrichment
- No local GMT enrichment results were produced.

## Autonomous Next-Step Recommendations
- Inspect PCA and sample-correlation plots for clone, batch, or differentiation-day structure.
- If a paired/isogenic design exists, add the pair column to the design before final inference.
- Confirm chromosome 21 annotation and inspect whether dosage-sensitive HSA21 genes drive the signal.
- Run pathway enrichment with curated local MSigDB/GO GMT files for reproducible interpretation.
- Treat Welch-test differential expression as a Python fallback; use PyDESeq2 or R/DESeq2 for final publication-grade statistics.
