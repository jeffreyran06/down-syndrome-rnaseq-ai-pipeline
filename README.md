# Down Syndrome RNA-seq Agentic Analysis Scaffold

This workspace contains a first-pass downstream analysis scaffold for nf-core/rnaseq output, starting from a raw counts matrix and a sample metadata table.

nf-core/rnaseq performs FASTQ processing, alignment or pseudoalignment, quantification, and extensive QC. It does not perform the final statistical comparison with FDR/P-values; this scaffold starts at that downstream handoff point.

## Files

- `rnaseq_downsyndrome_analysis.py` - Jupyter percent-format Python notebook. Open in VS Code, JupyterLab with Jupytext, or paste cells into a `.ipynb`.
- `autonomous_rnaseq_workflow.py` - command-line workflow that runs the core analyses and writes a narrative Markdown report.
- `analysis_config.json` - hard-coded project/input/design settings used by default.
- `analysis_config.yaml` - equivalent YAML config for environments with PyYAML.
- `README.md` - this guide.

## Expected Inputs

Place files under `data/`, or edit `analysis_config.yaml`.

- `raw_counts.tsv`: genes/transcripts as rows, sample IDs as columns. First column should be gene/transcript ID, or row names in the first unnamed column.
- `sample_metadata.csv`: one row per biological sample. Must include `sample`, `karyotype`, and any batch/covariate columns.
- `gene_annotation.tsv` optional but strongly recommended: columns such as `gene_id`, `gene_name`, `chromosome`, `biotype`.
- `gene_sets.gmt` optional: local GMT file for pathway enrichment.
- `tx2gene.tsv` optional if transcript-level counts need aggregation: columns `transcript_id`, `gene_id`.

## Run

```bash
python3 autonomous_rnaseq_workflow.py --config analysis_config.json
```

Outputs are written to `results/`.

## Recommended Metadata Columns

For this project, include at least:

- `sample`
- `karyotype`: `trisomic` or `disomic`
- `clone` or `isogenic_pair` if samples are paired
- `batch`, `library_prep_batch`, `sequencing_lane`, or similar if applicable
- `sex`, `passage`, `differentiation_day`, and `RIN` if known
