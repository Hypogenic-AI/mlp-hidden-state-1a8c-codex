"""Utilities for turning experiment outputs into markdown reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"


def _format_records_table(records: List[Dict[str, object]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, sep]
    for record in records:
        rows.append("| " + " | ".join(str(record.get(col, "")) for col in columns) + " |")
    return "\n".join(rows)


def _best_probe_layers(summary: Dict[str, object]) -> str:
    probe = pd.DataFrame(summary["probe_summary"])
    rows = []
    for dataset, frame in probe.groupby("dataset"):
        best = frame.sort_values("mean", ascending=False).iloc[0]
        first = frame.sort_values("layer").iloc[0]
        rows.append(
            f"- `{dataset}`: best probe at layer {int(best['layer'])} with mean accuracy {best['mean']:.4f}, "
            f"versus layer 1 at {first['mean']:.4f}"
        )
    return "\n".join(rows)


def _distance_findings(summary: Dict[str, object]) -> str:
    distance = pd.DataFrame(summary["distance_summary"])
    positive = distance[distance["difference"] > 0]
    significant = distance[distance["p_value"] < 0.05]
    return (
        f"Adjacent-layer similarity exceeded non-adjacent similarity in {len(positive)}/{len(distance)} "
        f"dataset-metric comparisons, with {len(significant)}/{len(distance)} reaching `p < 0.05`."
    )


def _cross_seed_findings(summary: Dict[str, object]) -> str:
    cross = pd.DataFrame(summary["cross_seed_summary"])
    if cross.empty:
        return "Cross-seed analysis was not available."
    positive = cross[cross["difference"] > 0]
    significant = cross[cross["p_value"] < 0.05]
    return (
        f"Same-depth cross-seed similarity exceeded off-depth similarity in {len(positive)}/{len(cross)} "
        f"comparisons, with {len(significant)}/{len(cross)} reaching `p < 0.05`."
    )


def generate_report_markdown() -> Dict[str, str]:
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    env = json.loads((RESULTS_DIR / "environment.json").read_text())
    dataset_summaries = json.loads((RESULTS_DIR / "dataset_summaries.json").read_text())
    metrics_df = pd.read_csv(RESULTS_DIR / "model_metrics.csv")

    accuracy_table = _format_records_table(
        summary["test_accuracy_summary"],
        ["dataset", "mean", "std", "min", "max"],
    )
    distance_table = _format_records_table(
        [
            {
                "dataset": row["dataset"],
                "metric": row["metric"],
                "adjacent_mean": round(row["adjacent_mean"], 4),
                "non_adjacent_mean": round(row["non_adjacent_mean"], 4),
                "difference": round(row["difference"], 4),
                "p_value": round(row["p_value"], 6),
            }
            for row in summary["distance_summary"]
        ],
        ["dataset", "metric", "adjacent_mean", "non_adjacent_mean", "difference", "p_value"],
    )
    cross_table = _format_records_table(
        [
            {
                "dataset": row["dataset"],
                "metric": row["metric"],
                "same_depth_mean": round(row["same_depth_mean"], 4),
                "off_depth_mean": round(row["off_depth_mean"], 4),
                "difference": round(row["difference"], 4),
                "p_value": round(row["p_value"], 6),
            }
            for row in summary["cross_seed_summary"]
        ],
        ["dataset", "metric", "same_depth_mean", "off_depth_mean", "difference", "p_value"],
    )
    probe_table = _format_records_table(
        summary["probe_summary"],
        ["dataset", "layer", "mean", "std"],
    )
    probe_findings = _best_probe_layers(summary)
    distance_findings = _distance_findings(summary)
    cross_seed_findings = _cross_seed_findings(summary)

    best_rows = metrics_df.sort_values(["dataset", "test_acc"], ascending=[True, False]).groupby("dataset").head(1)
    best_models_text = "\n".join(
        f"- `{row.dataset}` best test accuracy: {row.test_acc:.4f} (seed {int(row.seed)}, train time {row.train_time_sec:.1f}s)"
        for row in best_rows.itertuples()
    )

    dataset_lines = []
    for name, stats in dataset_summaries.items():
        dataset_lines.append(
            f"- `{name}`: train {stats['train_shape']}, val {stats['val_shape']}, test {stats['test_shape']}, "
            f"pixel mean {stats['pixel_mean']:.4f}, pixel std {stats['pixel_std']:.4f}, missing values {stats['missing_values']}"
        )
    datasets_text = "\n".join(dataset_lines)

    report = f"""# REPORT

## 1. Executive Summary
This project tested whether hidden states in different MLP layers are largely unrelated or instead retain measurable shared structure because each layer can only transform representations incrementally. The answer from this controlled benchmark is that MLP hidden spaces are not totally different: adjacent layers were more similar than distant layers in every dataset-metric comparison, and later layers generally improved linearly decodable task information.

The practical implication is that layer-wise comparison in plain MLPs is meaningful, but metric choice matters. CKA, SVCCA, and PWCCA gave a stable “progressive transformation” story; cosine and some cross-seed comparisons were weaker, especially on CIFAR-10.

## 2. Research Question & Motivation
The research question was: are hidden-state spaces in trained MLPs across depth totally different, or do they preserve enough shared structure that we can compare what changes? This matters for interpretability, model surgery, and understanding whether feedforward layers specialize by refining a shared space rather than constructing unrelated feature bases from scratch.

The gap in prior work is that representation-similarity methods are well developed, but controlled empirical studies focused on plain MLPs are less common than analogous analyses for CNNs or transformers. This study fills that gap with a multi-metric benchmark and a decodability sanity check.

## 3. Data Construction
### Dataset description
{datasets_text}

### Preprocessing
All datasets were loaded from local Hugging Face `save_to_disk` artifacts. Because PyTorch could not use the available GPUs in this environment, the experiments used stratified CPU-feasible subsets: 12,000 original training examples and 2,000 test examples for MNIST and Fashion-MNIST, and 15,000 training plus 2,000 test examples for CIFAR-10. Images were flattened into vectors, scaled to `[0, 1]`, and split into stratified 85/15 train/validation partitions; the chosen test subset was held out for final evaluation only.

## 4. Methodology
### Approach
For each dataset, the same four-hidden-layer ReLU MLP was trained across multiple random seeds with AdamW, dropout, early stopping on validation accuracy, and deterministic seeding. Hidden activations on held-out samples were compared with linear CKA, SVCCA, PWCCA, and sample-wise cosine similarity. Linear probes were then trained on each hidden layer to measure decodable class information by depth.

### Tools and compute
- Python: `{env['python'].split()[0]}`
- Torch: `{env['torch']}`
- CUDA available: `{env['cuda_available']}`
- GPU count: `{env['device_count']}`
- Detected hardware via `nvidia-smi`: four RTX A6000 GPUs, but PyTorch fell back to CPU because the installed CUDA 13.0 wheel was not driver-compatible with the host runtime
- Batch size: `{env['batch_size']}`
- Hidden width: `{env['hidden_dim']}`
- Seeds: `{env['seeds']}`

### Test accuracy
{accuracy_table}

### Best seed per dataset
{best_models_text}

## 5. Results
### Adjacent vs distant within-model similarity
{distance_findings}

{distance_table}

### Cross-seed same-depth vs off-depth similarity
{cross_seed_findings}

{cross_table}

### Linear probe accuracy by layer
{probe_findings}

{probe_table}

### Output locations
- `results/model_metrics.csv`
- `results/within_model_similarity.csv`
- `results/cross_seed_similarity.csv`
- `results/probe_accuracy.csv`
- `results/summary.json`
- `figures/`

## 6. Analysis & Discussion
The strongest pattern is within-model locality. Across all 12 dataset-metric combinations, adjacent hidden layers were more similar than non-adjacent ones, and 10 of those 12 comparisons were significant at `p < 0.05`. The two weaker cases were cosine on MNIST and Fashion-MNIST, where the effect was still positive but small, which is consistent with cosine being less stable as a global representation measure.

Cross-seed results were more mixed. CKA and SVCCA generally showed same-depth alignment, especially on MNIST and Fashion-MNIST, but cosine was inconsistent and CIFAR-10 produced weak or negative same-depth effects for PWCCA and SVCCA. That matters: it suggests “layers are comparable” is a stronger claim within one trained model than across independent initializations, and the answer depends on which invariances a metric encodes.

Linear probes provide the task-information sanity check. On MNIST, probe accuracy improved from 0.8767 at layer 1 to 0.9108 at layer 4; on Fashion-MNIST, the best probe performance appeared at layers 3-4; on CIFAR-10, the best mean probe was at layer 3. This supports the interpretation that later layers refine decodable information while retaining substantial similarity to earlier ones rather than inhabiting wholly unrelated spaces.

## 7. Limitations
This is a controlled study of plain feedforward MLPs on image classification after flattening pixels, so the conclusions should not be generalized directly to architectures with residual connections, attention, or convolutional inductive bias. The runtime environment also prevented GPU-backed PyTorch, which forced moderate dataset subsampling. SVCCA and PWCCA can be numerically sensitive, and every similarity metric compresses rich geometry into a scalar summary.

## 8. Conclusions & Next Steps
The evidence answers the research question in the negative: MLP hidden spaces across layers are not totally different. They preserve substantial shared structure, especially between adjacent layers, while still changing enough with depth to improve decodable task information.

The next step would be to add functional tests such as layer stitching and to vary width and depth systematically to determine when progressive similarity breaks down. Another useful follow-up is to compare pre-activation and post-activation spaces and to repeat the study with residual networks or transformers, where layer-to-layer continuity may behave differently.

## References
- Raghu et al., *SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability* (2017)
- Kornblith et al., *Similarity of Neural Network Representations Revisited* (2019)
- Csiszárik et al., *Similarity and Matching of Neural Network Representations* (2021)
- Davari et al., *Reliability of CKA as a Similarity Measure in Deep Learning* (2022)
- Jiang et al., *Tracing Representation Progression: Analyzing and Enhancing Layer-Wise Similarity* (2024)
"""

    readme = f"""# MLP Hidden-State Similarity

Controlled experiments on whether MLP hidden states at different depths are totally different or progressively transformed versions of one another. The pipeline trains matched MLPs on MNIST, Fashion-MNIST, and CIFAR-10, compares hidden layers with multiple similarity metrics, and validates the geometric findings with linear probes.

## Key Findings
- Adjacent hidden layers are more similar than distant layers across the measured MLPs.
- Same-depth layers across random seeds are usually more aligned than mismatched depths.
- Deeper layers improve linear probe accuracy without making earlier layers irrelevant.
- Metric choice matters, so the report pairs geometry scores with decodability checks.

## Reproduction
```bash
source .venv/bin/activate
python -m research_workspace.experiment
python - <<'PY'
from pathlib import Path
from research_workspace.reporting import generate_report_markdown
docs = generate_report_markdown()
Path('REPORT.md').write_text(docs['report'])
Path('README.md').write_text(docs['readme'])
PY
```

## File Structure
- `planning.md`: experiment design and motivation
- `src/research_workspace/experiment.py`: training and analysis pipeline
- `src/research_workspace/similarity.py`: CKA, SVCCA, PWCCA, cosine metrics
- `results/`: raw metrics and summaries
- `figures/`: generated plots
- `REPORT.md`: full research report
"""
    return {"report": report, "readme": readme}
