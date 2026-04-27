# MLP Hidden-State Similarity

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
