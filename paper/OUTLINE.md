# Outline: How Similar Are MLP Hidden States Across Depth?

## Title
- Emphasize the main finding: MLP hidden states change progressively rather than becoming unrelated.

## Abstract
- Context: representation similarity is widely used, but controlled MLP studies are rarer than CNN/transformer studies.
- Approach: four-layer ReLU MLPs on MNIST, Fashion-MNIST, CIFAR-10; three seeds; CKA, SVCCA, PWCCA, cosine; linear probes.
- Results: adjacent layers beat non-adjacent layers in 12/12 comparisons; same-depth cross-seed effects are weaker and mixed; later layers improve probe accuracy.
- Significance: within-model layer comparison is meaningful, but metric choice matters.

## Introduction
- Hook: interpretability and model editing assume layer transitions are incremental.
- Importance: if hidden spaces are comparable, layer-wise analysis is better grounded.
- Gap: plain MLPs have fewer controlled multi-metric studies than CNNs/transformers.
- Approach preview: fixed architecture, three datasets, four similarity metrics, probe sanity check.
- Quantitative preview: 12/12 adjacent > non-adjacent; 8/12 same-depth > off-depth; layer-4 probe gains on MNIST/Fashion-MNIST.
- Contributions:
  - controlled benchmark across three datasets
  - joint geometric and decodability analysis
  - evidence that within-model locality is stronger than cross-seed recurrence

## Related Work
- Theme 1: subspace similarity methods (SVCCA, PWCCA).
- Theme 2: kernel-style similarity and metric caveats (CKA, reliability concerns).
- Theme 3: functional compatibility and decodable information (model stitching, linear decoder perspective).
- Positioning: our work is a controlled plain-MLP benchmark combining these perspectives.

## Methodology
- Problem statement and hypotheses H1-H4.
- Datasets and preprocessing details from JSON.
- Model: four hidden layers, width 128, ReLU, dropout, AdamW, early stopping, seeds.
- Similarity metrics and comparison design:
  - within-model adjacent vs non-adjacent
  - cross-seed same-depth vs off-depth
  - linear probes by depth
- Statistical reporting: mean differences, bootstrap CIs where available, p-values, Cohen's d from summary file.

## Results
- Table 1: model accuracy summary.
- Table 2: adjacent vs non-adjacent similarity.
- Table 3: cross-seed same-depth vs off-depth.
- Figure 1: distance trends for all three datasets.
- Figure 2: probe accuracy by layer.
- Key claims:
  - strong locality within models
  - mixed recurrence across seeds
  - later layers improve decodable task information

## Discussion
- Interpretation: MLP layers refine a shared representational space.
- Metric dependence: CKA/SVCCA/PWCCA more stable than cosine.
- CIFAR-10 as the hardest case.
- Limitations: flattened image inputs, CPU-feasible subsets, scalar summaries.
- Broader implications: layer comparison, pruning, surgery, future stitching studies.

## Conclusion
- Summary of benchmark and main empirical answer.
- Main takeaway sentence: hidden spaces are not totally different.
- Future work: stitching, pre- vs post-activation, width/depth sweeps, residual/transformer extensions.

## Figures/Tables Plan
- `tables/model_accuracy.tex`
- `tables/within_model_similarity.tex`
- `tables/cross_seed_similarity.tex`
- `figures/distance_trends` via three minipages
- `figures/probe_accuracy_by_layer.png`

## Evidence Mapping
- Accuracy claims: `results/model_metrics.csv`, `results/summary.json`
- Dataset/setup claims: `results/dataset_summaries.json`, `results/environment.json`
- Within-model statistics: `results/summary.json`, `results/within_model_similarity.csv`
- Cross-seed statistics: `results/summary.json`, `results/cross_seed_similarity.csv`
- Probe claims: `results/probe_accuracy.csv`
