# Literature Review

## Research Area Overview

This project sits at the intersection of representation analysis and neural network interpretability. The central question is whether hidden states from different MLP layers are largely distinct or whether they retain measurable shared structure because each layer only incrementally transforms the representation. The literature strongly suggests that hidden representations are often neither identical nor unrelated: nearby layers tend to remain correlated, deeper layers become more task-specific, and conclusions depend heavily on which similarity metric is used.

## Search Keywords

- MLP hidden state similarity
- layer-wise representation similarity
- SVCCA neural networks
- CKA hidden layer comparison
- neural representation matching
- functional similarity model stitching
- decodable information representation similarity

## Key Papers

### 1. SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability
- **Authors**: Maithra Raghu, Justin Gilmer, Jason Yosinski, Jascha Sohl-Dickstein
- **Year**: 2017
- **Source**: NeurIPS 2017 / arXiv `1706.05806`
- **Key Contribution**: Introduces SVCCA for comparing hidden representations across layers, checkpoints, and models while being robust to affine transformations.
- **Methodology**: Applies SVD to retain dominant directions, then CCA to compare the resulting subspaces.
- **Datasets Used**: CIFAR-10, toy regression, language-model settings, and ImageNet-based interpretability examples.
- **Results**: Finds that lower layers converge earlier, learned representations are often lower-dimensional than layer width suggests, and class sensitivity emerges progressively.
- **Code Available**: Yes, `google/svcca`.
- **Relevance to Our Research**: This is the most natural baseline for an MLP hidden-state study because it directly compares internal subspaces, not just per-neuron correspondence.

### 2. Similarity of Neural Network Representations Revisited
- **Authors**: Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton
- **Year**: 2019
- **Source**: ICML 2019 / arXiv `1905.00414`
- **Key Contribution**: Shows why CCA-style measures can fail in high-dimensional settings and argues for CKA as a better representation-level similarity metric.
- **Methodology**: Theoretical analysis of invariance properties plus empirical correspondence tests across trained networks.
- **Datasets Used**: CIFAR-10 and Transformer encoder sanity checks.
- **Results**: Linear and RBF CKA recover matching layers far better than CCA/SVCCA in their correspondence benchmark.
- **Code Available**: Reference code linked from the project page; several community implementations exist.
- **Relevance to Our Research**: Strong metric choice paper. If the goal is a layer-by-layer similarity matrix for trained MLPs, CKA should be one of the default comparisons.

### 3. Similarity and Matching of Neural Network Representations
- **Authors**: Adrián Csiszárik, Péter Kőrösi-Szabó, Ákos K. Matszangosz, Gergely Papp, Dániel Varga
- **Year**: 2021
- **Source**: NeurIPS 2021 / arXiv `2110.14633`
- **Key Contribution**: Distinguishes representational similarity from functional similarity through model stitching.
- **Methodology**: Connects one model’s intermediate activations into another model using affine stitching layers and evaluates task performance.
- **Datasets Used**: Vision benchmarks with convolutional models.
- **Results**: Functionally compatible representations can exist even when geometric similarity scores are not obviously large.
- **Code Available**: Not identified during this pass.
- **Relevance to Our Research**: Important warning that geometry alone is not the full story. For MLPs, a small stitching experiment between layers or seeds would be a valuable extension.

### 4. Reliability of CKA as a Similarity Measure in Deep Learning
- **Authors**: MohammadReza Davari, Stefan Horoi, Amine Natik, Guillaume Lajoie, Guy Wolf, Eugene Belilovsky
- **Year**: 2022
- **Source**: ICLR 2023 poster / arXiv `2210.16156`
- **Key Contribution**: Shows that CKA can be sensitive to outliers and certain simple transformations, even when useful task information is preserved.
- **Methodology**: Theoretical characterization plus empirical manipulation of representations and linear probe analyses.
- **Datasets Used**: CIFAR-10 and vision transformer settings.
- **Results**: High CKA can appear in cases where feature usefulness differs, especially in early layers; CKA values can be manipulated without strongly changing final behavior.
- **Code Available**: Not identified in this pass.
- **Relevance to Our Research**: Essential caveat paper. It argues against trusting a single metric and supports combining CKA with probes, stitching, or classifier transfer.

### 5. Similarity of Neural Network Models: A Survey of Functional and Representational Measures
- **Authors**: Max Klabunde, Tobias Schumacher, Markus Strohmaier, Florian Lemmerich
- **Year**: 2023
- **Source**: Survey / arXiv `2305.06329`
- **Key Contribution**: Organizes the space of representational and functional similarity measures and summarizes known tradeoffs.
- **Methodology**: Literature survey spanning more than thirty measures and multiple application settings.
- **Datasets Used**: Survey paper, not a single benchmark contribution.
- **Results**: Clarifies that different measures encode different invariances and answer different scientific questions.
- **Code Available**: Not central.
- **Relevance to Our Research**: Useful framing document for explaining why this project should report more than one similarity notion.

### 6. Tracing Representation Progression: Analyzing and Enhancing Layer-Wise Similarity
- **Authors**: Jiachen Jiang, Jinxin Zhou, Zhihui Zhu
- **Year**: 2024
- **Source**: arXiv `2406.14479`
- **Key Contribution**: Recent evidence that hidden representations across layers are positively correlated and become more similar as layer distance decreases.
- **Methodology**: Uses sample-wise cosine similarity, compares it against CKA, studies layer-wise classifier transfer and aligned training.
- **Datasets Used**: CIFAR-10, ImageNet-1K, and NLP tasks on transformers.
- **Results**: Nearby layers are more similar; stronger similarity correlates with more stable predictions and better shallow-layer reuse.
- **Code Available**: Not identified during this pass.
- **Relevance to Our Research**: Although transformer-focused, its main empirical claim is directly aligned with the hypothesis here: layer transitions are often incremental, not arbitrary.

### 7. What Representational Similarity Measures Imply about Decodable Information
- **Authors**: Sarah E. Harvey, David Lipshutz, Alex H. Williams
- **Year**: 2024
- **Source**: UniReps 2024 / arXiv `2411.08197`
- **Key Contribution**: Interprets common similarity measures through the lens of linear decoding.
- **Methodology**: Theoretical analysis linking CKA, CCA, and Procrustes-style distances to alignment between optimal linear readouts.
- **Datasets Used**: Theory-heavy paper rather than a benchmark-focused one.
- **Results**: Similar geometry often implies alignment of downstream linear decoders, but only under explicit assumptions.
- **Code Available**: Not central.
- **Relevance to Our Research**: Useful for motivating linear-probe checks on each hidden layer in addition to pairwise similarity scores.

## Common Methodologies

- **SVCCA / PWCCA**: Compare low-dimensional subspaces after removing noisy directions.
- **CKA**: Compare representational similarity matrices or feature kernels; useful for correspondence heatmaps.
- **Sample-wise cosine similarity**: Cheap metric for within-model adjacent-layer analysis.
- **Model stitching**: Tests functional compatibility instead of only geometry.
- **Linear probes / transferred classifiers**: Measure whether similar hidden states preserve decodable task information.

## Standard Baselines

- **SVCCA**: Foundational baseline, especially for subspace-level comparison.
- **Linear CKA**: Strong default for high-dimensional representation comparison.
- **PWCCA**: Variant that weights canonical directions by contribution.
- **Cosine similarity between paired activations**: Good lightweight baseline for nearby layers.
- **Linear probe accuracy**: Not a similarity metric, but an essential sanity check on representation usefulness.

## Evaluation Metrics

- **Pairwise layer similarity matrix**: Main output for comparing all hidden layers with one another.
- **Adjacent-layer similarity trend**: Tests whether similarity decreases smoothly with layer distance.
- **Cross-seed similarity**: Checks whether learned geometry is stable across random initializations.
- **Linear probe accuracy per layer**: Measures task information at each depth.
- **Classifier transfer from final layer to earlier layers**: Useful if testing decodable-information claims.

## Datasets in the Literature and Recommended Benchmarks

- **MNIST**: Best low-cost starting point for dense MLPs.
- **Fashion-MNIST**: Same shape as MNIST but more semantically difficult.
- **CIFAR-10**: Harder benchmark that can reveal whether similarity patterns survive under weaker MLP inductive bias.

These three benchmarks are enough to test whether hidden-state similarity trends are stable across difficulty levels.

## Gaps and Opportunities

- Most recent layer-similarity work emphasizes transformers, not plain MLPs.
- Functional similarity tests such as stitching are less common in simple feedforward MLP studies and would be a worthwhile extension.
- Metric sensitivity remains a real issue, so conclusions should not rely on one score alone.
- There is room for a clean controlled study where architecture, optimizer, width, and dataset difficulty are varied systematically in plain MLPs.

## Recommendations for Our Experiment

- **Recommended datasets**: Start with `MNIST` and `Fashion-MNIST`, then validate on `CIFAR-10`.
- **Recommended baselines**: Linear CKA, SVCCA, PWCCA, and sample-wise cosine similarity.
- **Recommended metrics**: Pairwise hidden-layer similarity matrices, similarity versus layer distance, and linear probe accuracy per layer.
- **Methodological considerations**:
  - Train multiple seeds per architecture.
  - Compare both within-model layer pairs and cross-model same-depth pairs.
  - Normalize hidden activations consistently before similarity analysis.
  - Interpret CKA with caution and pair it with either linear probes or simple stitching.
  - Record width and depth explicitly, because similarity behavior changes with overparameterization.
