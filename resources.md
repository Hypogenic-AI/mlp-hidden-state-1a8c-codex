# Resources Catalog

## Summary

This document catalogs the papers, datasets, and code repositories gathered for the project **How similar are MLP hidden states to each other?** The collected resources are aimed at enabling controlled experiments on layer-wise representation similarity in plain feedforward networks.

## Papers

Total papers downloaded: 7

| Title | Authors | Year | File | Key Info |
|---|---|---:|---|---|
| SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability | Raghu et al. | 2017 | `papers/1706.05806_svcca.pdf` | Foundational SVCCA method |
| Similarity of Neural Network Representations Revisited | Kornblith et al. | 2019 | `papers/1905.00414_cka_revisited.pdf` | Introduces CKA as a strong metric |
| Similarity and Matching of Neural Network Representations | Csiszárik et al. | 2021 | `papers/2110.14633_similarity_matching.pdf` | Functional similarity via stitching |
| Reliability of CKA as a Similarity Measure in Deep Learning | Davari et al. | 2022 | `papers/2210.16156_reliability_cka.pdf` | CKA caveats and failure modes |
| Similarity of Neural Network Models: A Survey of Functional and Representational Measures | Klabunde et al. | 2023 | `papers/2305.06329_similarity_survey.pdf` | Survey of metric options |
| Tracing Representation Progression: Analyzing and Enhancing Layer-Wise Similarity | Jiang et al. | 2024 | `papers/2406.14479_tracing_representation_progression.pdf` | Recent evidence of progressive inter-layer similarity |
| What Representational Similarity Measures Imply about Decodable Information | Harvey et al. | 2024 | `papers/2411.08197_decodable_information.pdf` | Links similarity to linear decodability |

See `papers/README.md` for details.

## Datasets

Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| MNIST | Hugging Face `mnist` | 60k train / 10k test | Classification | `datasets/mnist/` | Best first benchmark for MLPs |
| Fashion-MNIST | Hugging Face `fashion_mnist` | 60k train / 10k test | Classification | `datasets/fashion_mnist/` | Same shape as MNIST, harder task |
| CIFAR-10 | Hugging Face `cifar10` | 50k train / 10k test | Classification | `datasets/cifar10/` | Harder natural-image stress test |

See `datasets/README.md` for download and loading instructions.

## Code Repositories

Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| `svcca` | https://github.com/google/svcca | SVCCA and PWCCA reference code | `code/svcca/` | Best baseline for subspace comparison |
| `pytorch-model-compare` | https://github.com/AntixK/PyTorch-Model-Compare | Minibatch CKA in PyTorch | `code/pytorch-model-compare/` | Includes MNIST example |
| `cka-similarity` | https://github.com/jayroxis/CKA-similarity | Lightweight NumPy/PyTorch CKA | `code/cka-similarity/` | Good fallback implementation |

See `code/README.md` for detailed notes.

## Resource Gathering Notes

### Search Strategy

- Tried the local `paper-finder` helper first, but the backing service at `localhost:8000` did not return within the timeout window.
- Fell back to manual arXiv-based search focused on layer-wise representation similarity, SVCCA, CKA, model stitching, and metric reliability.
- Selected papers that are either foundational metrics papers or directly support the hypothesis that representations can remain similar across depth.
- Preferred benchmarks that are standard, small enough for repeated MLP runs, and easy to flatten into vector inputs.

### Selection Criteria

- Direct relevance to hidden-state or representation similarity
- Methods that can be reused in code immediately
- At least one recent paper to anchor current framing
- Datasets suitable for plain MLPs rather than architectures with strong spatial inductive bias

### Challenges Encountered

- The local `paper-finder` service timed out instead of returning a graceful fallback.
- Hugging Face image datasets required `Pillow` for sample extraction.
- Dataset sample export needed custom handling because image objects are not directly JSON serializable.

### Gaps and Workarounds

- The literature is richer for transformers and CNNs than for plain MLP-only hidden-state studies.
- To address that, this workspace emphasizes reusable methodology papers and chooses datasets that make controlled MLP experiments straightforward.

## Recommendations for Experiment Design

1. **Primary datasets**: Use `MNIST` and `Fashion-MNIST` first; reserve `CIFAR-10` for robustness checks.
2. **Baseline methods**: Report SVCCA, PWCCA, linear CKA, and sample-wise cosine similarity.
3. **Evaluation metrics**: Measure pairwise layer similarity, similarity versus depth distance, cross-seed same-layer similarity, and per-layer linear probe accuracy.
4. **Code to adapt/reuse**: Use `code/svcca/cca_core.py` and `code/pytorch-model-compare/torch_cka/cka.py` as the main implementation starting points.
