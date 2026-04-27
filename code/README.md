# Cloned Repositories

## Repo 1: `svcca`
- URL: https://github.com/google/svcca
- Purpose: Reference implementation of SVCCA and PWCCA-style representation comparison.
- Location: `code/svcca/`
- Key files:
  - `cca_core.py`
  - `pwcca.py`
  - `numpy_pca.py`
  - `tutorials/001_Introduction.ipynb`
- Notes:
  - Best direct baseline for comparing hidden states in this project.
  - Tutorials include examples with MNIST and SVHN activations.
  - The repo also discusses learning dynamics and freeze-training ideas derived from similarity analysis.

## Repo 2: `pytorch-model-compare`
- URL: https://github.com/AntixK/PyTorch-Model-Compare
- Purpose: Practical PyTorch implementation of minibatch CKA for comparing feature maps between models or datasets.
- Location: `code/pytorch-model-compare/`
- Key files:
  - `torch_cka/cka.py`
  - `torch_cka/utils.py`
  - `examples/mnist_test.py`
- Notes:
  - Suitable if the experiment runner trains MLPs in PyTorch and wants ready-made CKA matrices.
  - README dependencies are `torch`, `torchvision`, `tqdm`, `matplotlib`, `numpy`.
  - The bundled MNIST example is directly aligned with this workspace.

## Repo 3: `cka-similarity`
- URL: https://github.com/jayroxis/CKA-similarity
- Purpose: Lightweight NumPy and PyTorch CKA implementation with CUDA support.
- Location: `code/cka-similarity/`
- Key files:
  - `CKA.py`
  - `CKA.ipynb`
- Notes:
  - Minimal codebase that is easy to adapt into custom experiments.
  - Useful as a second implementation for cross-checking metric behavior.

## Overall Recommendation
- Start with `code/svcca` for SVCCA/PWCCA baselines.
- Use `code/pytorch-model-compare` if you want scalable minibatch CKA on trained PyTorch MLPs.
- Keep `code/cka-similarity` as a simpler fallback or validation implementation.
