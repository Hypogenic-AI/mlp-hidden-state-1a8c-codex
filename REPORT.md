# REPORT

## 1. Executive Summary
This project tested whether hidden states in different MLP layers are largely unrelated or instead retain measurable shared structure because each layer can only transform representations incrementally. The answer from this controlled benchmark is that MLP hidden spaces are not totally different: adjacent layers were more similar than distant layers in every dataset-metric comparison, and later layers generally improved linearly decodable task information.

The practical implication is that layer-wise comparison in plain MLPs is meaningful, but metric choice matters. CKA, SVCCA, and PWCCA gave a stable “progressive transformation” story; cosine and some cross-seed comparisons were weaker, especially on CIFAR-10.

## 2. Research Question & Motivation
The research question was: are hidden-state spaces in trained MLPs across depth totally different, or do they preserve enough shared structure that we can compare what changes? This matters for interpretability, model surgery, and understanding whether feedforward layers specialize by refining a shared space rather than constructing unrelated feature bases from scratch.

The gap in prior work is that representation-similarity methods are well developed, but controlled empirical studies focused on plain MLPs are less common than analogous analyses for CNNs or transformers. This study fills that gap with a multi-metric benchmark and a decodability sanity check.

## 3. Data Construction
### Dataset description
- `mnist`: train [10200, 784], val [1800, 784], test [2000, 784], pixel mean 0.1303, pixel std 0.3077, missing values 0
- `fashion_mnist`: train [10200, 784], val [1800, 784], test [2000, 784], pixel mean 0.2865, pixel std 0.3539, missing values 0
- `cifar10`: train [12750, 3072], val [2250, 3072], test [2000, 3072], pixel mean 0.4729, pixel std 0.2512, missing values 0

### Preprocessing
All datasets were loaded from local Hugging Face `save_to_disk` artifacts. Because PyTorch could not use the available GPUs in this environment, the experiments used stratified CPU-feasible subsets: 12,000 original training examples and 2,000 test examples for MNIST and Fashion-MNIST, and 15,000 training plus 2,000 test examples for CIFAR-10. Images were flattened into vectors, scaled to `[0, 1]`, and split into stratified 85/15 train/validation partitions; the chosen test subset was held out for final evaluation only.

## 4. Methodology
### Approach
For each dataset, the same four-hidden-layer ReLU MLP was trained across multiple random seeds with AdamW, dropout, early stopping on validation accuracy, and deterministic seeding. Hidden activations on held-out samples were compared with linear CKA, SVCCA, PWCCA, and sample-wise cosine similarity. Linear probes were then trained on each hidden layer to measure decodable class information by depth.

### Tools and compute
- Python: `3.12.8`
- Torch: `2.11.0+cu130`
- CUDA available: `False`
- GPU count: `4`
- Detected hardware via `nvidia-smi`: four RTX A6000 GPUs, but PyTorch fell back to CPU because the installed CUDA 13.0 wheel was not driver-compatible with the host runtime
- Batch size: `512`
- Hidden width: `128`
- Seeds: `[42, 123, 456]`

### Test accuracy
| dataset | mean | std | min | max |
| --- | --- | --- | --- | --- |
| cifar10 | 0.3937 | 0.0033 | 0.39 | 0.3965 |
| fashion_mnist | 0.8228 | 0.0098 | 0.812 | 0.831 |
| mnist | 0.927 | 0.0039 | 0.9245 | 0.9315 |

### Best seed per dataset
- `cifar10` best test accuracy: 0.3965 (seed 123, train time 4.3s)
- `fashion_mnist` best test accuracy: 0.8310 (seed 123, train time 2.4s)
- `mnist` best test accuracy: 0.9315 (seed 456, train time 1.8s)

## 5. Results
### Adjacent vs distant within-model similarity
Adjacent-layer similarity exceeded non-adjacent similarity in 12/12 dataset-metric comparisons, with 10/12 reaching `p < 0.05`.

| dataset | metric | adjacent_mean | non_adjacent_mean | difference | p_value |
| --- | --- | --- | --- | --- | --- |
| cifar10 | cka | 0.954 | 0.8335 | 0.1206 | 0.0002 |
| cifar10 | cosine | 0.3334 | 0.2885 | 0.0449 | 0.004799 |
| cifar10 | pwcca | 0.9361 | 0.9004 | 0.0357 | 0.0002 |
| cifar10 | svcca | 0.7674 | 0.7305 | 0.0369 | 0.032194 |
| fashion_mnist | cka | 0.9703 | 0.9028 | 0.0674 | 0.0002 |
| fashion_mnist | cosine | 0.4166 | 0.4063 | 0.0103 | 0.227954 |
| fashion_mnist | pwcca | 0.9591 | 0.9254 | 0.0338 | 0.0002 |
| fashion_mnist | svcca | 0.7499 | 0.6841 | 0.0658 | 0.0002 |
| mnist | cka | 0.9793 | 0.9253 | 0.054 | 0.0002 |
| mnist | cosine | 0.478 | 0.4636 | 0.0144 | 0.063387 |
| mnist | pwcca | 0.94 | 0.9161 | 0.0239 | 0.0002 |
| mnist | svcca | 0.7271 | 0.6591 | 0.068 | 0.0002 |

### Cross-seed same-depth vs off-depth similarity
Same-depth cross-seed similarity exceeded off-depth similarity in 8/12 comparisons, with 5/12 reaching `p < 0.05`.

| dataset | metric | same_depth_mean | off_depth_mean | difference | p_value |
| --- | --- | --- | --- | --- | --- |
| cifar10 | cka | 0.9359 | 0.8576 | 0.0783 | 0.0008 |
| cifar10 | cosine | 0.3211 | 0.302 | 0.0191 | 0.4993 |
| cifar10 | pwcca | 0.8878 | 0.8887 | -0.001 | 0.623675 |
| cifar10 | svcca | 0.6715 | 0.7018 | -0.0303 | 0.119376 |
| fashion_mnist | cka | 0.916 | 0.8777 | 0.0383 | 0.037792 |
| fashion_mnist | cosine | 0.4051 | 0.4137 | -0.0086 | 0.426515 |
| fashion_mnist | pwcca | 0.9411 | 0.9254 | 0.0158 | 0.139972 |
| fashion_mnist | svcca | 0.6954 | 0.6761 | 0.0193 | 0.0018 |
| mnist | cka | 0.956 | 0.9234 | 0.0326 | 0.0014 |
| mnist | cosine | 0.4562 | 0.4696 | -0.0134 | 0.457908 |
| mnist | pwcca | 0.9166 | 0.9078 | 0.0089 | 0.358328 |
| mnist | svcca | 0.6818 | 0.6568 | 0.025 | 0.0014 |

### Linear probe accuracy by layer
- `cifar10`: best probe at layer 3 with mean accuracy 0.4292, versus layer 1 at 0.4100
- `fashion_mnist`: best probe at layer 4 with mean accuracy 0.8450, versus layer 1 at 0.8308
- `mnist`: best probe at layer 4 with mean accuracy 0.9108, versus layer 1 at 0.8767

| dataset | layer | mean | std |
| --- | --- | --- | --- |
| cifar10 | 1 | 0.41 | 0.0198 |
| cifar10 | 2 | 0.4183 | 0.0123 |
| cifar10 | 3 | 0.4292 | 0.0038 |
| cifar10 | 4 | 0.4217 | 0.008 |
| fashion_mnist | 1 | 0.8308 | 0.0038 |
| fashion_mnist | 2 | 0.8308 | 0.0126 |
| fashion_mnist | 3 | 0.845 | 0.0075 |
| fashion_mnist | 4 | 0.845 | 0.0043 |
| mnist | 1 | 0.8767 | 0.0166 |
| mnist | 2 | 0.905 | 0.0025 |
| mnist | 3 | 0.9083 | 0.0058 |
| mnist | 4 | 0.9108 | 0.0014 |

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
