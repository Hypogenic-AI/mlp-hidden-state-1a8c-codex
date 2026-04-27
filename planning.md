# Research Plan: How Similar Are MLP Hidden States to Each Other?

## Motivation & Novelty Assessment

### Why This Research Matters
Layer-wise hidden-state similarity matters because many interpretability, pruning, and transfer claims assume that intermediate representations evolve gradually rather than being completely rewritten at every layer. If plain MLPs already preserve substantial structure across depth, that supports the view that representational change is constrained and measurable, which is useful for theory, diagnostics, and lightweight model editing.

### Gap in Existing Work
The gathered literature shows strong methodology for representation comparison, but most recent empirical work emphasizes CNNs and transformers rather than controlled plain-MLP studies. There is also a repeated warning that no single similarity metric is sufficient, yet many studies still rely on one geometric score without checking whether task-relevant information is preserved.

### Our Novel Contribution
This project runs a controlled, multi-metric study on plain feedforward MLPs across three standard image benchmarks. The main contribution is a combined analysis of within-model layer similarity, cross-seed same-depth similarity, and per-layer linear decodability, allowing us to ask not only whether layers are geometrically similar, but also what representational properties change with depth.

### Experiment Justification
- Experiment 1: Train matched MLPs on MNIST, Fashion-MNIST, and CIFAR-10 to establish whether adjacent layers are consistently more similar than distant layers across tasks of increasing difficulty.
- Experiment 2: Compare hidden states with linear CKA, SVCCA, PWCCA, and sample-wise cosine similarity to test whether conclusions are metric-stable rather than artifacts of one similarity definition.
- Experiment 3: Measure cross-seed same-depth similarity to test whether layer roles recur across random initializations.
- Experiment 4: Train linear probes on each hidden layer to determine whether deeper layers differ mainly by improving decodable task information rather than by becoming wholly unrelated spaces.

## Research Question
Are hidden-state spaces in different layers of trained MLPs substantially shared rather than totally different, and if so, what measurable properties change across depth?

## Background and Motivation
Prior work suggests that nearby layers in neural networks often remain correlated, while deeper layers become more task-specialized. For plain MLPs, however, there is still room for a clean controlled benchmark that separates geometric similarity from task decodability and checks whether conclusions hold across multiple metrics and datasets.

## Hypothesis Decomposition
- H1: Adjacent hidden layers within a trained MLP are more similar than layers separated by larger depth gaps.
- H2: The monotonic decline with layer distance is visible across multiple similarity metrics, not only one.
- H3: Same-depth hidden layers from different random seeds are more similar than mismatched depths, indicating recurring functional roles.
- H4: Deeper layers improve linear probe accuracy even when geometric similarity to earlier layers remains non-trivial.
- Alternative explanation A: Similarity trends are artifacts of activation dimensionality or normalization rather than representational continuity.
- Alternative explanation B: A single metric may overstate similarity; this is why probes and multiple metrics are included.

## Proposed Methodology

### Approach
Train the same MLP architecture on three image datasets converted to flat vectors, extract hidden activations on held-out data, and compare every pair of hidden layers using several representation similarity measures. Complement geometric comparisons with per-layer linear probes and cross-seed comparisons so the analysis can distinguish “same space,” “compatible information,” and “same functional role.”

### Experimental Steps
1. Load local datasets from `datasets/`, validate schema differences, and normalize flattened inputs.
2. Train a fixed-depth ReLU MLP with dropout and early stopping for multiple seeds on each dataset.
3. Save hidden activations for a fixed held-out subset and compute pairwise within-model similarity matrices.
4. Aggregate similarity by layer distance and compare adjacent versus non-adjacent layers statistically.
5. Compute cross-seed same-depth and mismatched-depth similarities.
6. Train linear probes on each hidden layer and compare depth against probe accuracy.
7. Summarize results with tables, heatmaps, and statistical tests in `REPORT.md`.

### Baselines
- Sample-wise cosine similarity between paired activations.
- SVCCA.
- PWCCA.
- Linear CKA.
- Final-layer classifier performance as the task-performance reference.
- Earlier-layer linear probes as task-information baselines.

### Evaluation Metrics
- Test accuracy of each trained MLP.
- Pairwise hidden-layer similarity matrices for each metric.
- Mean similarity as a function of layer distance.
- Cross-seed same-depth versus off-depth similarity gap.
- Linear probe accuracy by depth.
- Correlation between probe accuracy and similarity to the final hidden layer.

### Statistical Analysis Plan
- Null hypothesis H0a: Adjacent-layer similarity is not higher than non-adjacent-layer similarity.
- Null hypothesis H0b: Same-depth cross-seed similarity is not higher than mismatched-depth similarity.
- Use paired permutation or paired t-tests across trained models where sample counts permit.
- Report mean differences, 95% confidence intervals from bootstrap resampling, and Cohen's d where appropriate.
- Significance level: 0.05, with Benjamini-Hochberg correction across metric families if multiple related tests are reported together.

## Expected Outcomes
Support for the hypothesis would look like consistently high adjacent-layer similarity, a smooth decline with layer distance, and a rise in linear probe performance with depth rather than a collapse into unrelated spaces. Refutation would require weak or unstable adjacent-layer similarity and no consistent cross-seed layer alignment.

## Timeline and Milestones
1. Planning and environment validation: completed first.
2. Dataset loading and script scaffolding: immediate next step.
3. Training and activation extraction: primary runtime phase.
4. Similarity/probe analysis and visualization: after successful model runs.
5. Documentation and validation: final phase.

## Potential Challenges
- CIFAR-10 may be relatively weak for plain MLPs; if needed, keep the same architecture but allow slightly longer training while documenting the limitation.
- SVCCA/PWCCA can be numerically brittle; activations will be centered, downsampled consistently, and wrapped with error handling.
- Metric values may disagree; this is treated as a result, not an implementation failure.

## Success Criteria
- Reproducible training and evaluation scripts run end-to-end in the local `.venv`.
- Results are produced for at least MNIST and Fashion-MNIST, with CIFAR-10 included unless a documented resource issue blocks it.
- The final report contains actual similarity matrices, aggregated statistics, probe results, and a clear answer to the research question.
