# Downloaded Papers

This directory contains papers most directly useful for studying whether hidden states in multilayer perceptrons remain similar across depth and how that similarity should be measured.

1. [SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability](1706.05806_svcca.pdf)
   - Authors: Maithra Raghu, Justin Gilmer, Jason Yosinski, Jascha Sohl-Dickstein
   - Year: 2017
   - arXiv: `1706.05806`
   - Why relevant: Introduces SVCCA, a strong baseline for comparing hidden representations across layers and across independently trained models.

2. [Similarity of Neural Network Representations Revisited](1905.00414_cka_revisited.pdf)
   - Authors: Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton
   - Year: 2019
   - arXiv: `1905.00414`
   - Why relevant: Establishes CKA as a more reliable layer-comparison metric than CCA-style methods in common high-dimensional regimes.

3. [Similarity and Matching of Neural Network Representations](2110.14633_similarity_matching.pdf)
   - Authors: Adrián Csiszárik, Péter Kőrösi-Szabó, Ákos K. Matszangosz, Gergely Papp, Dániel Varga
   - Year: 2021
   - arXiv: `2110.14633`
   - Why relevant: Adds a functional notion of similarity via stitching, useful when geometric similarity alone may be misleading.

4. [Reliability of CKA as a Similarity Measure in Deep Learning](2210.16156_reliability_cka.pdf)
   - Authors: MohammadReza Davari, Stefan Horoi, Amine Natik, Guillaume Lajoie, Guy Wolf, Eugene Belilovsky
   - Year: 2022
   - arXiv: `2210.16156`
   - Why relevant: Important cautionary paper showing when CKA can be manipulated or overstate similarity.

5. [Similarity of Neural Network Models: A Survey of Functional and Representational Measures](2305.06329_similarity_survey.pdf)
   - Authors: Max Klabunde, Tobias Schumacher, Markus Strohmaier, Florian Lemmerich
   - Year: 2023
   - arXiv: `2305.06329`
   - Why relevant: Broad survey of representational and functional similarity measures, useful for experiment design and metric selection.

6. [Tracing Representation Progression: Analyzing and Enhancing Layer-Wise Similarity](2406.14479_tracing_representation_progression.pdf)
   - Authors: Jiachen Jiang, Jinxin Zhou, Zhihui Zhu
   - Year: 2024
   - arXiv: `2406.14479`
   - Why relevant: Recent empirical and theoretical evidence that adjacent layers tend to become progressively more similar.

7. [What Representational Similarity Measures Imply about Decodable Information](2411.08197_decodable_information.pdf)
   - Authors: Sarah E. Harvey, David Lipshutz, Alex H. Williams
   - Year: 2024
   - arXiv: `2411.08197`
   - Why relevant: Connects similarity geometry to linear decodability, which is useful when interpreting hidden-state similarity in task terms.

Chunked copies for reading are in `papers/pages/`.
