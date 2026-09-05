# Project provenance

## Origin of the method

The central PSOD method used in this repository predates this project.

Thomas Meißner published a project named **PSOD (Pseudo-supervised outlier detection)** that treats each feature as a prediction target and uses supervised prediction errors in an unsupervised outlier-detection setting. The original project is available at:

https://github.com/ThomasMeissnerDS/PSOD

Its package metadata identifies Thomas Meißner as the author and declares the project under **GPL-3.0-only**.

## Relationship to this repository

The early implementation in this repository retained substantial structure and vocabulary from that project, including parameters such as:

- `min_cols_chosen`
- `max_cols_chosen`
- `stdevs_to_outlier`
- `sample_frac`
- `correlation_threshold`
- `transform_algorithm`
- `cat_encode_on_sample`
- `flag_outlier_on`

Later development added additional learners, preprocessing options, persistence, visualization, benchmarking, testing and scikit-learn-oriented interfaces. Those extensions do not make the original PSOD method a novel contribution of this repository.

## Attribution policy

During this refactor:

1. The repository will not claim invention of PSOD or the feature-as-target residual method.
2. Thomas Meißner's earlier implementation will remain explicitly credited.
3. Existing source derived from that implementation will be treated as GPL-3.0-only.
4. New algorithmic work will be documented separately so that original extensions are distinguishable from inherited design.
5. Benchmark or scientific claims will only be made when backed by reproducible committed evidence.

## Refactor direction

The planned methodological work is not to rename the existing implementation and call it new. The goal is to build a cleaner conditional-anomaly detector around ideas such as cross-fitted residuals, robust residual calibration, explicit feature-wise contributions and reproducible anomaly-geometry benchmarks.

Where those changes depart materially from the historical implementation, the distinction will be documented in code, tests and release notes.
