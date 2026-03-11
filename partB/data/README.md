# partB/data — Datasets

This folder contains the datasets used in the Part B notebooks.

## How the dataset is obtained

**Synthetic multifaceted toy dataset** (used in Task 2.1, 2.2, 2.3):

- **Source:** Generated in the notebook `task 2 1.ipynb` using NumPy (no external download).
- **Random seed:** 42 (set at the top of the generation cell in task 2 1.ipynb).
- **Procedure:** Two independent Gaussian mixtures are sampled:
  - **Facet 1:** 3-component mixture on 2 dimensions (features 0, 1); equal weights; component means and a shared covariance defined in the notebook.
  - **Facet 2:** 2-component mixture on 2 dimensions (features 2, 3); equal weights; component means and a shared covariance defined in the notebook.
- Samples: 1,000. Combined into one matrix **X** of shape (1000, 4) and two label vectors for the two facets.

## How it is used in the notebooks

- **task 2 1.ipynb:** Generates the data and saves it to this folder; documents dataset choice and preprocessing.
- **task 2 2.ipynb:** Loads `toy_multifacet_X.npy` and the label files to run the PLTM EM algorithm (fixed two-pouch structure) and evaluate NMI.
- **task 2 3.ipynb:** Uses the same data and the fitted model to produce feature curves (NMI per latent per feature) and interpretation.

## Files

- `toy_multifacet_X.npy` — Data matrix, shape (1000, 4). Features 0–1 = Facet 1 subspace; features 2–3 = Facet 2 subspace.
- `toy_multifacet_labels_facet1.npy` — Ground-truth cluster labels for Facet 1 (values 0, 1, 2).
- `toy_multifacet_labels_facet2.npy` — Ground-truth cluster labels for Facet 2 (values 0, 1).
