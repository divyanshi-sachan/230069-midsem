# Advanced Machine Learning: Mid-Semester (230069-midsem)

**Repository:** 230069-midsem  
**Paper:** *Variable Selection in Model-Based Clustering: To Do or To Facilitate* (Poon et al., ICML 2010)

---

## Overview

This repository contains the full submission for the AML Mid-Semester examination: **Part A** (paper selection and understanding) and **Part B** (reproduction, ablations, failure mode, and report). The selected paper introduces the **Pouch Latent Tree Model (PLTM)**, which generalises Gaussian Mixture Models into a tree of discrete latent variables and “pouch” nodes of continuous observed variables, enabling **multiple facets** of clustering (multiple meaningful partitions) instead of a single variable subset.

---

## Repository Structure

```
230069-midsem/
├── README.md                 # This file
├── partA/                    # Part A deliverables
│   ├── paper_understanding.md   # Teaching notes on PLTM, assumptions, EM, eigenvalue constraint
│   └── llm_usage_partA.json     # LLM disclosure for paper selection and research
└── partB/                    # Part B deliverables
    ├── task 1 1.ipynb .. task 3 2.ipynb   # Question 1–3 notebooks
    ├──  report.pdf              # Question 4 report (max 4 pages)
    ├── llm task 1 1.json .. llm task 4 2.json  # 10 LLM disclosure files
    ├── data/                  # Synthetic datasets (+ data/README.md)
    ├── results/               # Figures (feature curve, ablations, failure, etc.)
    └── requirements.txt      # Python dependencies
```

---

## Part A (`partA/`)

| Item | Description |
|------|-------------|
| **paper_understanding.md** | Notes on the paper: problem (variable selection in clustering), PLTM architecture (tree, pouches, conditional Gaussians), key assumptions (singular parentage, tree, Gaussian), EM with eigenvalue constraint (γ=20), baselines (CVS, LFJ), and reproducibility. |
| **llm_usage_partA.json** | Disclosure of LLM use for paper search and method understanding (Gemini, ChatGPT, Claude). |

---

## Part B (`partB/`)

### Question 1 — Understanding (Tasks 1.1–1.3)

| Notebook | Content |
|----------|---------|
| **task 1 1.ipynb** | Step-by-step description of the PLTM: tree, latents, pouches, conditional Gaussians, and eigenvalue constraint formula (σ²_min ≤ λ ≤ γ·σ²_max). |
| **task 1 2.ipynb** | Three key assumptions (singular parentage, tree structure, conditional Gaussian) with brief justification, violation examples, and reference to the paper. |
| **task 1 3.ipynb** | Baselines (CVS, LFJ), limitation of “do” variable selection, how PLTM “facilitates” multiple facets, when it does not (e.g. Iris), and a conceptual explanation in several sections. |

### Question 2 — Reproduction (Tasks 2.1–2.3)

| Notebook | Content |
|----------|---------|
| **task 2 1.ipynb** | Justification and generation of the **synthetic multifaceted dataset**: 1000×4, Facet 1 (3-component GMM on features 0–1), Facet 2 (2-component GMM on features 2–3). Saves `toy_multifacet_X.npy` and label files to `data/`. |
| **task 2 2.ipynb** | **EM with eigenvalue constraints** on a **fixed two-pouch PLTM** (K1=3, K2=2, γ=20): load data, init, E-step, M-step, eigenvalue constraint, EM loop, hard assignments Z₁/Z₂, NMI evaluation. Saves Z₁, Z₂, NMI, and log-likelihood history to `results/`. |
| **task 2 3.ipynb** | Reported result vs paper (our NMI ≈ 0.91/0.96 vs paper 0.81±0.04), honest gap explanation, **multiple visualisations** (feature curve, scatter recovered/ground-truth, EM convergence, NMI comparison) saved to `results/`, and reproducibility checklist. |

**Metric:** Normalised Mutual Information (NMI) between recovered Z₁, Z₂ and ground-truth facet labels (same as paper Section 5.2).

### Question 3 — Ablation and Failure Mode (Tasks 3.1–3.2)

| Notebook | Content |
|----------|---------|
| **task 3 1.ipynb** | **Two-component ablation:** (1) Remove eigenvalue constraint — comparison of full vs ablated NMI; (2) Fix mixing proportions π (uniform) — same comparison. Plots saved to `results/` (ablation1_eigenvalue_constraint.png, ablation2_mixing_proportions.png). Interpretation for each ablation (5–7 sentences). |
| **task 3 2.ipynb** | **Failure mode:** Entangled facets (linear mixing so every feature depends on both latents), violating singular parentage and tree assumptions. Run PLTM on original vs entangled data; NMI for Facet 1 drops (~0.016) while Facet 2 can remain high. Plot saved to `results/failure_entangled_facets.png`. Explanation tied to Task 1.2 assumptions and one-sentence suggested fix. |

### Question 4 — Report and LLM Disclosures

| Item | Description |
|------|-------------|
| **report.pdf** | Compiled report (generate via `pdflatex report.tex` or Overleaf). **For Overleaf:** upload `report.tex` and a folder `results/` containing the PNGs from the notebooks (feature_curve.png, ablation1_eigenvalue_constraint.png, ablation2_mixing_proportions.png, failure_entangled_facets.png). |
| **llm task 1 1.json** … **llm task 4 2.json** | Ten LLM usage disclosure files (1.5 marks each): task tag, interaction log (prompt, tool, code used verbatim, student modification), top prompts, and student verification. Tools used: ChatGPT, Claude, Gemini for reading comprehension and light writing/implementation assistance; ablation choices, failure mode, and reflection are declared as the student’s own. |

### Data and Results

| Path | Description |
|------|-------------|
| **data/** | Synthetic dataset from Task 2.1. See **data/README.md** for generation procedure and file descriptions (`toy_multifacet_X.npy`, `toy_multifacet_labels_facet1.npy`, `toy_multifacet_labels_facet2.npy`). |
| **results/** | All figures produced by the notebooks: feature_curve.png, scatter_recovered.png, scatter_groundtruth.png, em_loglik.png, nmi_comparison.png, ablation1_eigenvalue_constraint.png, ablation2_mixing_proportions.png, failure_entangled_facets.png. Run the notebooks in order to generate them. |
| **requirements.txt** | Python dependencies (numpy, scikit-learn, matplotlib, scipy, jupyter, notebook). CPU-only; no GPU required. |

---

## How to Run

1. **Environment:** From the repo root,  
   `cd partB` then `pip install -r requirements.txt`.
2. **Data:** Run **task 2 1.ipynb** first to generate and save the synthetic dataset in `data/`.
3. **Reproduction:** Run **task 2 2.ipynb** to fit the PLTM and save Z₁, Z₂, NMI, and log-likelihood history to `results/`.
4. **Result and figures:** Run **task 2 3.ipynb** to produce the feature curve and other visualisations in `results/`.
5. **Ablation and failure:** Run **task 3 1.ipynb** and **task 3 2.ipynb** to produce ablation and failure-mode figures in `results/`.
6. **Report:** Compile `report.tex` (e.g. with `pdflatex report.tex` or by uploading to Overleaf with the `results/` folder).

**Note:** Do not clear notebook outputs before submission. All notebooks are intended to be run from top to bottom in order.

---

## Reproduction Summary

- **Contribution reproduced:** EM algorithm with eigenvalue constraints (Section 3) on a fixed two-pouch PLTM.
- **Setup:** K1=3, K2=2, γ=20, synthetic 1000×4 data with two facets (3 clusters on features 0–1, 2 clusters on features 2–3).
- **Result:** NMI(Z₁, Facet 1) ≈ 0.91, NMI(Z₂, Facet 2) ≈ 0.96. Paper reports ~0.81±0.04; gap explained by fixed structure, fewer restarts, and simpler 4-attribute setup.
- **Ablations:** Removing the eigenvalue constraint or fixing π gave slight NMI changes on this toy data; both components support stability and balance-adaptation.
- **Failure mode:** Entangled facets (every feature depends on both latents) violate singular parentage/tree; NMI for Facet 1 collapses while Facet 2 can still be recovered depending on the mixing.

---

## Constraints

- **Hardware:** All code is CPU-only; no GPU or cloud compute required.
- **Submission:** Part B is a self-contained reproduction and analysis of the paper selected in Part A.
- **Execution:** Results in the notebooks were produced by running cells in order in a clean environment.
