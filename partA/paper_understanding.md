# Teaching Notes: Variable Selection in Model-Based Clustering

**Paper:** Variable Selection in Model-Based Clustering: To Do or To Facilitate  
**Authors:** Poon, Zhang, Chen, Wang (ICML 2010)

This document explains the paper in a descriptive way and my understanding of the pape5r.

---

## 1. The problem: why variable selection for clustering is hard

In **supervised learning** (e.g. classification), variable selection has a clear goal: find the subset of features that gives the best prediction of the label. We can measure “best” with accuracy or cross-validation. In **clustering**, there are no labels. We don’t know the “true” partition, so we can’t directly say which subset of attributes is “best.” That’s the first difficulty.

The paper adds a second, deeper point: **high-dimensional data are often multifaceted**. By this they mean the same set of objects can be meaningfully grouped in **several different ways**, depending on which attributes we emphasize. For example, people might be clustered by (a) age group, (b) income level, or (c) geographic region—each is a different “facet” of the data. If we insist on finding **one** subset of attributes that gives the “best” clustering, we are implicitly assuming there is only one meaningful way to cluster—which is often false.

So the paper’s stance is: instead of **performing** variable selection (choosing one subset and one clustering), we should **facilitate** it. That is: **discover multiple facets**, cluster the data along each facet, and **present these options to a domain expert** who can then choose what is interesting. The technical contribution is a model that can represent and learn these multiple facets: the **Pouch Latent Tree Model (PLTM)**. PLTM generalizes the Gaussian Mixture Model (GMM) so that we get **multiple latent variables**, each corresponding to one way of partitioning the data (one facet).

---

## 2. PLTM architecture: structure and parameters

Think of a **PLTM** as a **rooted tree** with two types of nodes.

### 2.1 Two types of nodes

- **Internal nodes** are **latent (hidden) variables**. They are **discrete** (e.g. each has a finite number of states, like 2 or 3). They are never observed; we only see the data and infer them. Each latent variable represents one possible **partition** (clustering) of the data.
- **Leaf nodes** are **pouch nodes**. Each pouch is a **set of continuous observed (manifest) variables**. So a leaf is not necessarily a single feature—it can be a “pouch” containing several features that are grouped together under one parent latent. That’s why the model is called “pouch” latent tree.

**Example:** We might have latents Y₁, Y₂, Y₃ and manifest variables X₁,…, X₉. The tree could look like: root Y₁ has two children—another latent Y₂ and a pouch {X₁, X₂, X₃}. Y₂ might have children that are pouches {X₄, X₅} and {X₆, X₇, X₈, X₉}. So we have a hierarchy: latents at the top, and at the leaves we have groups of observed variables.

### 2.2 Parameters: how the model generates data

- **For each latent node Y with parent Y′:** we have a **conditional probability table** P(Y | Y′). So the distribution of Y depends on the value of its parent. (The root is treated as having a dummy parent with one value, so we just have P(Y_root).)
- **For each pouch of manifest variables X with parent latent Y:** we assume that **given a value y of Y**, the vector X is **Gaussian**: P(x | y) = N(x | μ_y, Σ_y). So each value of the parent latent corresponds to one mean vector and one covariance matrix for that pouch. Different values of Y give different Gaussians; that’s how the latent “clusters” the observations in that pouch.

So in words: first we sample the latent variables from the root down the tree (using the conditional tables). Then, at each pouch, we sample the continuous variables from a Gaussian whose parameters depend on the value of the parent latent. This gives a clear generative story.

### 2.3 How PLTM relates to GMM

A **GMM** has one latent variable (the mixture component) and **one** “pouch” that contains **all** observed variables. So GMM = one latent + one Gaussian per component over the full feature vector. PLTM generalizes this by allowing **many** latent variables arranged in a **tree** and **many** pouches (each with its own set of manifest variables and its own conditional Gaussians). So GMM is the special case of PLTM with a single latent and a single pouch containing everything.

---

## 3. Key assumptions (and why they matter)

When you teach or implement the model, it helps to state the assumptions clearly.

1. **Tree structure / singular parentage**  
   The graph is a **tree**: each node has at most one parent, and there are no cycles. So each variable (latent or manifest) is “owned” by exactly one parent. This keeps inference and learning tractable and avoids ambiguous dependencies.

2. **Discrete latents, Gaussian pouches**  
   Latent variables are **discrete** (finite state space). Given the parent latent, the manifest variables in a pouch follow a **conditional Gaussian** distribution. So we are not modeling heavy tails or discrete observations in the pouches; the Gaussian assumption is central to the EM updates and the eigenvalue constraint below.

3. **Each manifest variable belongs to exactly one pouch**  
   So each observed attribute is attached to exactly one latent (via its pouch). The model does not allow one attribute to directly depend on two different latents. This keeps the “facet” interpretation clean: each facet (latent) is associated with a subset of attributes (its descendant pouches).

These assumptions are what make the model well-defined and learnable; when they are violated (e.g. non-Gaussian data, or a true structure that is not a tree), we can expect the method to underperform—which is useful for the “failure mode” task later.

---

## 4. Parameter estimation: EM and the eigenvalue constraint

Suppose the **structure** of the PLTM (the tree and the pouches) is **fixed**. We want to estimate the parameters: the conditional tables P(Y | Y′) and the means and covariances μ_y, Σ_y for each pouch.

### 4.1 The EM algorithm

We use the **Expectation–Maximization (EM)** algorithm, as in GMMs.

- **E-step:** Given current parameters, we compute for each data point and each latent (and parent) the **posterior** probabilities P(y, y′ | d_i) and P(y | d_i). So we’re inferring the distribution over the hidden variables given the observed data. The paper does this with exact inference in the mixed (discrete + Gaussian) Bayesian network (Lauritzen & Jensen style).

- **M-step:** We update the parameters using these posteriors. For P(y | y′), we use the expected counts. For each pouch, the new μ_y is the expected value of the observations in that pouch given y, and the new Σ_y is the expected covariance given y. So the M-step looks like the usual M-step for a GMM, but applied per pouch and per latent value, with expectations taken over the posterior from the E-step.

EM is iterated until the log-likelihood improvement is below a threshold (e.g. 0.01) or we hit a maximum number of iterations (e.g. 500).

### 4.2 Why the likelihood can be unbounded (and why we need a constraint)

In a GMM, if we allow covariance matrices to be arbitrary, the likelihood can go to infinity (e.g. one component shrinks to a single point). The same can happen in a PLTM: a Gaussian in a pouch can collapse, giving a spike and an unbounded likelihood. So we can get **spurious local maxima** that don’t correspond to sensible clusterings.

### 4.3 The eigenvalue constraint (γ = 20)

To avoid this, the paper **constrains the eigenvalues** of each pouch covariance matrix Σ_y. Let σ²_min and σ²_max be the **minimum and maximum of the sample variances** of the variables in that pouch (i.e. the diagonal of the sample covariance of that pouch). Then for **every eigenvalue λ_i** of Σ_y they require:

- **σ²_min ≤ λ_i ≤ γ · σ²_max**

Here **γ** is a constant; the paper sets **γ = 20**. So no eigenvalue can be smaller than the smallest sample variance (avoiding collapse) and no eigenvalue can be larger than 20 times the largest sample variance (avoiding huge, degenerate directions). In the M-step, after computing the usual Σ_y, they presumably project or clip its eigenvalues to lie in this range (Ingrassia-style). This keeps the optimization in a “reasonable” region and helps recovery of multifaceted structure.

For reproduction, you need to implement this constraint in the M-step when updating the covariance matrices of the pouches.

---

## 5. Structure learning: how do we get the tree and pouches?

In Section 4 we assumed the **structure** (which latents exist, how they’re connected, and how manifest variables are grouped into pouches) was fixed. In practice we don’t know it. The paper learns structure by **maximizing the BIC score**:

- **BIC(m | D) = log P(D | m, θ*) − (d(m)/2) log N**

Here m is the structure, θ* is the MLE of the parameters for that structure, d(m) is the number of free parameters, and N is the sample size. The first term rewards fit to the data; the second penalizes complexity. So BIC balances fit and model size.

They use a **hill-climbing** search:

- **Start:** One latent node with two states, and **one pouch per manifest variable** (each attribute in its own pouch).
- **Iteration:** Apply **search operators** to the current structure to get candidates. Operators include: introduce/delete a latent node, add/delete a state of a latent, relocate a node (change edges), merge two pouches (“pouching”), split a pouch (“unpouching”). For each candidate, run EM (with the eigenvalue constraint) to get θ* and compute BIC. Accept the candidate if its BIC is higher than the current model; otherwise stop and return the current model.

So the algorithm grows (or shrinks) the tree and the pouches step by step, always improving BIC. The paper mentions acceleration tricks (e.g. approximate BIC, not running full EM for every candidate) but doesn’t give full details; for a reproduction you might implement a simpler version (e.g. fixed small structure) and focus on the EM with eigenvalue constraint.

---

## 6. Baselines and the paper’s core improvement

### 6.1 What the baselines do

- **Plain GMM:** No variable selection; all attributes are used. One latent (mixture component), one partition.
- **MCLUST:** Still a GMM but with constrained covariances (eigenvalue-type constraints). No explicit variable selection.
- **CVS (ClustVarsel, Raftery & Dean):** **Performs** variable selection. It searches for a **single** subset of attributes and fits a GMM on that subset (e.g. greedy search, Bayes factor comparison). So the output is one subset and one clustering.
- **LFJ (Law, Figueiredo, Jain):** **Performs** variable selection by giving each attribute a “saliency” (between 0 and 1). Saliency is learned with EM; attributes with low saliency are effectively down-weighted. Again, the goal is to find **one** clustering that depends on a subset of attributes.

So CVS and LFJ both **do** variable selection: they try to find the “best” subset and the “best” single clustering.

### 6.2 What PLTM does differently (facilitate, not do)

PLTM does **not** choose one subset. It learns a **tree of latents** and **multiple** partitions (one per latent, or combinations). Each partition may depend mainly on a **subset** of attributes (those in its descendant pouches). So the output is **several** clusterings, each corresponding to a different facet. The **user** then inspects these and chooses what is interesting. So PLTM **facilitates** variable selection by exposing multiple facets rather than performing a single selection.

### 6.3 Why PLTM often wins in the experiments

On the paper’s experiments (synthetic and UCI data), they evaluate by **NMI** between the **true class label** and the clustering. For PLTM they take the **maximum** NMI over the latent variables (or a greedy combination) to see if **any** of the facets matches the class. They find that PLTM often outperforms CVS and LFJ (e.g. on synthetic: PLTM ≈ 0.81, CVS ≈ 0.07, LFJ ≈ 0.54). The intuition: when data are multifaceted, CVS/LFJ commit to **one** subset and one partition; if they pick the “wrong” facet (e.g. the one not aligned with the class), NMI is poor. PLTM instead recovers **both** facets and lets the user pick the one that matches the class, so max-NMI is high. So the “facilitate” approach is often more appropriate than “do” when there are multiple meaningful ways to cluster.

---

## 7. Synthetic data (for reproduction)

The paper uses a **synthetic** data set generated from a known PLTM so they can check whether the method recovers the true facets.

- **Structure:** 9 manifest variables X₁,…, X₉ and **2** latent variables Y₁, Y₂, each with **3** states. The generative structure is like Figure 1(c) in the paper (tree with root Y₁, and Y₂ and pouches below).
- **Parameters:** Y₁ is uniform. P(Y₂ | Y₁): with probability 0.5, Y₂ equals Y₁; with probability 0.25 each, Y₂ takes each of the other two values. So Y₁ and Y₂ are dependent but not identical.
- **Gaussian conditionals:** For each pouch, given the parent latent value, the mean vector has components in {0, 2.5, 5}; variance of each variable is 1; covariance between two variables in the **same** pouch is 0.5. So within-pouch correlation is 0.5; across pouches it’s determined by the tree.
- **Sampling:** They draw N = 1000 samples from this model. The **class label** used for evaluation is Y₁. Y₂ is **not** given to the algorithm; it’s hidden. So the data are multifaceted: one facet is Y₁ (class), another is Y₂.
- **Facets in the data:** In the true model, Y₁ is strongly associated with X₁–X₃ (one facet), and Y₂ with X₄–X₉ (another facet). So when you implement, you want to generate data that respect this so that a good algorithm can recover two facets and one of them (Y₁) should match the class and give high NMI.

---

## 8. Evaluation protocol (for reproduction)

- **Metric:** **Normalized Mutual Information (NMI)** between the **class variable C** and the clustering (latent) **Y**: NMI(C, Y) = I(C; Y) / √(H(C) H(Y)). Higher means the clustering agrees more with the class.
- **Protocol:** **5-fold stratified cross-validation**. In each fold, the model is learned on the training set **without** class labels (unsupervised). Then on the test set, we compute NMI between the class and the clustering implied by the model.
- **PLTM:** A learned PLTM has **several** latents. They evaluate **max over latents** (or a greedy combination) of NMI with the class—i.e. “if the user picked the best facet, how well would they do?” So the reported PLTM score is max_W ⊆ latents NMI(C, W) (approximately, via greedy search).
- **Settings:** γ = 20 for the eigenvalue constraint; **64 random restarts** for EM; stop when log-likelihood increase < 0.01 or at 500 iterations. These are the numbers to use for a faithful reproduction.

---

## 9. Failure modes and limitations (for critical reflection)

The paper itself and the experiments suggest when and why the method can fail.

- **Iris:** PLTM (0.76) is **worse** than CVS (0.87). Iris has only **4** attributes. There may be essentially **one** dominant facet. In that case, “facilitate” doesn’t add much—variable selection (CVS) finds that one subset and does well. PLTM might still find multiple latents, but with so few dimensions the structure is oversimplified or the extra facets are noise. So **low dimensionality** can be a failure mode: when there’s only one meaningful facet, methods that **do** variable selection can win.

- **Ionosphere, wdbc:** Sometimes **GMM** or **MCLUST** (no variable selection) beat PLTM. Possible reasons: the **class** might not align with any clear facet (e.g. class is an artifact of the labeling); or the **true** structure might not be tree-like or Gaussian (e.g. non-Gaussian clusters, or dependencies that don’t fit a tree). So **violations of the model’s assumptions** (tree, Gaussian, one-facet-per-latent) can be a failure mode.

- **Computational cost:** Training PLTM (structure search + EM with multiple restarts) can be **slow** (hours to days per fold on larger data). So scalability is a practical limitation.

When you write the “failure mode” task (e.g. in task 3.2), you can design an experiment that highlights one of these (e.g. very few features, or non-Gaussian data) and explain in terms of the assumptions above.

---

## Summary for quick revision

- **Problem:** Clustering + variable selection is ill-defined and multifaceted data have several valid clusterings.
- **Idea:** **Facilitate** selection by finding **multiple facets** (PLTM) instead of **doing** selection (CVS, LFJ).
- **Model:** PLTM = tree of **discrete** latents + **pouch** leaves of **Gaussian** manifest variables; GMM = one latent + one pouch.
- **Learning:** EM for parameters; **eigenvalue constraint** (σ²_min ≤ λ ≤ γ σ²_max, γ = 20) to avoid degeneracy; BIC + hill-climbing for structure.
- **Evaluation:** NMI vs class; for PLTM use max over latents; 5-fold CV, 64 restarts, γ = 20.
- **Failure cases:** Low dimension (one facet), wrong assumptions (non-tree, non-Gaussian), or class not aligned with any facet.
