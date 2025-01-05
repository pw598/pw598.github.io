---
layout: post
title:  "Dimension Reduction II: Principal Component Analysis (PCA)"
date:   2025-01-03 00:00:00 +0000
categories: DimensionReduction Python
---

<p></p>

<img src="https://github.com/pw598/pw598.github.io/blob/main/_posts/images/dr2.png?raw=true" style="height: 500px; width:auto;">


# Outline 

- What is Principal Component Analysis (PCA)?
- Numpy Implementation
- Scikit-Learn Implementation
- Reconstruction of Data
    - The Iris Dataset
    - The Olivietti Faces Dataset
- Advantages and Limitations
- Scikit-Learn Implementation Details
    - Parameters
    - Atributes
    - Methods
- What’s Next?



# What is Principal Component Analysis (PCA)?

PCA is essentially the eigendecomposition of a covariance matrix. It creates a weighted combination of channels such that each resulting component has maximal variance under the constraints that the magnitude of the weights vector is equal to $1$, and that the basis vectors are all orthogonal. $PC_1$, i.e., principal component #1, finds the direction of maximal covariance in the data space, and $PC_2$ finds the direction of maximal covariance that is orthogonal to all previous principal components, etc. Orthogonality, can be defined from several perspectives:

- Geometrically, orthogonal vectors meet at a right angle.
- Algebraically, orthogonal vectors have a dot product of $0$.
- Statistically, orthogonal vectors have correlation of $0$.

The diagonal matrix $\mathbf{M}$ contains the eigenvalues, and the columns of matrix $W$ are the eigenvectors, also called the principal components.

The steps to compute a principal component analysis are:

<ol>
<li>Mean-center the data.</li>
<li>Compute the covariance matrix of the data.</li>
<li>Perform eigendecomposition on the covariance matrix.</li>
<li>Sort eigenvectors according to eigenvalue magnitude.</li>
<li>Compute the component scores, calculated as the weighted combination of all data features, where the eigenvector provides the weights. i.e., for data matrix X, component one is calculated as \mathbf{v}_i^T \mathbf{X}, where v_1 is the eigenvector with the largest associated eigenvalue.</li>
<li>(Optional) Convert to variance explained. The eigenvalues are in the scale of the variance of the data. To normalize, divide by the sum of eigenvalues.</li>
</ol>



# Numpy Implementation

We’ll start by importing our libraries.




























