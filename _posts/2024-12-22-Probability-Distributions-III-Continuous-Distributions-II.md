---
layout: post
title:  "Probability III: Continuous Distributions II"
date:   2024-12-22 00:00:00 +0000
categories: Probability
---

The second of two articles listing continuous distributions, their properties, and their relationships. This second article focuses on a family of distributions closely related to the Gamma.


<img src="https://github.com/pw598/pw598.github.io/blob/main/_posts/images/pd3.png?raw=true" style="height: 350px; width:auto;">


# Outline 

- The Gamma Function
- The Generalized Gamma Distribution
- Gamma
- Inverse-Gamma
- Weibull
- Exponential
- Chi-Squared
- F-Distribution
	


# The Gamma Function

This article includes multiple distributions for which the PDF utilizes the Gamma function. The Gamma function $\Gamma(a)$ is an extension of the factorial function to real numbers, so just as factorials are often useful for normalization in basic probability problems, the Gamma function may play a role in normalizing a continuous probability distribution.

$\Gamma(a) = \int_0^{\infty} x^{\alpha} e^{-y} \frac{dy}{y}, ~\alpha \gt 0$

With integer-valued inputs, it reduces to 

$\Gamma(n) = (n-1)!$



# The Generalized Gamma Distribution

The Generalized Gamma distribution is a flexible family of distributions that extend the Gamma distribution (discussed next) to accommodate a wider range of shapes.

#### PDF:

$f(x; \alpha, \beta, \delta) = \frac{\delta}{\alpha \Gamma(1/\delta)} \left( \frac{x}{\alpha} \right)^{\delta-1} ~exp \left( - \left( \frac{x}{\alpha} \right)^{\delta} \right)$

- $\alpha \gt 0$ is the scale parameter

- $\beta \gt 0$ is the shape parameter

- $\delta \gt 0$ is the "distribution shape" parameter

Derived distributions include the Gamma (when $\delta = 1$), and its special cases (Weibull, Exponential), as well as the Log-Normal (when $\delta = 2$ and a logarithm is taken).



# Gamma

When $\delta=1$, the Generalized Gamma reduces to a Gamma distribution with shape parameter $\beta$ and scale parameter $\alpha$. The story is that it describes the amount of time we have to wait for α arrivals in a Poisson process, because a sum of Exponential random variables is Gamma-distributed. However, the Gamma is defined for non-integer arrival numbers, and supported on a continuous domain. It is the continuous analog of the Negative Binomial distribution.

#### PDF:

$f(y;\alpha, \beta) = \frac{1}{\Gamma(\alpha)}\,\frac{(\beta y)^\alpha}{y} \,\mathrm{e}^{-\beta y}$

The expected value is $\alpha / \beta$, and the variance is $\alpha / \beta^2$. 

<img src="https://www.pymc.io/projects/docs/en/latest/_images/pymc-Gamma-1.png" style="height: 300px; width:auto;">

<i>https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.Gamma.html</i>



# Inverse Gamma

The Inverse Gamma represents the reciprocal of a Gamma-distributed random variable, meaning that if $Y$ is Gamma distributed, then $1/Y$ is Inverse-Gamma distributed.

#### PDF:

$f(y;\alpha, \beta) = \frac{1}{\Gamma(\alpha)}\,\frac{\beta^\alpha}{y^{\alpha+1}} \,\mathrm{e}^{-\beta / y}$

The expected value is 

$\displaystyle{\frac{\beta}{\alpha - 1}}, ~\alpha \gt 1$

And the variance is 

$\displaystyle{\frac{\beta^2}{(\alpha-1)^2(\alpha-2)}}, ~\alpha \gt 2$

<img src="https://www.pymc.io/projects/docs/en/stable/_images/pymc-InverseGamma-1.png
" style="height: 300px; width:auto;">

<i>https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.InverseGamma.html</i>



# Weibull

The Weibull is derived from the Generalized Gamma distribution where $\delta = \beta$, and is commonly used in survival analysis. It is a generalization of the Exponential distribution, with more flexible shape.

#### PDF:

$f(y;\alpha, \sigma) = \frac{\alpha}{\sigma}\left(\frac{y}{\sigma}\right)^{\alpha - 1}\,\mathrm{e}^{-(y/\sigma)^\alpha}$

The expected value is 

$\displaystyle{\sigma \Gamma(1 + 1/\alpha)}$

and the variance is 

$\displaystyle{\sigma^2\left[\Gamma(1+2/\alpha) - \left(\Gamma(1 + 1/\alpha)\right)^2\right]}$

<img src="https://www.pymc.io/projects/docs/en/stable/_images/pymc-Weibull-1.png
" style="height: 300px; width:auto;">

<i>https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Weibull.html</i>



# Exponential

The Exponential distribution is a special case of the Weibull (and therefore the Gamma). It arises from the Generalized Gamma where $\delta=1$ and $\beta=1$, from the Gamma when $\alpha=1$, and from the Weibull when $\alpha=1$.

The story is that the Exponential random variable describes the time between rare events that occur with a rate of $\beta$ per unit of time (following a Poisson process). The inter-arrival time of a Poisson process is Exponentially distributed. The single parameter is the positive arrival rate, $\lambda$.

#### PDF:

$f(y; \lambda) = \lambda e ^{- \lambda y}$

The expected value is $1/\lambda$ and the variance is $1 / \lambda^2$.

<img src="https://www.pymc.io/projects/docs/en/stable/_images/pymc-Exponential-1.png
" style="height: 300px; width:auto;">

<i>https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Exponential.html</i>



# Chi-Squared

The Chi-Squared distribution is a Gamma distribution with fixed shape parameter $\alpha = k^2$ and scale parameter $\beta = 2$, where $k$ is the degrees of freedom. The Chi-Squared distribution with $k$ degrees of freedom is the distribution of the sum of squares of $k$ independent standard Normal variables. Each squared standard normal random variable $Z_i^2$ follows a Chi-Squared distribution with $1$ degree of freedom, and when you sum such distributions, the resulting distribution is $\chi^2(k)$.

#### PDF:

$f(y;k) = \frac{ y^{k/2-1} e^{-x/2} }{ 2^{k/2} \Gamma \left( \frac{k}{2} \right) }$

The expected value is $k$ and the variance is $2k$.

<img src="https://www.pymc.io/projects/docs/en/stable/_images/pymc-ChiSquared-1.png
" style="height: 300px; width:auto;">

<i>https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.ChiSquared.html</i>



# F-Distribution

The F-distribution arises when comparing the variances of two independent Chi-Squared distributions.

#### PDF:

For independent Chi-Squared random variables $X_1$ and $X_2$ with $d_1$ and $d_2$ degrees of freedom respectively, it is defined as:

$F = \frac{ Y_1/d_1 }{ Y_2/d_2 }$

The mean of the F-distribution is defined if $d_2 \gt 2$, and is given by

$\frac{d_2}{d_2-2}$

The variance is defined if $d_2 \gt 4$, and is given by 

$\frac{ 2 \cdot d_2^2 \cdot (d_1 + d_2 - 2) }{ d_2 \cdot (d_2 - 2)^2 \cdot (d_2 - 4) }$

We’ll code the visual from scratch for this one:

```python
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
sns.set_theme(style="darkgrid")

x = np.linspace(0, 5, 200)
d1_values = [1, 2, 5, 10]
d2_values = [1, 5, 10, 20]

plt.figure(figsize=(8, 6))

for d1, d2 in zip(d1_values, d2_values):
    pdf = st.f.pdf(x, d1, d2)
    plt.plot(x, pdf, label=r'$d_1 = {}, d_2 = {}$'.format(d1, d2))

plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.ylim(0, 1.0)  # Adjust based on the observed range of the F-distribution
plt.title('F-Distribution PDFs for Various Degrees of Freedom')
plt.legend(loc='upper right', fontsize=10)
plt.grid(alpha=0.3)
plt.show()
```

<img src="https://github.com/pw598/pw598.github.io/blob/main/_posts/images/f_dist.png?raw=true" style="height: 400px; width:auto;">




# References

- (n.d.). Distribution Explorer. [https://distribution-explorer.github.io/index.html](https://distribution-explorer.github.io/index.html)

- (n.d.). PyMC API. Distributions. [https://www.pymc.io/projects/docs/en/stable/api/distributions.html](https://www.pymc.io/projects/docs/en/stable/api/distributions.html)

- Blitzstein, J. (2019). Introduction to Probability (2nd ed.). Harvard University and Stanford University. [https://drive.google.com/file/d/1VmkAAGOYCTORq1wxSQqy255qLJjTNvBI/view](https://drive.google.com/file/d/1VmkAAGOYCTORq1wxSQqy255qLJjTNvBI/view)

