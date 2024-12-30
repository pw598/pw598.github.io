---
layout: post
title:  "Probability I: Discrete Distributions"
date:   2024-12-04 00:00:00 +0000
categories: Probability
---

<p></p>

<img src="https://github.com/pw598/pw598.github.io/blob/main/_posts/images/pd1.png?raw=true" style="height: 450px; width:auto;">


# Outline 

- Random Variables
- Bernoulli
- Binomial
- Multinomial
- Poisson
- Negative Binomial
- Geometric
- Hypergeometric
- What’s Next?



# Random Variables

When we perform an experiment, we can calculate the probability that a random variable Y will take on a range of values, given its randomness. A random variable (RV) maps the sample space to the real domain, where it can take on one of many values, and probabilities are assigned to each of the outcomes.

Discrete random variables represent values in a finite countable set, or intervals of real numbers, and summarize probabilities with a probability mass function (PMF), providing an analytical expression of the outcome probabilities in which the probabilities are between $0$ and $1$, and sum to $1$.

Continuous probability distributions (the subject of the next two articles) have an infinite number of points along an unbroken function, and the probabilities are summarized by a probability density function (PDF). For many distributions, a closed-form cumulative density function (CDF) representing the accumulated sum or integral of the probabilities exists. All parametric distributions have ‘moments’ which include the expected value $E[Y]$ and the variance $Var[Y]$, with standard deviation equal to the square root of the variance.

You will encounter PMFs and PDFs expressed in ways that give alternative symbols to various parameters. You will also find alternative parameterizations, where for example, a term may be represented in reciprocal fashion, or as a relation to the expected value, rather than something more abstract.

I’ve aligned pretty consistently with https://distribution-explorer.github.io/ for the parameterizations below, though in some cases, have opted for symbols more commonly used in literature. This site does a good job of specifying the parameterizations used in various software (Scipy, Numpy, Stan, Julia) and provides some interactive exploration tools.

The code to generate the visuals I’ve linked to below are available in the user documentation for the Python library PyMC.



# Bernoulli

The Bernoulli distribution is, I would imagine, the most intuitive, as it describes the simplest type of experiment. A Bernoulli random variable represents a single trial with a binary outcome of success (represented by $1$) equal to probability $p$, and a probability of failure (represented by $0$) intuitively equal to $1-p$, represented symbolically as $q$.

$f(y;p) = \left\{ \begin{array}{ccc} q=1-p & & y = 0 \\[0.5em] p & & y = 1. \end{array} \right.$

<img src="https://www.pymc.io/projects/docs/en/stable/_images/pymc-Bernoulli-1.png" style="height: 300px; width:auto;">

<i>https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Bernoulli.html</i>

As a distribution of probabilities for two outcomes, the expected value of the probability is the probability of the greater (or equally) weighted outcome, which is simply $p$. The variance (less intuitively) is $pq$.

In Bayesian statistics - the subject of a set of future articles -  the conjugate prior for the Bernoulli is the Beta (a continuous distribution), meaning that if the parameter $p$ of a Bernoulli distribution is taken from a Beta, then there is a closed form posterior distribution (the probability distribution of a set of parameters after observing data) that is also Beta distributed, with parameter values derived from that of the Beta ‘prior’ and that of the Bernoulli likelihood.















