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

<img src="https://github.com/pw598/pw598.github.io/blob/main/_posts/images/bern_pmf.png?raw=true" style="height: 50px; width:auto;">

<img src="https://www.pymc.io/projects/docs/en/stable/_images/pymc-Bernoulli-1.png" style="height: 300px; width:auto;">

<i>https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Bernoulli.html</i>

As a distribution of probabilities for two outcomes, the expected value of the probability is the probability of the greater (or equally) weighted outcome, which is simply $p$. The variance (less intuitively) is $pq$.

In Bayesian statistics - the subject of a set of future articles -  the conjugate prior for the Bernoulli is the Beta (a continuous distribution), meaning that if the parameter $p$ of a Bernoulli distribution is taken from a Beta, then there is a closed form posterior distribution (the probability distribution of a set of parameters after observing data) that is also Beta distributed, with parameter values derived from that of the Beta ‘prior’ and that of the Bernoulli likelihood.



# Binomial

The Binomial distribution is a generalization of the Bernoulli, from one trial to multiple trials. It represents the likelihood of the number of successes $k$ in $n$ Bernoulli trials of success probability $p$.

#### PMF:

$P(Y = k) = \binom{n}{k} p^k q^{n-k}, ~~k \in \{0,1,2,\ldots,n\}$

where $\binom{n}{k}$ represents the binomial operation,

$\binom{n}{k} = \frac{n!}{k!(n-k)!}$

This ‘binomial coefficient’ determines the number of ways to arrange $k$ successes in $n$ trials. For a specific sequence of $k$ successes and $n−k$ failures, the probability is given by multiplying the probabilities of each trial outcome. Since the trials are independent, we have:

$p^k (1-p)^{n-k}$

Putting it together: the total probability of getting exactly $k$ successes in $n$ trials is the product of the number of ways to arrange $k$ successes and the probability of any specific arrangement of $k$ successes and $n−k$ failures.

$P(Y = k) = \binom{n}{k} p^k q^{n-k}$

<img src="https://www.pymc.io/projects/docs/en/stable/_images/pymc-Binomial-1.png" style="height: 300px; width:auto;">

<i>https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Binomial.html</i>

The conjugate prior of the Binomial distribution is also the Beta.



# Multinomial

The Multinomial distribution is a multivariate distribution that generalizes the Binomial to more than two categories of outcome. The marginal distribution of a Multinomial for two variables is Binomial. Each of $n$ objects is independently placed into one of $k$ categories, each with probability $p_k$.

$P(\mathbf{y}; {\mathbf{p}}, n) = \frac{n!}{y_1!\,y_2! \cdots y_K!}\,p_1^{y_1}\,p_2^{y_2} \cdots p_K^{y_K}$

The expected value is $np_k$. The variance is $np_k(1-p_k)$. The intuition is that if $n_1, \ldots, n_k$ do add up to $n$, then any particular way of putting $n_1$ objects into $k_1$, $n_2$ into $k_2$, etc., has probability $p_1^{n_1}, p_2^{n_2}, \ldots, p_k^{n_k}$, and there are $\frac{n!}{n_1! n_2! \ldots n_k!}$ ways to do this.

The conjugate prior of the Multinomial is the Dirichlet, which, like the Beta, we will get to in the next article, on continuous distributions.



# Poisson

The Poisson distribution is a limiting case of the Binomial, as $n \rightarrow \infty$ and  $p \rightarrow 0$. It’s more intuitive to think of it as a model for the frequency of rare events, with data defined as counts. It expresses the probability of a given number of events occurring in a fixed interval of time or space, in which events occur with a constant mean rate, and independently of the time since the last event. The derivation is natural, but a constraint toward usage is that the mean of the data is expected to be equal (or very close) to the variance.

For large $n$, using the approximation

$\binom{n}{k} \approx \frac{n^k}{k!}$

we have:

$\binom{n}{k}p^k (1-p)^{n-k} \approx \frac{n^k}{k!}p^k (1-p)^{n-k}$

As $n \rightarrow \infty$,

$\left( 1 - \frac{\lambda}{n} \right)^n \rightarrow e^{- \lambda}$

and 

$\left( 1 - \frac{\lambda}{n} \right)^{-k} \rightarrow 1$

Combining the results, we get:

#### PMF:

$P(Y=k) = \frac{e^{-\lambda}\lambda^k}{k!}$

<img src="https://www.pymc.io/projects/docs/en/stable/_images/pymc-Poisson-1.png" style="height: 300px; width:auto;">

<i>https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Poisson.html</i>

Both the expected value and variance are equal to $\lambda$.

The conjugate prior of the Poisson likelihood is the Gamma (continuous).



# Negative Binomial

The contrast between the Negative Binomial and the Binomial is that the Binomial defines a distribution in terms of a fixed number of trials and random probability, whereas the Negative Binomial represents a random number of trials and fixed probability. It asks, how many Bernoulli trials before a certain number of successes?

#### PMF Parameterization #1:

$f(y;\alpha,\beta) = \begin{pmatrix} y+\alpha-1 \\ \alpha-1 \end{pmatrix} \left(\frac{\beta}{1+\beta}\right)^\alpha \left(\frac{1}{1+\beta}\right)^y.$

$\alpha$ is the desired number of successes, and the probability of each Bernoulli trial is given by $\beta/(1+\beta)$. Generally speaking, $\alpha$ does not need to be an integer, so we can use an alternative parameterization in which the first term is not a binomial coefficient.

$f(y;\alpha,\beta) = \frac{\Gamma(y+\alpha)}{\Gamma(\alpha) \, y!}\,\left(\frac{\beta}{1+\beta}\right)^\alpha \left(\frac{1}{1+\beta}\right)^y.$

$\Gamma$ is the Gamma function, an extension to the factorial function to continuous values that is commonly used in the normalization of probability distributions.

Each of the above have expected value $\alpha/\beta$, and variance 

$\displaystyle{\frac{\alpha(1+\beta)}{\beta^2}}$

The Negative Binomial has several parameterizations, one of which is based upon a mean $\mu$ and dispersion parameter $\varphi$. In this case, $1/\varphi$ is the rate of overdispersion compared to a variance that equals the mean, and therefore, the Poisson is a limiting case of the Negative Binomial where $\varphi \rightarrow \infty$.

#### PMF Parameterization #2:

$f(y;\mu,\phi) = \frac{\Gamma(y+\phi)}{\Gamma(\phi) \, y!}\,\left(\frac{\phi}{\mu  +\phi}\right)^\phi\left(\frac{\mu}{\mu+\phi}\right)^y.$

In this parameterization, the expected value is $\mu$ and the variance is

$\displaystyle{\mu\left(1 + \frac{\mu}{\phi}\right)}$

https://distribution-explorer.github.io/ notes that there is yet another parameterization used in Scipy and Numpy:

#### PMF Parameterization #3:

$f(y;n, p) = \frac{\Gamma(y+n)}{\Gamma(n) \, y!}\,p^n \left(1-p\right)^y.$

<u><i>Expected Value:</i></u>

$\displaystyle{n\,\frac{1-p}{p}}$

<u><i>Variance:</i></u>

$\displaystyle{n\,\frac{1-p}{p^2}}$

The conjugate prior of the success probability $p$ is the Beta distribution.

<img src="https://www.pymc.io/projects/docs/en/stable/_images/pymc-NegativeBinomial-1.png" style="height: 300px; width:auto;">

<i>https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.NegativeBinomial.html</i>



# Geometric

The Geometric distribution is a special case of the Negative Binomial where $\varphi=1$. In other words, it models the number of trials until the first success.

#### PMF:

$f(y;p) = (1-p)^y \, p.$

<u><i>Expected Value:</i></u>

$\displaystyle{\frac{1-p}{p}}$

<u><i>Variance:</i></u>

$\displaystyle{\frac{1-p}{p^2}}$

<img src="https://www.pymc.io/projects/docs/en/stable/_images/pymc-Geometric-1.png" style="height: 300px; width:auto;">

<i>https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Geometric.html</i>

The conjugate prior for the Geometric likelihood is the Beta distribution.



# Hypergeometric

The Hypergeometric distribution is more directly connected to the Binomial than the Geometric; it is what the Binomial would be if sampling were done without replacement, instead of independently. Like the Binomial, it measures the probability of $k$ successes in $n$ trials.

#### PMF: 

The number of ways to choose $k$ successes from $K$ successes is given by $\binom{K}{k}$. The number of ways to choose $n−k$ failures from $N−k$ failures is given by $\binom{N-K}{n-k}$. The number of ways to choose $n$ items from $N$ items is given by $\binom{N}{n}$.

Putting it together, the probability of obtaining exactly $k$ successes in the sample is the ratio of the number of favorable outcomes to the total number of possible outcomes.

$P(X = k) = \frac{\binom{K}{k}\binom{N-k}{n-k}}{\binom{N}{n}}$

- $N$ is the population size
- $K$ is the number of success states in the population
- $n$ is the number of trials/draws
- $k$ is the number of observed successes

<u><i>Expected Value:</i></u>

$\mu = \frac{nK}{N}$

<u><i>Variance:</i></u>

$\frac{N-n}{N-1} ~n \frac{\mu}{n} ~\left( 1 - \frac{\mu}{n} \right)$

<img src="https://www.pymc.io/projects/docs/en/latest/_images/pymc-HyperGeometric-1.png" style="height: 300px; width:auto;">

<i>https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.HyperGeometric.html</i>



# What’s Next?

That concludes this summary of discrete probability distributions. The next two articles will focus on continuous distributions, roughly categorized into two separate families.



# References

- (n.d.). Distribution Explorer. [https://distribution-explorer.github.io/index.html](https://distribution-explorer.github.io/index.html)

- (n.d.). PyMC API. Distributions. [https://www.pymc.io/projects/docs/en/stable/api/distributions.html](https://www.pymc.io/projects/docs/en/stable/api/distributions.html)

- Blitzstein, J. (2019). Introduction to Probability (2nd ed.). Harvard University and Stanford University. [https://drive.google.com/file/d/1VmkAAGOYCTORq1wxSQqy255qLJjTNvBI/view](https://drive.google.com/file/d/1VmkAAGOYCTORq1wxSQqy255qLJjTNvBI/view)