---
layout: post
title:  "Bayesian Inference I: Bayesian Data Analysis"
date:   2024-12-23 00:00:00 +0000
categories: Probability Bayes
---

<p></p>

<img src="https://github.com/pw598/pw598.github.io/blob/main/_posts/images/bi1.png?raw=true" style="height: 250px; width:auto;">


# Outline 

<ul>
<li>Two Schools of Thought</li>
<li>What to Make of Priors</li>
<li>Choosing a Prior</li>
<li>Types of Priors</li>
    <ul>
    <li>Conjugate Priors</li>
    <li>Flat Priors</li>
    <li>Weakly Informative Priors</li>
    <li>Informative Priors</li>
    </ul>
<li>What Next?</li>
</ul>
	


# Two Schools of Thought

Whereas in frequentist statistics, the parameters of a model are considered fixed but unknown, in a Bayesian model, the parameters themselves are random variables, leading to a more probabilistic interpretation, which is what we tend to intuitively seek.

Let’s consider the Bayesian ‘credible interval’ as compared to its frequentist analog, the confidence interval. The Bayesian credible interval, at a 95% threshold, states that “there is a 95% chance the parameter will fall within this interval”, whereas with the frequentist confidence interval requires an interpretation like “if we repeated this experiment many times and calculated a confidence interval each time, 95% of those intervals would contain the true parameter.”

Bayesian inference utilizes Bayes’ theorem to update the probability of a hypothesis, dependent upon related conditions. In addition to the law of total probability, it is derived from Bayes’ rule, which is defined as:

$P(A|B) = \frac{ P(B|A)P(A) }{ P(B) }$

Nothing is controversial about this general formulation, as it follows directly from the law of conditional probability.

$P(A \text{ and } B) = P(A|B)P(B) = P(B|A)P(A)$

Incorporating the law of total probability,

$P(B) = \sum_{i=1}^n P(B|A_i) P(A_i)$

we can write:

$P(A|B) = \frac{ P(B|A)P(A) }{ \sum_{i=1}^n P(B|A_i)P(A_i) }$

$P(A|B)$ we call the posterior. If we were describing the parameters of a model (distribution) given data, we would analogously write $f(\theta|y)$ for a posterior distribution, where $f(\cdot)$ is the model, $\theta$ is shorthand for the parameters, and $y$ is a vector of data. We generalize the sum to an integral to accommodate for continuous distributions.



# What to Make of Priors

The subjective nature of injecting a prior distribution has historically been somewhat of a point of contention, though in recent decades, researchers and educators have been more accepting. It may not be obvious why this subjectiveness is justified; allow me to argue on its behalf.

A good starting point may be to point out that the assumptions made under the frequentist approach are not trivial. For example, we rely on the law of large numbers, and so quantifying uncertainty is premised upon the imaginary resampling of data. The sampling distribution is defined by the stopping and testing intuitions of the researcher, and different stopping or testing interactions yield different p-values and confidence intervals. 

At the same time, frequentism provides no formal mechanism for incorporating prior expectations about model parameters. Including such information can be highly valuable with, for example, a small-scale study for which information about the models in adjacent studies is available (e.g., in psychology). Even if the alignment with your study is such that the data from an external one could be directly incorporated, getting your hands on the raw data may be difficult, whereas model parameters are usually described. You can still communicate the details of your model in their entirety, it’s just that information about the choice of prior distributions and their parameters is now required.

Andrew Gelman, foremost author and professor in the field, often speaks of the ‘reproducibility crisis’, the tendency for small-scale academic studies to disagree even with very similar ones. Bayesian inference can provide a helpful regularizing effect, and with a large amount of data, the prior will have a minimal effect. As Andrew argues in <a href="https://statmodeling.stat.columbia.edu/2005/07/31/n_is_never_larg/">this</a> short blog post, when N is large, it’s time to further segment the data, given a capable method of analysis. Frequentism can get dangerous when data size is small, but Bayesian data analysis helps to avoid overfitting (and hierarchical models can be engineered).



# Choosing a Prior

Choosing a prior can be a little anxiety-inducing. How can we be sure it’s justifiable in the eyes of colleagues and critics? How much does being conservative correspond to being ‘fair’ or ‘objective’? Is it cheating to adjust the prior after looking at the data?

To answer the first question, there is no universally ‘objective’ answer, so it should be based upon something you and your team can justify. There will likely be some prior precedent, but either understand and agree, or adjust, and then be transparent and open to feedback.

I’ll avoid using the word ‘objective’ (<a href="http://www.stat.columbia.edu/~gelman/research/published/objectivityr5.pdf">here</a> is an interesting paper arguing its semantics in the context of Bayesian analysis), and speak toward how much being ‘fair’ corresponds to being conservative with our assumptions. It’s true that the sweet spot will be somewhere between the extremes of being too conservative and too assumptive.

As to whether it’s cheating to adjust the prior after looking at the data, strictly Bayesian principles suggest so, but leading researchers (including Gelman) conclude the prior can only be fully understood in the context of the likelihood (see <a href="http://www.stat.columbia.edu/~gelman/research/unpublished/prior_context.pdf">this paper</a>).



# Types of Priors

### Conjugate Priors:

Before technology offered limitless options over choices of prior-likelihood pairing, conjugate priors were heavily relied upon. They provide closed-form calculation of a posterior that is of the same form as the prior, for a given form of likelihood. In doing so, they reduce the task of calculating a complicated integral to simple operations upon point-form summary statistics. Even today, there is no faster method. 

An example is that a Beta prior with a Bernoulli or Binomial likelihood will combine to form a posterior that is itself Beta-distributed. Though the Beta is very flexible, and may take shapes between a Uniform and a dense spike, the method of conjugate priors is most restrictive compared to other methods, which do not constrain us to particular forms.

Specific conjugate-prior relationships will be the subject of a future article.

### Flat Priors:

Back to our discussion of conservatism vs. fairness. Is the prior of most fair judgment a flat one, wide one, lumpy one, spiky one? A flat (i.e., Uniform) prior assumes the same probability across the entire space, therefore providing as little information as possible. This can seem ‘objective’, but allows the data to dominate the posterior if a sufficient amount exists, and can give the illusion of being too neutral if data are small. Generally, the prior expectations gathered from prior or external knowledge will not represent a ‘flat’ outcome, but a flat prior can be sometimes justified.

### Empirical Priors:

A data-based prior can also seem most ‘objective’, as the results will be sure to align with the data, and like uninformative (flat) priors, it would serve as a default that one can explain in simple fashion. Generally, it is seen as double-dipping upon the likelihood, although formal methods do exist, including Jeffreys’ priors, which seek to avoid the potentially harmful effects of reparameterization under a data-based prior. The method of bootstrapping may also prove useful.

### Weakly Informative Priors:

Setting a prior consistent with at least somewhat-informed expectations will probably involve setting at least a weakly informative prior. This could be, for example, a Normal distribution with wide variance. We could also opt for heavier tails with a Student’s t-distribution, perhaps reducing the degrees of freedom down to the infinite variance of a Cauchy. If dealing with positive values, we could use a truncated ‘half’ version of any of the above, or if desiring a different shape, we can use any other distribution, so long as the parameters make for a broad range of values with significant probability.

Where to center the data is an issue one can easily solve by centering the data, but adjusting the probability density function is also an option.

### Informative Priors:

How informative is too informative will be context-dependent, but a general example is that if something follows a natural phenomenon, particularly with additive error or data-generating processes, then the results will follow a Normal distribution. We might have good reason to assume a standard Normal, N(0,1), in which case N(0,1) would be considered informative, and something slightly wider (or much wider) might be used to make our assumptions more conservative. Where exactly it turns from ‘informative’ to ‘weakly informative’ is somewhat subjective.

For what it’s worth, in terms of defaults, Andrew Gelman has expressed a preference for weakly informative priors in the past, and N(0,1) more recently, and fellow respected author and professor Aki Vehtari has expressed a fondness for t(3,0,1).



# What’s Next?

That concludes this first article on Bayesian inference. Following articles in the series will include code, and cover topics such as Bayesian regression with generalized linear models (GLMs).



# References

- Davidson-Pilon, C. Bayesian Methods for Hackers. [https://dataorigami.net/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/#contents](https://dataorigami.net/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/#contents)

- Gelman, A., Carlin, J., Stern, H., Rubin, D., Dunson, D., & Vehtari, A. (1995). Bayesian Data Analysis (3rd ed.). [https://www.researchgate.net/publication/46714374_Bayesian_data_analysis](https://www.researchgate.net/publication/46714374_Bayesian_data_analysis)

- McElreath, R. (2017). Statistical Rethinking (2nd ed.). [https://github.com/Booleans/statistical-rethinking/blob/master/Statistical%20Rethinking%202nd%20Edition.pdf](https://github.com/Booleans/statistical-rethinking/blob/master/Statistical%20Rethinking%202nd%20Edition.pdf)

- [Richard McElreath (Statistical Rethinking) Lectures](https://www.youtube.com/watch?v=FdnMWdICdRs&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus)

- Kruschke, J. K., & Liddell, T. M. (2017). The Bayesian New Statistics. Psychonomic Society. [https://doi.org/10.3758/s13423-016-1221-4](https://doi.org/10.3758/s13423-016-1221-4)

- Martin, O. (2024). Bayesian Analysis with Python (3rd ed.). Packt. [https://www.packtpub.com/en-ca/product/bayesian-analysis-with-python-9781805127161](https://www.packtpub.com/en-ca/product/bayesian-analysis-with-python-9781805127161)

- (2005, July 31). N is never large. Statistical Modeling, Causal Inference, and Social Science. [https://statmodeling.stat.columbia.edu/2005/07/31/n_is_never_larg/](https://statmodeling.stat.columbia.edu/2005/07/31/n_is_never_larg/)

- Gelman, A., & Hennig, C. (2017). Beyond subjective and objective in statistics. Columbia University. [http://www.stat.columbia.edu/~gelman/research/published/objectivityr5.pdf](http://www.stat.columbia.edu/~gelman/research/published/objectivityr5.pdf)

- Gelman, A., Simpson, D., & Betancourt, M. (2017). The Prior Can Often Only Be Understood in the Context of the Likelihood. Entropy. [http://www.stat.columbia.edu/~gelman/research/unpublished/prior_context.pdf](http://www.stat.columbia.edu/~gelman/research/unpublished/prior_context.pdf)

- Betancourt, M. (2017). How the Shape of a Weakly Informative Prior Affects Inferences. [https://mc-stan.org/users/documentation/case-studies/weakly_informative_shapes.html](https://mc-stan.org/users/documentation/case-studies/weakly_informative_shapes.html)

