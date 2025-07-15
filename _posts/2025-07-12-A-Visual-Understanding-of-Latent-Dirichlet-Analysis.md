---
layout: post
title:  "A Visual Understanding of Latent Dirichlet Allocation (LDA)"
date:   2025-07-12 00:00:00 +0000
categories: TopicModeling LDA
---

Latent Dirichlet Allocation (LDA) is an unsupervised clustering method, largely used for the topic-modeling of text documents. While it produces a very rich interpretation, this comes with some complexity. The aim of this article is to provide visual and intuitive understanding, setting the stage for some real-word application.

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/library.jpg" style="height: 600px; width:auto;">



# Outline 

- Introduction
- Intuition
- The Dirichlet Distribution
  - The Stick-Breaking Analogy
- Plate Notation
  - The Unigram Model
  - Mixture of Unigrams
  - LDA
- Further Python Visualization
- Effect of Varying Parameters
- Solving LDA
- What's Next?



# Introduction

Introduced in <a href="https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf" target="_blank">this paper</a> by David Blei et al., Latent Dirichlet Allocation (LDA) is an unsupervised clustering method commonly used for topic-modeling of documents in a corpus (collection of documents). Because it is unsupervised, labels are neither required as input, nor explicitly produced as output. We do provide the number of topics $K$, expecting to find meaningful distinctions and associations, and adjusting our approach if not. LDA is a soft-clustering method, so unlike the crisp topic-predictions we obtain from a method like K-Means, it assigns a probability distribution over topics to each document.

As a quick sidenote, the 'LDA' we're referring to is not related to the dimension-reduction technique Linear Discriminant Analysis, which shares the same acronym.

<p>Latent Dirichlet Allocation is a generative model. Rather than a discriminative model, which predicts $P(Y|X)$ for labeling or inference, we model the joint probability $P(X,Y)$ to simulate data, with $X$ representing the observed data, and $Y$ representing latent (hidden) variables. In solving, we actually generate sample documents, mixtures of topics and words drawn from topic-specific distributions. The generated documents are nonsensical in terms of natural language, as we use the bag-of-words (BoW) approach to vectorization, counting word frequencies, but disregarding sequences. Nevertheless, the model tends to perform well, and delivers a very rich interpretation of results, compared to other common methods of topic-modeling. We can (and next article, will) collapse the distributions into class predictions, like a supervised method (referred to as sLDA).</p>



# Intuition

As a soft-clustering method, LDA facilitates overlapping topic assignments. Rather than each document belonging to exactly one cluster, LDA assigns:

<ol>
  <li>A probability distribution over topics to each document.</li>
  <ul>
    <li>e.g., a document is 40% represented by fiction, 20% by science, and so on with regard to other topics.</li>
  </ul>
  <p></p>
  <li>A probability distribution over words to each topic.</li>
  <ul>
    <li>e.g., a topic is represented 5% by "quantum", 2% by "teleport", and so on with regard to other words (perhaps thousands of them).</li>
  </ul>
</ol>

Imagine a library where many books are strewn across the tables and floor, with no labels or organization. This collection of books represents our corpus. You need to create an algorithm to sort them into sections, but you do not know which categorizations exist beforehand. The books are our documents, the unknown categorizations are our topics, and the words are clues as to which topic a book belongs to. If a book has words like "experiment" and "hypothesis", the latent category corresponding to 'science' may be the one it predominantly belongs to. It may also have a strong probability of belonging to 'biology' or 'physics', and have small (often very small) probabilities of belonging to many other categories.

I promise I won't subject us to (too) much math, but some is necessary to describe the Dirichlet distribution and its relatives. After all, it's in the name. Feel free to skim over the mathematical notation, it won't harm the ability to gain insight from the visuals.




# The Dirichlet Distribution

We can describe the Dirichlet in terms of its relationship to the Multinomial, which represents probabilities over discrete categorical  outcomes. This is perfect for word counts, as the words represent discrete categories, and our BoW vectors can be normalized to represent the probabilities of each.


<u><i>Multinomial Probability Mass Function (PMF):</i></u>

$P(X_1 = x_1, X_2 = x_2, \ldots, X_k = x_k) = \frac{n!}{x_1! x_2! \cdots x_V!} p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k}$

- The $ k $ outcomes correspond to the number of words in the vocabulary.
- The $ n $ trials correspond to the total number of words in a document. 
- The counts $x_1, x_2, \ldots, x_k$ represent the number of times each word appears in the document.

With a Binomial, representing two discrete potential outcomes, we can describe the probability of both outcomes (as knowing one determines the other) using the Beta distribution, dubbed the 'distribution of probabilities' because of its range of $0$ to $1$. You can visit <a href="https://pw598.github.io/probability/2024/12/04/Probability-Distributions-I-Discrete-Distributions.html" target="_blank">this article</a> for further detail on the Binomial, and <a href="https://pw598.github.io/probability/202 4/12/09/Probability-Distributions-II-Continuous-Distributions-I.html" target="_blank">this article</a> for further detail on the Beta.

Just as the Multinomial generalizes the Binomial to occurrences of multiple categories, the Dirichlet is a continuous distribution that generalizes the Beta to probabilities of multiple categories. Therefore, the Dirichlet serves as a prior (a conjugate prior) to the Multinomial. The parameters of the Dirichlet are a vector $\mathbf{\alpha} = [\alpha_1, \alpha_2, \ldots, \alpha_k]$ of concentration parameters, with each element corresponding to a discrete category of a Multinomial.


<i><u>Dirichlet Probability Density Function (PDF):</u></i>

<p>$f(p_1, \ldots, p_k | \alpha_1, \ldots, \alpha_k) = \frac{1}{B(\alpha)} \prod_{i=1}^k p_i^{\alpha_i - 1}$</p>

- $p_1, \ldots, p_k$ are probabilities of the components of a k-dimensional simplex.

- $\alpha_1, \ldots, \alpha_k$ are the concentration parameters of the Dirichlet prior (for the document-topic distribution $\mathbf{\theta}$ in LDA).

- $\frac{1}{B(\alpha)}$ is a normalizing constant that ensures the PDF integrates to 1. $B(\alpha)$ is the <a href="https://en.wikipedia.org/wiki/Beta_function" target="_blank">Beta function</a>, a somewhat hairy mathematical formula which involves the <a href="https://simple.wikipedia.org/wiki/Gamma_function" target="_blank">Gamma function</a>, for which I'll also refer you to Wikipedia; but is an extension of the factorial function from discrete to real numbers, and therefore has utility in probability distributions.

- $\prod_{i=1}^k$ is the 'kernel' of the Dirichlet PDF, weighting each probability $p_i$ by its corresponding concentration parameter $\alpha_i$.


In LDA, we have both a distribution of topics over documents, and a distribution of words over topics, so to avoid using two vectors of parameters denoted $\mathbf{\alpha}$, we name the vector of topic proportions over documents $\mathbf{\alpha}$, and the vector of topic proportions over words $\mathbf{\beta}$.

If you have read my first article on <a href="https://pw598.github.io/probability/2024/12/09/Probability-Distributions-II-Continuous-Distributions-I.html" target="_blank">continuous distributions</a> (the second on distributions in general), you may have noticed that the Dirichlet is the only one not visualized. This is because of its abstract nature. We can visualize it as a simplex, which you can think of as an abstraction of a triangle, but I figured that just raises more questions. Now would be a good time to elaborate.

A simplex is the simplest possible polytope (a geometric object with flat sides), generalized to any dimension. A simplex in zero dimensions is a point, in one dimension is a line, and in two dimensions is a triangle. In three dimensions, we can visualize it as a tetrahedron. $K$, our number of topics, represents the number of vertices, so $K=3$ in two dimensions, $K=4$ in three dimensions, etc. 

With regard to its visualization in the context of LDA, we can think of a subplot of bar charts, with both rows and columns representing discrete categories (such as topics and words), and their lengths reflecting probabilities. A nice property of this is that we can abstract it to an arbitrary number of dimensions, however it is only partially correct with regard to visualizing the Dirichlet, which is continuous rather than discrete. To provide some further intuition into the simplex representation, we can think of the following 'stick-breaking' analogy.


## The Stick-Breaking Analogy

Consider a stick of length $1$, representing a probability mass of $100\%$, which we break into $K$ pieces, representing probabilities $[p_1, p_2, \ldots, p_k]$, which sum to $1$. The probability represented by the first piece we break off, $p_1$, is chosen by a Beta distribution, leaving a mass of $(1-p_1)$ behind. Then, we break off a second piece, $p_2$, from the remainder, and continue until $K$ pieces. These pieces form a probability vector, and always fit together to form a simplex. The Dirichlet distribution governs these splits, generalizing the Beta to any $K$. Large $\alpha_i$ favor bigger pieces for component $i$, equal $\alpha_i$ yield balanced splits, and small $\alpha_i$ produce sparse splits. 

Still a little abstract, I get it. The below visualizations will relate the Dirichlet directly to LDA, and hopefully provide further intuition. We'll make reference to probabilistic graphical models (PGMs); specifically, plate notation. You may be familiar with PGMs if you've worked with Bayesian modeling through a library like PyMC. These visualize the relationships and direction of influence between upstream and downstream parameters, each of which have a random component, through nodes and arrows. Plate notation is very similar, but describes iterative activity as well. LDA is indeed a Bayesian process, because of the sampling algorithm required to obtain the soft-clustering, and the ability/necessity to set priors over hyperparameters.



# Plate Notation

Plate notation is a form of PGM, which incorporates the following elements.

- Circles (nodes) for random variables.
- Rectangular plates for replication over samples or dimensions.
- Arrows for dependencies between variables.

To provide intuition, we'll build up from the simplest type of topic model, the unigram, to the mixture of unigrams, and then the hierarchical Bayesian model represented by LDA.


## Unigram Model

A unigram implies a single, Multinomial distribution over the entire vocabulary of the corpus, collectively. The probability of the topic being defined by a particular word is relative to its frequency in the bag-of-words distribution. In the plate diagram below, node $x$ represents a particular word count, the $D$ in the encompassing plate can be thought of as a loop over dimensions, and the $N$ in the plate encompassing that as a loop over samples.

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/unigram_plate.png" style="height: 300px; width:auto;">


The mathematical formula this represents is:

$p(x) = \prod_{j=1}^D ~p(x_j)$

And as pseudocode:

```
for i = 1 to N:
  for j = 1 to D:
    x(i,j) ~ p(x)
```

If we consider the vertices of a triangle to represent each word in a 3-word, 3-topic vocabulary, a point representing the probability vector that the singular topic is defined by will be roughly centered in the middle of a triangle, assuming a sufficiently large and unbiased corpus. A colormap or shading-scale would also indicate greatest density in the center, indicative of a roughly uniform distribution over topics.

The triangular diagrams below refer to a simple toy corpus, consisting of the following 6 documents:

```python
corpus = [
    "space space car computer",
    "space car computer",
    "car car space computer",
    "car space car",
    "computer computer space car",
    "computer computer computer",
]
```

The first diagram will represent the unigram model described directly above. The vector of probabilities notated above the red dot indicate the estimated probability that the corpus belongs to each of the words represented by the vertices.

<details markdown="1">
  <summary>View Code</summary>

  ```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Corpus with balanced but distinct topics
corpus = [
    "space space car computer",
    "space car computer",
    "car car space computer",
    "car space car",
    "computer computer space car",
    "computer computer computer",
]

# Vectorize the corpus
vectorizer = CountVectorizer(vocabulary=["space", "car", "computer"])
X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()
word_counts = X.toarray()

# Unigram Model: Single distribution based on word frequencies
unigram_probs = np.sum(word_counts, axis=0) / np.sum(word_counts)

# Function to convert probabilities to 2D simplex coordinates
def probs_to_simplex(probs):
    x = probs[:, 1] + 0.5 * probs[:, 2]
    y = np.sqrt(3) / 2 * probs[:, 2]
    return x, y

# Convert unigram probabilities to simplex coordinates
unigram_x, unigram_y = probs_to_simplex(np.array([unigram_probs]))

# Plotting the simplex visualization
fig, ax = plt.subplots(figsize=(6, 6.5))  # Height for title space

# Define triangle vertices
triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

# Function to plot simplex triangle
def plot_simplex(ax):
    ax.plot([triangle[0, 0], triangle[1, 0], triangle[2, 0], triangle[0, 0]],
            [triangle[0, 1], triangle[1, 1], triangle[2, 1], triangle[0, 1]], 'k-')
    ax.text(-0.05, -0.05, feature_names[0], fontsize=15, ha='right')
    ax.text(1.05, -0.05, feature_names[1], fontsize=15, ha='left')
    ax.text(0.5, np.sqrt(3)/2 + 0.06, feature_names[2], fontsize=15, ha='center')
    ax.set_aspect('equal')
    ax.axis('off')

# Plot the unigram model
plot_simplex(ax)
ax.scatter(unigram_x, unigram_y, facecolors='red', edgecolors='none', s=150)
ax.scatter(unigram_x, unigram_y, facecolors='none', edgecolors='red', s=150, linewidth=2.0)
ax.text(unigram_x[0], unigram_y[0] + 0.06, 
        f'({unigram_probs[0]:.2f}, {unigram_probs[1]:.2f}, {unigram_probs[2]:.2f})', 
        fontsize=12, ha='center', va='bottom')
unigram_legend = ax.scatter([], [], facecolors='red', edgecolors='none', s=150, label='Corpus Distribution')
ax.legend(handles=[unigram_legend], fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.05))
ax.set_title("Unigram Model", fontsize=20, pad=30)

plt.tight_layout()
plt.savefig('unigram_model.png', dpi=300, bbox_inches='tight') 
plt.show()
  ```
</details> 


<p></p>

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/unigram_model.png" style="height: 450px; width:auto;">



## Mixture of Unigrams

A mixture of unigrams extends the unigram approach by adding an additional variable, introducing latent topics as discrete components that generate the observed data. Each document is assumed to be produced by selecting a single topic from a set of predefined topics, with each topic defined by a Multinomial distribution over the vocabulary, and words within the document drawn independently from that topic's distribution. This allows the clustering of documents into thematic groups.

<p>So, whereas the unigram collapses all text into a single distribution, ignoring thematic diversity, the mixture of unigrams accomodates one distribution per topic, capturing document-level variation. The plate diagram below adds a node $z$ to the unigram model, representing the latent topic that generated the observed data (as the mixture model is generative). For each sample, we generate a topic $z_i$ so there is one topic per sample. From this topic, we loop through each of our $D$ words, and for each of these words, generate a count $x_{ij}$.</p>

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/mixture_plate.png" style="height: 300px; width:auto;">

Represented mathematically:

<p>$p(x) = \sum_z ~p(z) ~\prod_{j=1}^D ~p(x_j | z)$</p>

And in pseudocode:

```
for i = 1 to N:
  z(i) ~ p(z)
  for j = 1 to D:
    x(i,j) ~ p(x|z = z(i))
```

Geometrically, in our triangle representing a 3-word, 3-topic vocabulary, we would have one dot per topic, with some closer to particular vertices than others are to their respective vertices. The vertex with the dot closer to it than any other dot to other vertices (i.e., the one with the greatest-value element) represents the word that defines the crisp prediction of a particular topic.


<details markdown="1">
  <summary>View Code</summary>

  ```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Corpus with balanced but distinct topics
corpus = [
    "space space car computer",
    "space car computer",
    "car car space computer",
    "car space car",
    "computer computer space car",
    "computer computer computer",
]

# Vectorize the corpus
vectorizer = CountVectorizer(vocabulary=["space", "car", "computer"])
X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()
word_counts = X.toarray()

# Mixture of Unigrams: Hard clustering with controlled initialization
def mixture_of_unigrams(X, n_components, random_state=42):
    np.random.seed(random_state)
    n_docs, n_words = X.shape
    # Initialize clusters based on dominant words
    cluster_assignments = np.array([0, 0, 1, 1, 2, 2])  # Space, car, computer clusters
    topic_probs = np.zeros((n_components, n_words))
    for k in range(n_components):
        cluster_docs = X[cluster_assignments == k]
        topic_probs[k] = np.sum(cluster_docs, axis=0)
        topic_probs[k] /= topic_probs[k].sum() + 1e-10  # Normalize
    return topic_probs

# Compute mixture probabilities
n_topics = 3
mixture_probs = mixture_of_unigrams(word_counts, n_topics)

# Determine crispness for coloring (crisp if any word's probability >= 0.7)
crisp_threshold = 0.7
facecolors = ['blue' if np.any(probs >= crisp_threshold) else 'none' for probs in mixture_probs]
edgecolors = ['blue' for _ in mixture_probs]

# Function to convert probabilities to 2D simplex coordinates
def probs_to_simplex(probs):
    x = probs[:, 1] + 0.5 * probs[:, 2]
    y = np.sqrt(3) / 2 * probs[:, 2]
    return x, y

# Convert mixture probabilities to simplex coordinates
mixture_x, mixture_y = probs_to_simplex(mixture_probs)

# Plotting the simplex visualization
fig, ax = plt.subplots(figsize=(6, 6.5))  # Height for title space

# Define triangle vertices
triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

# Function to plot simplex triangle
def plot_simplex(ax):
    ax.plot([triangle[0, 0], triangle[1, 0], triangle[2, 0], triangle[0, 0]],
            [triangle[0, 1], triangle[1, 1], triangle[2, 1], triangle[0, 1]], 'k-')
    ax.text(-0.05, -0.05, feature_names[0], fontsize=15, ha='right')
    ax.text(1.05, -0.05, feature_names[1], fontsize=15, ha='left')
    ax.text(0.5, np.sqrt(3)/2 + 0.06, feature_names[2], fontsize=15, ha='center')
    ax.set_aspect('equal')
    ax.axis('off')

# Plot the mixture of unigrams
plot_simplex(ax)
for i, (x, y, fc, ec) in enumerate(zip(mixture_x, mixture_y, facecolors, edgecolors)):
    ax.scatter(x, y, facecolors=fc, edgecolors='none', s=150)
    ax.scatter(x, y, facecolors='none', edgecolors=ec, s=150, linewidth=2.0)
    ax.text(x, y + 0.06, f'T{i+1}: ({mixture_probs[i][0]:.2f}, {mixture_probs[i][1]:.2f}, {mixture_probs[i][2]:.2f})', 
            fontsize=12, ha='center', va='bottom')
mixture_legend = ax.scatter([], [], facecolors='blue', edgecolors='none', s=150, label='Topic Distributions')
ax.legend(handles=[mixture_legend], fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.05))
ax.set_title("Mixture of Unigrams", fontsize=20, pad=30)

plt.tight_layout()
plt.savefig('mixture_model.png', dpi=300, bbox_inches='tight') 
plt.show()
  ```
</details> 


<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/mixture_model.png" style="height: 450px; width:auto;">



## LDA

LDA is a hierarchical extension of probabilistic topic models, positing that each document exhibits a mixture of topics, and that for each word in a given document, a topic is sampled from the list of topics.

This is similar to the mixture of unigrams in terms of foundation and objectives, as both are unsupervised probabilistic framewords aimed at uncovering latent topics as word distributions. However, LDA differs in its granularity and flexibility, offering soft, mixed-membership representations of topics, whereas the mixture model performs hard-clustering.

In the plate diagram below, $\mathbf{\alpha}$ and $\mathbf{\beta}$ are fixed parameters, because they appear outside of the plates. We can encode our prior expectations regarding these concentration parameters, but its quite typical to set a uniform prior, using $1/K$.

<ul>
  <li><p>$\theta$, sampled per documents, governs topic proportions. $\theta_d$ is the document-topic distribution for document $d$, $p(z|d)$, drawn from a Dirichlet parameterized by $\mathbf{\alpha}$.</p></li>

  <li><p>$z$, sampled per word in each document, assigns topics based on $\theta$. $z_{d,n}$ is the topic assignment for the $n^{th}$ word in document $d$, drawn from the Multinomial distribution parameterized by $\theta_d$.</p></li>

  <li><p>$x_{d,n}$ is the $n^{th}$ word in each document, assigned topics based on $\theta$.</p></li>
</ul>

<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/lda_plate.png" style="height: 300px; width:auto;">

It might be most intuitive to provide the pseudocode before the math.

```
α, β = fixed parameters
for i = 1 to N:
    θ(i) ~ Dirichlet(α)
    for j = 1 to D:
        z(i,j) ~ Multinomial(θ(i))
        x(i,j) ~ Multinomial(β, z(i,j))
```

For all the hairy mathematical details, I'll refer you to <a href="https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation">Wikipedia</a> or the <a href="https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf" target="_blank">original paper</a> (there will be some differences, in terms of notation, and whether referring to a plurality of corpora), but explain it up to a certain level of detail. A point of clarification I'll provide about the model parameters is that, in the mathematics of LDA, there are several vector-valued parameters:

- $\mathbf{\theta}_d$ is the document-topic distribution of length $K$.
- $\mathbf{\alpha}$ is the Dirichlet prior for $\mathbf{\theta}_d$, of length $K$.
- $\mathbf{\beta}$ is the Dirichlet prior for $\mathbf{\phi}_k$, of length $V$ for vocabulary.
- $\mathbf{\phi}_k$ is the topic-word distribution, of length $V$.

<p>Our mission is to infer the latent topic structure of a corpus, represented by the document-topic distribution $\mathbf{\theta}_d$ and topic-word distributions $\mathbf{\phi}_k$, given the model parameters $\mathbf{\alpha}$ and $\mathbf{\beta}$. The model seeks to infer those distributions by maximizing the probability of observing the document's words given the model parameters, $p(\mathbf{w}_d | \mathbf{\alpha}, \mathbf{\beta})$, which is typically approximated through <a href="https://en.wikipedia.org/wiki/Gibbs_sampling" target="_blank">Gibbs sampling</a> or <a href="https://arxiv.org/pdf/2108.13083v2" target="_blank">variational inference</a>.</p>

An overview of the generative process is as follows:

<ol>
  <li><p>For each topic $1, \ldots, K$:</p></li>
  <ul>
    <li><p>Draw a topic-word distribution $\mathbf{\phi}_k \sim \text{Dirichlet}(\mathbf{\beta})$, where $\mathbf{\theta}_k$ is a $V$-dimensional vector of word probabilities, and $\mathbf{\beta}$ is the Dirichlet prior parameter.</p></li>
  </ul>
  <li><p>For each document $d = 1, \ldots, D$:</p></li>
  <ul>
    <li><p>Draw a document-topic distribution $\mathbf{\theta}_d \sim \text{Dirichlet}(\mathbf{\alpha})$, where $\mathbf{\theta}_d$ is a $K$-dimensional vector of topic proportions, and $\mathbf{\alpha}$ is the Dirichlet prior parameter.</p></li>
    <li>For each word position in $n = 1, \ldots, N_d$ in document $d$:</li>
    <ul>
      <li>Draw a topic assignment $z_{d,n} \in [1, \ldots, V]$ is the observed word from the vocabulary, conditioned on the topic $z_{d,n}$.</li>
    </ul>
  </ul>
</ol>

A more concentrated document-topic distribution, driven by a large $\mathbf{\alpha}$, means documents are more likely to be dominated by a smaller number of topics. A more concentrated topic-word distribution, driven by a large $\mathbf{\beta}$, means topics are dominated by fewer words; i.e., higher probabilities for specific words within each topic.

Geometrically, the interpretation is similar to the mixture of unigrams. The difference is that, rather than a point within the triangle defining a singular topic chosen, there are degrees to which each of the topics are represented for each document.


<details markdown="1">
  <summary>View Code</summary>

  ```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Corpus with balanced but distinct topics
corpus = [
    "space space car computer",
    "space car computer",
    "car car space computer",
    "car space car",
    "computer computer space car",
    "computer computer computer",
]

# Vectorize the corpus
vectorizer = CountVectorizer(vocabulary=["space", "car", "computer"])
X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

# LDA: Fit with higher priors
n_topics = 3
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                               doc_topic_prior=1.0, topic_word_prior=1.0)
lda.fit(X)
lda_probs = lda.components_ / lda.components_.sum(axis=1, keepdims=True)

# Compute topic role as max document-topic probability and scale for transparency
doc_topic_dist = lda.transform(X)  # P(z|d) for each document
topic_roles = np.max(doc_topic_dist, axis=0)  # Max P(z|d) for each topic
# Scale to [0.2, 1.0] for visible but distinct transparency
lda_alphas = 0.2 + 0.8 * (topic_roles - np.min(topic_roles)) / (np.max(topic_roles) - np.min(topic_roles) + 1e-10)

# Function to convert probabilities to 2D simplex coordinates
def probs_to_simplex(probs):
    x = probs[:, 1] + 0.5 * probs[:, 2]
    y = np.sqrt(3) / 2 * probs[:, 2]
    return x, y

# Convert LDA probabilities to simplex coordinates
lda_x, lda_y = probs_to_simplex(lda_probs)

# Plotting the simplex visualization
fig, ax = plt.subplots(figsize=(6, 6.5))  # Height for title space

# Define triangle vertices
triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

# Function to plot simplex triangle
def plot_simplex(ax):
    ax.plot([triangle[0, 0], triangle[1, 0], triangle[2, 0], triangle[0, 0]],
            [triangle[0, 1], triangle[1, 1], triangle[2, 1], triangle[0, 1]], 'k-')
    ax.text(-0.05, -0.05, feature_names[0], fontsize=15, ha='right')
    ax.text(1.05, -0.05, feature_names[1], fontsize=15, ha='left')
    ax.text(0.5, np.sqrt(3)/2 + 0.06, feature_names[2], fontsize=15, ha='center')
    ax.set_aspect('equal')
    ax.axis('off')

# Plot the LDA model
plot_simplex(ax)
for i, (x, y, alpha) in enumerate(zip(lda_x, lda_y, lda_alphas)):
    ax.scatter(x, y, facecolors='green', edgecolors='none', s=150, alpha=alpha)
    ax.scatter(x, y, facecolors='none', edgecolors='darkgreen', s=150, linewidth=2.0)
    ax.text(x, y + 0.06, f'T{i+1}: ({lda_probs[i][0]:.2f}, {lda_probs[i][1]:.2f}, {lda_probs[i][2]:.2f})', 
            fontsize=12, ha='center', va='bottom')
lda_legend = ax.scatter([], [], facecolors='green', edgecolors='none', s=150, label='Topic Distributions')
ax.legend(handles=[lda_legend], fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.05))
ax.set_title("LDA", fontsize=20, pad=30)

plt.tight_layout()
plt.savefig('lda_plot.png', dpi=300, bbox_inches='tight') 
plt.show()
  ```
</details> 


<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/lda_model.png" style="height: 450px; width:auto;">



# Further Python Visualization

To take the visualization a step further, let's include the distribution of topics over documents.

<details markdown="1">
  <summary>View Code</summary>

  ```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Corpus with balanced but distinct topics
corpus = [
    "space space car computer",
    "space car computer",
    "car car space computer",
    "car space car",
    "computer computer space car",
    "computer computer computer",
]

# Vectorize the corpus
vectorizer = CountVectorizer(vocabulary=["space", "car", "computer"])
X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()
word_counts = X.toarray()

# Unigram Model: Single distribution based on word frequencies
unigram_probs = np.sum(word_counts, axis=0) / np.sum(word_counts)

# Mixture of Unigrams: Hard clustering with controlled initialization
def mixture_of_unigrams(X, n_components, random_state=42):
    np.random.seed(random_state)
    n_docs, n_words = X.shape
    cluster_assignments = np.array([0, 0, 1, 1, 2, 2])  # Space, car, computer clusters
    topic_probs = np.zeros((n_components, n_words))
    for k in range(n_components):
        cluster_docs = X[cluster_assignments == k]
        topic_probs[k] = np.sum(cluster_docs, axis=0)
        topic_probs[k] /= topic_probs[k].sum() + 1e-10  # Normalize
    return topic_probs, cluster_assignments

n_topics = 3
mixture_probs, mixture_assignments = mixture_of_unigrams(word_counts, n_topics)

# Determine crispness for Mixture of Unigrams
crisp_threshold = 0.7
facecolors = ['blue' if np.any(probs >= crisp_threshold) else 'none' for probs in mixture_probs]
edgecolors = ['blue' for _ in mixture_probs]

# LDA: Fit with higher priors
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                               doc_topic_prior=1.0, topic_word_prior=1.0)
lda.fit(X)
lda_probs = lda.components_ / lda.components_.sum(axis=1, keepdims=True)

# Compute topic role as max document-topic probability and scale for transparency
doc_topic_dist = lda.transform(X)  # P(z|d) for each document
topic_roles = np.max(doc_topic_dist, axis=0)  # Max P(z|d) for each topic
lda_alphas = 0.2 + 0.8 * (topic_roles - np.min(topic_roles)) / (np.max(topic_roles) - np.min(topic_roles) + 1e-10)

# Function to convert probabilities to 2D simplex coordinates
def probs_to_simplex(probs):
    x = probs[:, 1] + 0.5 * probs[:, 2]
    y = np.sqrt(3) / 2 * probs[:, 2]
    return x, y

# Convert probabilities to simplex coordinates
unigram_x, unigram_y = probs_to_simplex(np.array([unigram_probs]))
mixture_x, mixture_y = probs_to_simplex(mixture_probs)
lda_x, lda_y = probs_to_simplex(lda_probs)

# Plotting the visualizations (2x3 grid)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Define triangle vertices for simplex plots
triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

# Function to plot simplex triangle
def plot_simplex(ax):
    ax.plot([triangle[0, 0], triangle[1, 0], triangle[2, 0], triangle[0, 0]],
            [triangle[0, 1], triangle[1, 1], triangle[2, 1], triangle[0, 1]], 'k-')
    ax.text(-0.05, -0.05, feature_names[0], fontsize=15, ha='right')
    ax.text(1.05, -0.05, feature_names[1], fontsize=15, ha='left')
    ax.text(0.5, np.sqrt(3)/2 + 0.06, feature_names[2], fontsize=15, ha='center')
    ax.set_aspect('equal')
    ax.axis('off')

# Unigram Plot (Top-left)
plot_simplex(axes[0, 0])
axes[0, 0].scatter(unigram_x, unigram_y, facecolors='red', edgecolors='none', s=150)
axes[0, 0].scatter(unigram_x, unigram_y, facecolors='none', edgecolors='red', s=150, linewidth=2.0)
axes[0, 0].text(unigram_x[0], unigram_y[0] + 0.06, 
                f'({unigram_probs[0]:.2f}, {unigram_probs[1]:.2f}, {unigram_probs[2]:.2f})', 
                fontsize=12, ha='center', va='bottom')
unigram_legend = axes[0, 0].scatter([], [], facecolors='red', edgecolors='none', s=150, label='Corpus Distribution')
axes[0, 0].legend(handles=[unigram_legend], fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.05))
axes[0, 0].set_title("Unigram Model", fontsize=18, pad=28)

# Mixture of Unigrams Plot (Top-middle)
plot_simplex(axes[0, 1])
for i, (x, y, fc, ec) in enumerate(zip(mixture_x, mixture_y, facecolors, edgecolors)):
    axes[0, 1].scatter(x, y, facecolors=fc, edgecolors='none', s=150)
    axes[0, 1].scatter(x, y, facecolors='none', edgecolors=ec, s=150, linewidth=2.0)
    axes[0, 1].text(x, y + 0.06, f'T{i+1}: ({mixture_probs[i][0]:.2f}, {mixture_probs[i][1]:.2f}, {mixture_probs[i][2]:.2f})', 
                    fontsize=12, ha='center', va='bottom')
mixture_legend = axes[0, 1].scatter([], [], facecolors='blue', edgecolors='none', s=150, label='Topic Distributions')
axes[0, 1].legend(handles=[mixture_legend], fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.05))
axes[0, 1].set_title("Mixture of Unigrams", fontsize=18, pad=28)

# LDA Plot (Top-right)
plot_simplex(axes[0, 2])
for i, (x, y, alpha) in enumerate(zip(lda_x, lda_y, lda_alphas)):
    axes[0, 2].scatter(x, y, facecolors='green', edgecolors='none', s=150, alpha=alpha)
    axes[0, 2].scatter(x, y, facecolors='none', edgecolors='darkgreen', s=150, linewidth=2.0)
    axes[0, 2].text(x, y + 0.06, f'T{i+1}: ({lda_probs[i][0]:.2f}, {lda_probs[i][1]:.2f}, {lda_probs[i][2]:.2f})', 
                    fontsize=12, ha='center', va='bottom')
lda_legend = axes[0, 2].scatter([], [], facecolors='green', edgecolors='none', s=150, label='Topic Distributions')
axes[0, 2].legend(handles=[lda_legend], fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.05))
axes[0, 2].set_title("LDA", fontsize=18, pad=28)

# Unigram Document-Topic Distribution Plot (Bottom-left)
doc_labels = [f'Doc {i+1}' for i in range(len(corpus))]
axes[1, 0].bar(doc_labels, np.ones(len(corpus)), color='red', label='Topic 1')
axes[1, 0].set_xlabel("Documents", fontsize=14)
axes[1, 0].set_ylabel("Topic Proportion")
axes[1, 0].legend()

# Mixture of Unigrams Document-Topic Assignment Plot (Bottom-middle)
topic_labels = [f'Topic {i+1}' for i in range(n_topics)]
colors = ['#0000FF', '#3333FF', '#6666FF']  # Blue shades
alphas = [1.0, 0.7, 0.4]
mixture_doc_topic = np.zeros((len(corpus), n_topics))
for d, k in enumerate(mixture_assignments):
    mixture_doc_topic[d, k] = 1.0
bottom = np.zeros(len(corpus))
for i in range(n_topics):
    axes[1, 1].bar(doc_labels, mixture_doc_topic[:, i], bottom=bottom, label=topic_labels[i], color=colors[i], alpha=alphas[i])
    bottom += mixture_doc_topic[:, i]
axes[1, 1].set_xlabel("Documents", fontsize=14)
axes[1, 1].set_ylabel("Topic Assignment")
axes[1, 1].legend()

# LDA Document-Topic Distribution Plot (Bottom-right)
colors = ['#00FF00', '#00CC00', '#009900']  # Green shades
bottom = np.zeros(len(corpus))
for i in range(n_topics):
    axes[1, 2].bar(doc_labels, doc_topic_dist[:, i], bottom=bottom, label=topic_labels[i], color=colors[i])
    bottom += doc_topic_dist[:, i]
axes[1, 2].set_xlabel("Documents", fontsize=14)
axes[1, 2].set_ylabel("Topic Proportions")
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('subplot_with_topic_dists.png', dpi=300, bbox_inches='tight')
plt.show()
  ```
</details> 


<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/subplot_with_topic_dists.png" style="height: 500px; width:auto;">


We see that the unigram model, because it is only able to infer a distribution over the entire corpus, must select the same topic across documents. The mixture of unigrams, as promised, is able to assign different topics to different documents. But the LDA distribution of topics over documents is more granular, providing a distribution of topics over each document. 

In Scikit-Learn's <code>LatentDirichletAllocation</code> class, $\mathbf{\phi}_k$ corresponds to <code>lda.probs_list</code>, accessible thorugh <code>lda.components_</code>, and visualized in the plots by the top-row simplexes' topic-word distributions. $\mathbf{\theta}_d$ corresponds to <code>doc_topic_dists</code>, produced by the transformation <code>lda.transform(X)</code>. These are visualized by the bottom-row charts, showing the topic proportions per document.

<p>The marginal likelihood $P(\mathbf{w}_d | \mathbf{\alpha}, \mathbf{\beta})$ is implicitly optimized during <code>lda.fit(X)</code>, and <code>lda.score(X)</code> provides the <a href="https://en.wikipedia.org/wiki/Likelihood_function" target="_blank">log-likelihood</a>.</p>



# Effect of Varying Parameters

To demonstrate the impact of varying the Dirichlet parameters, the next subplot will only include LDA representations, but with varying values for $\mathbf{\alpha}$ and $\mathbf{\beta}$. Though vectors mathematically, they are input as scalars because Scikit-Learn applies the same value to all topics. To vary the values for particular categories, we would need a more customized approach.

The two charts on the left hold $\beta$ constant and manipulate $\alpha$, and the two charts on the right hold $\alpha$ constant and manipulate $\beta$. The values held constant are equal to $1$ divided by the number of topics, producing uniformly distributed values by category, as is the common default.


<details markdown="1">
  <summary>View Code</summary>

  ```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Corpus with balanced but distinct topics
corpus = [
    "space space car computer",
    "space car computer",
    "car car space computer",
    "car space car",
    "computer computer space car",
    "computer computer computer",
]

# Vectorize the corpus
vectorizer = CountVectorizer(vocabulary=["space", "car", "computer"])
X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

# Define hyperparameter settings
alpha_values = [0.1, 1.0]  # Vary alpha for first two columns
beta_values = [0.1, 1.0]   # Vary beta for last two columns
constant_beta = 1/3       # Constant beta for alpha variation
constant_alpha = 1/3      # Constant alpha for beta variation
n_topics = 3

# Fit LDA models for each hyperparameter setting
lda_models = []
doc_topic_dists = []
for alpha in alpha_values:
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                    doc_topic_prior=alpha, topic_word_prior=constant_beta)
    lda.fit(X)
    lda_models.append(lda)
    doc_topic_dists.append(lda.transform(X))
for beta in beta_values:
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                    doc_topic_prior=constant_alpha, topic_word_prior=beta)
    lda.fit(X)
    lda_models.append(lda)
    doc_topic_dists.append(lda.transform(X))

# Compute topic-word probabilities and topic roles
lda_probs_list = [lda.components_ / lda.components_.sum(axis=1, keepdims=True) for lda in lda_models]
topic_roles_list = [np.max(doc_topic_dist, axis=0) for doc_topic_dist in doc_topic_dists]
lda_alphas_list = [
    0.2 + 0.8 * (roles - np.min(roles)) / (np.max(roles) - np.min(roles) + 1e-10)
    for roles in topic_roles_list
]

# Function to convert probabilities to 2D simplex coordinates
def probs_to_simplex(probs):
    x = probs[:, 1] + 0.5 * probs[:, 2]
    y = np.sqrt(3) / 2 * probs[:, 2]
    return x, y

# Convert probabilities to simplex coordinates
lda_xy_list = [probs_to_simplex(probs) for probs in lda_probs_list]

# Plotting the visualizations (2x4 grid)
fig, axes = plt.subplots(2, 4, figsize=(24, 10))

# Define triangle vertices for simplex plots
triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

# Function to plot simplex triangle
def plot_simplex(ax):
    ax.plot([triangle[0, 0], triangle[1, 0], triangle[2, 0], triangle[0, 0]],
            [triangle[0, 1], triangle[1, 1], triangle[2, 1], triangle[0, 1]], 'k-')
    ax.text(-0.05, -0.05, feature_names[0], fontsize=15, ha='right')
    ax.text(1.05, -0.05, feature_names[1], fontsize=15, ha='left')
    ax.text(0.5, np.sqrt(3)/2 + 0.06, feature_names[2], fontsize=15, ha='center')
    ax.set_aspect('equal')
    ax.axis('off')

# Simplex Plots (Top row)
labels = [
    f"Alpha={alpha_values[0]}, Beta={constant_beta:.2f}",
    f"Alpha={alpha_values[1]}, Beta={constant_beta:.2f}",
    f"Alpha={constant_alpha:.2f}, Beta={beta_values[0]}",
    f"Alpha={constant_alpha:.2f}, Beta={beta_values[1]}"
]
for i, ((lda_x, lda_y), lda_probs, lda_alphas) in enumerate(zip(lda_xy_list, lda_probs_list, lda_alphas_list)):
    plot_simplex(axes[0, i])
    for j, (x, y, alpha) in enumerate(zip(lda_x, lda_y, lda_alphas)):
        axes[0, i].scatter(x, y, facecolors='green', edgecolors='none', s=150, alpha=alpha)
        axes[0, i].scatter(x, y, facecolors='none', edgecolors='darkgreen', s=150, linewidth=2.0)
        axes[0, i].text(x, y + 0.06, f'T{j+1}: ({lda_probs[j][0]:.2f}, {lda_probs[j][1]:.2f}, {lda_probs[j][2]:.2f})', 
                        fontsize=12, ha='center', va='bottom')
    lda_legend = axes[0, i].scatter([], [], facecolors='green', edgecolors='none', s=150, label='Topic Distributions')
    axes[0, i].legend(handles=[lda_legend], fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.05))
    axes[0, i].set_title(labels[i], fontsize=18, pad=35)

# Document-Topic Distribution Plots (Bottom row)
doc_labels = [f'Doc {i+1}' for i in range(len(corpus))]
topic_labels = [f'Topic {i+1}' for i in range(n_topics)]
colors = ['#00FF00', '#00CC00', '#009900']  # Green shades
for i, doc_topic_dist in enumerate(doc_topic_dists):
    bottom = np.zeros(len(corpus))
    for j in range(n_topics):
        axes[1, i].bar(doc_labels, doc_topic_dist[:, j], bottom=bottom, label=topic_labels[j], color=colors[j])
        bottom += doc_topic_dist[:, j]
    axes[1, i].set_xlabel("Documents", fontsize=14)
    axes[1, i].set_ylabel("Topic Proportions")
    axes[1, i].legend()
    axes[1, i].set_title(f"Doc-Topic ({labels[i]})", fontsize=18, pad=35)

# Adjust subplot spacing
plt.subplots_adjust(hspace=0.4)  # Vertical spacing between rows
plt.tight_layout()
plt.savefig('lda_with_params_varied.png', dpi=300, bbox_inches='tight')
plt.show()
  ```
</details> 


<img src="https://raw.githubusercontent.com/pw598/pw598.github.io/main/_posts/images/lda_with_params_varied.png" style="height: 450px; width:auto;">


A notebook providing all the code used above is available <a href="https://github.com/pw598/Articles/blob/main/Topic%20Modeling%20Diagrams.ipynb" target="_blank">here</a>.


# Solving LDA

We haven't discussed the specifics of the algorithms used to solve LDA, and there are several, such as Gibbs sampling, variational inference, and expectation-maximization (EM). The Scikit-Learn model utilizes variational inference. Each of these could warrant an article to themselves, and for the sake of brevity, I will avoid going down those rabbit holes in this article.

There are many variants of LDA, such as Correlated Topic Models (CTM), Hierarchical Dirichlet Process (HDP), and Dynamic Topic Models (DTM). We are also not limited to the bag-of-words method of vectorization, and could use something like TF-IDF or continuous word embeddings. Furthermore, we could take into account word-sequence information, using bi-grams, tri-grams, etc., or treating sentences as sub-documents.



# What's Next?

The next article(s) will focus on practical application. We will:

- Link up to the Reddit API, because it's free of cost and restrictions, and conducive to topic modeling. 
- Use the visualization tool <code>pyLDAviz</code> to produce some cool and interactive visualizations.
- Compare supervised LDA (sLDA) to other methods such as Latent Semantic Indexing (LSI) and Non-Negative Matrix Factorization (NMF).



# References

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research. [https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)

- [LazyProgrammer]. Machine Learning: Natural Language Processing in Python (V2) [Video]. DeepLearningCourses.com. [https://deeplearningcourses.com/c/natural-language-processing-in-python](https://deeplearningcourses.com/c/natural-language-processing-in-python)

- Ruozzi, N. (n.d.). Latent Dirichlet Allocation. UT Dallas. [https://personal.utdallas.edu/~nrr150130/cs6375/2015fa/lects/Lecture_20_LDA.pdf](https://personal.utdallas.edu/~nrr150130/cs6375/2015fa/lects/Lecture_20_LDA.pdf)

- Grok



