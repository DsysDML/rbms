# Restricted Boltzmann Machines

The RBMs are generative models, part of the Energy Based Models family. They first appeared in 1986 under the name of Harmonium {cite:p}`rumelhart1986parallel`. They were popularized after the introduction of the Contrastive Divergence by Hinton {cite:p}`hinton2002training`, a training algorithm that allowed to train the machine on MNIST (a handwritten digit dataset).

One of the main advantages of the RBMs over other generative models is the simplicity of its structure allowing one to compute conditional probabilities of variables given others and extract multi-body interactions learned by the model {cite:p}`decelle2023inferring`.


## Model
The Restricted Boltzmann Machine (RBM) are defined by $N_v$ visible variables $\pmb s$ (corresponding to the variables of the dataset) and $N_h$ hidden variables $\pmb \tau$ (which encode interactions between variables). 

The distinctive feature of RBMs is their bipartite structure. The variables are organized into a visible layer for the visible variables and a hidden layer for the hidden variables. The visible variables interact only with the hidden variables and vice versa, meaning there is no intra-layer dependency (hence the name "Restricted"). 
These interactions and the local biases of each variable are encoded in the Hamiltonian:

$$
  \mathcal H(\pmb s, \pmb \tau) = -\sum_{ia}s_iw_{ia}\tau_a - \sum_i s_i\theta_i - \sum_a\tau_a\eta_a.
$$

where the parameters of the model are :
 - the weight matrix $\pmb w\in\mathbb R^{N_v\times N_h}$ encoding the interactions between the visible and hidden nodes,
 - the bias vector on the visible units $\pmb \theta\in\mathbb R^{N_v}$,
 - the bias vector on the hidden units $\pmb \eta\in\mathbb R^{N_h}$.

The Boltzmann distribution associated to the RBM is then defined by

$$
  p(\pmb s, \pmb \tau) = \frac{1}{Z}\exp\left(-\beta\mathcal H(\pmb s, \pmb \tau)\right).
$$

where the normalization constant $Z$ is the sum of the exponential term over all possible configurations of visible and hidden units.

$$
  Z = \sum_{\pmb s,\pmb \tau}\exp\left(-\mathcal H(\pmb s, \pmb \tau)\right).
$$

However, computing this term is computationally intractable in practice, as it involves summing over all possible variable configurations, which are exponentially large in number. For example, in the case where all variables are binary, this complexity is dominated by $2^{\min(N_v, N_h)}$.

Another advantageous property of the RBM structure is the ability to compute the conditional distributions of one layer given the other using Bayes' rule:

$$
  p(\pmb \tau |\pmb s) = \frac{p(\pmb s, \pmb \tau)}{p(\pmb s)} = \frac{p(\pmb s, \pmb \tau)}{\sum_{\pmb \tau} p(\pmb s, \pmb \tau)} ,
$$

$$
  p(\pmb s |\pmb \tau) = \frac{p(\pmb s, \pmb \tau)}{\sum_{\pmb s} p(\pmb s, \pmb \tau)}.
$$

Since there is no intra-layer dependency, once the values of one layer have been set, all the neurons on the other layer are independent of each other, leading to simple expressions and permitting a high degree of parallelization in practice. For example, in the case of the Bernoulli-Bernoulli RBM (where the prior distributions on the visible and hidden variables are Bernoulli distributions), the conditional distributions are given by:

$$
  p(s_i = 1|\pmb \tau) = \text{sigmoid}(\theta_i + \sum_a w_{ia}\tau_a),
$$

$$
  p(\tau_a=1|\pmb s) = \text{sigmoid}(\eta_a + \sum_i w_{ia}s_i),
$$

where the sigmoid function is defined on $\mathbb R$ as

$$
  \text{sigmoid}(x) = \frac{1}{1+\exp(-x)}.
$$

## Log-likelihood and gradient update 

We train the model by maximizing the log-likelihood of the samples given the model marginalized over the visible variables. The log-likelihood is given by:

$$
\log \mathcal{L}(\pmb{s}_\text{data}; \pmb{w}, \pmb{\theta}, \pmb{\eta}) = \frac{1}{N_s} \sum_{d=1}^{N_s} \log\left[\sum_{\pmb{\tau}}  \exp(-\mathcal{H}(\pmb{s}^{(d)}_\text{data}, \pmb{\tau}))\right] - \log(Z).
$$

We can then explicitly compute the gradient over the parameters of the RBM:

$$
\frac{\partial \log \mathcal{L}}{\partial w_{ia}} =  \left\langle s_i\tau_a\right\rangle_\text{data} - \left\langle s_i\tau_a\right\rangle_{\mathcal H}
$$

where $\left\langle\cdot \right\rangle_\text{data}$ denotes the average under the empirical distribution and $\left\langle\cdot \right\rangle_{\mathcal H}$ the average under the model distribution.
The gradient components associated with the biases are:

$$
\frac{\partial \log \mathcal{L}}{\partial \theta_i} = \left\langle s_i\right\rangle_{\text{data}} - \left\langle s_i\right\rangle_{\mathcal{H}},
$$

$$
\frac{\partial \log \mathcal{L}}{\partial \eta_a} = \left\langle \tau_a\right\rangle_{\text{data}} - \left\langle \tau_a\right\rangle_{\mathcal{H}}.
$$

Each of these gradients is divided into two parts, often referred to as the $\textit{positive}$ and $\textit{negative}$ terms. The positive term is easy to compute since it only relies on sampling the conditional distributions of the hidden variables given a sample from the dataset. In contrast, the negative term depends on the model distribution, which is generally intractable due to the partition function. This term is estimated in practice using configurations sampled from the model distribution.

Once the gradient is computed, the parameters are updated the gradient ascent rule. 

$$
    w_{ia}^{(t+1)} = w_{ia}^{(t)} + \gamma\frac{\partial \mathcal L}{\partial w_{ia}}
$$

$$
    \theta_i^{(t+1)} = \theta_i^{(t)} + \gamma \frac{\partial \mathcal L}{\partial \theta_{i}}
$$

$$
    \eta_a^{(t+1)} = \eta_a^{(t)} + \gamma \frac{\partial \mathcal L}{\partial \eta_{a}}
$$

where $\gamma$ is the learning rate. 

## Sampling
Since the normalization constant $Z$ is in practice impossible to compute, we cannot directly sample the Gibbs-Boltzmann distribution of the RBM. Instead, we rely on Monte Carlo Markov Chains (MCMC) simulations. The bipartite structure of the model makes it a perfect candidate for [Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling).

One challenge for MCMC methods is to ensure the process has iterated long enough, so that the process has reached the equilibrium measure and starts to sample the Gibbs-Boltzmann distribution of the RBM. In other words, we want to ensure the chains length is long enough compared to the mixing time. There is no simple way to determine this time. Therefore, one generally sets a fixed number of steps and hopes that it is sufficient to estimate the gradient correctly. Otherwise, memory effects may appear in the trained model, where the machine learns to reproduce the statistics of the dataset at a fixed number of steps rather than at the level of the Boltzmann measure {cite:p}`agoritsas2023explaining,decelle2021equilibrium`.

A major drawback of Gibbs sampling is that the mixing time can become prohibitive if the distribution from which one wishes to sample from is highly clustered and has large regions with low probability between modes. In this case, the probability of jumping from one cluster to another is essentially
zero, which makes the mixing time extremely long.

Since we want to sample the model every time the parameters are updated, it is important to find a way to speed up the convergence to equilibrium of the sampling methods. To this end, one possibility is choosing good initializations for the chains:

The first method is \emph{Contrastive Divergence (CD)}, introduced by {cite:p}`hinton2002training`. The method chosen is to initialize the MCMC chains with random samples from the dataset and perform some steps (typically $\mathcal O(1)$). This was chosen because initializing the model with configurations close to its distribution should lead to fast convergence if the model is well trained. However, this method performs poorly, since the distribution of the model at the beginning of training is very far from the empirical distribution of the dataset. In other words, this method does not provide an accurate estimate of the negative term of the gradient.

An improvement of this method was presented in {cite:p}`tieleman2008training` as \emph{Persistent Contrastive Divergence (PCD)}. Here, the chains are randomly initialized only at the beginning of the training. After each parameter update, the chains are not reinitialized, but the previous final configuration is reused as the starting point. These are referred to as persistent chains. The underlying assumption is that the distribution of the model at time $t$ should not be too different from the distribution at time $t-1$ if the parameter update is small enough. The chains are expected to remain in equilibrium if the model is slow enough. This method allows obtaining near-equilibrium models, but at the cost of prohibitive generation times after training.


Alternative sampling methods to improve convergence have been exploited for the case of highly clustered dataset was introduced in {cite:p}`swendsen1986replica, earl2005parallel` as \emph{Parallel Tempering (PT)}. In this method, one samples several replicas of the model at different temperatures and exchanges MCMC chains between replicas of the model $k$ and $l$ with an acceptance ratio

$$
    A = \min \left(1, \exp\left[\beta_k\mathcal H(\pmb s^{(k)}, \pmb \tau^{(k)})\right]/\exp\left[\beta_l\mathcal H(\pmb s^{(l)}, \pmb \tau^{(l)})\right]\right),
$$

where $\beta_k$ and $\beta_l$ are the inverse temperatures of replicas $k$ and $l$ and $(s^{(k)},\tau^{(k)})$ and $(s^{(k)},\tau^{(k)})$ are variables in the chains from replicas $k$ and $l$.The intuition is that at higher temperature, the replicas will have less trouble escaping valleys because the modes are flattened, but the sampling will not be very accurate, whereas at lower temperature the replicas will have trouble jumping between clusters but will be able to sample accurately the modes that the chains are in. So if you allow the two replicas to swap at different temperatures, you can exploit the faster mixing at high temperature to mix faster the low temperature chains. This method however performs poorly on highly clustered distributions.


```{bibliography} references.bib
```