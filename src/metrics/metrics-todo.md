# Metrics - TODO

## Currently implemented

- **Identical Class Test (Retrainining: None)**
    - The most positively relevant training data point for the decision for class $c$ should be labeled as class $c$. Record the ratio of all test points for which this holds.
- **Identical Subclass Test (Retraining: Once)**
    - Create superclasses $C$ as the union of multiple (sub-)classes. If the test point $x$ has true label $c$ with subclass $c$ belonging to superclass $C$, then the most positively relevant traning data point for the decision for superclass $C$ should be labeled as class $c$ (and not one of the other subclasses belonging to $C$). Record the ratio of all test points for which this holds.
- **Label Poison Test (Retraining: Once)**
    - Poison the label of a random subset of the training data. Sort all training data points by relevance (either the relevance on the entire training set or by the self-relevance $V_i(x_i)$), go through the list from high to low relevance and record the proportion of poisoned training data points. This gives a cumulative density curve and a better explainer should have more poisoned training data points with high relevance, thus the curve should increase more rapidly. Compute the area under the curve and normalise the metric to be between 0 and 1.
- **Domain Mismatch Test (Retraining: Once)**
    - Take a fixed perturbation in the input space $\omega$ and apply it to a random subset of the training data points of only one class $c$, i.e. replace training data points $(x_i,c)$ with $(\omega(x_i), c)$. Retraining on the new partly perturbed training data set introduces a correlation between class $c$ and perturbation $\omega$. For the test set, discard all datapoints of class $c$ and apply the perturbation $\omega$ to all remaining test points. Now for every test point that was classified to class $c$, the most positively relevant training data point for this decision should be a perturbed training data point of class $c$. Record the ratio of all test points classified as class $c$ for which this holds.

## What to implement

- [x] Top-k (instead of only top-1) version of Identical Class and Subclass test
    - [ ] Could find a way to change $k$ from config file and not hardcode

## Ideas

### No Retraining
- **Self-relevance test (Retraining: None)**
    - A training data point should have high relevance for itself (maybe not?) but at least low (ie high negative) relevance for predicting a class label ${y_{train}}'$ which is different from the label $y_{train}$ in the dataset.
    - Think more about whether this makes sense 
    - Idea by Moritz
    - At least mislabeled training data points should have high negative self-attribution (https://www.youtube.com/watch?v=unDA9yPjG68 at 15:55)
- **Input sanity test (Retraining: None)**
    - Calculated relevances should depend on the test point, similar test point should have similar relevances, very different test points should have different relevances
    - Idea by Moritz

### $O(1)$ Retraining
- **Targeted Poison Test (Retraining: Once)**
    - Add a backdoor poison to some training data points from class $c$ and relabel them as $c'$. The most relevant examples for perturbed test points for class decision $c'$ should now be the perturbed training sets.
    - How is this exactly different from Domain Mismatch Test?
    - Idea from https://arxiv.org/pdf/2111.04683.pdf
- **Bayesian Switch Test (Retraining: Once but only partially)**
    - Need to fix notation. High relevance of $(x_{train}, y_{train})$ for the test point $x_{test}$ and class label decision $y_{test}$ should imply a high value for the conditional probability
    \[ p(h(x_{test})=y_{test} \mid h(x_{train})=y_{train})=\frac{p(h(x_{test})=y_{test})}{p(h(x_{train})=y_{train})}p(h(x_{train})=y_{train} \mid  h(x_{test})=y_{test}),\]
    where the unconditional probabilities equal the prior probabilities and thus the ratio of training data points with label $y_{train}$ and $y_{test}$ in the training data set (which for a balanced dataset cancels out). Therefore this should be associated with a high relevance of the *test* data point $(x_{test}, y_{test})$ for the *training* data point $x_{train}$ and class label decision $y_{train}$.
    Retrain (or use incremental training), adding $(x_{test}, y_{test})$ to the training set. Check the relevances for $x_{train}$ and class label $y_{train}$.
    - Idea by Moritz
- **Class Split Test (Retraining: Once)**
    - Split up a class $c$ into two classes $c_1$ and $c_2$ and randomly allocate half of the training data points of class $c$ to be class $c_1$ and the other half to be class $c_2$, making them undistinguishable. Take a test point previously (clearly) allocated to class $c$. The most positively and negatively relevant training data points for decision of label $c_1$ and label $c_2$ should now all be of class $c_1$ or $c_2$.
    - Idea by Moritz
- **Randomisation Sanity Check**
    - For a correctly trained model and a ranodmly initialised model, the ranks of relevances should have small correlation
    - This only works for non-training based models (e.g. not TracIn)...
    - Idea from https://arxiv.org/pdf/2006.04528.pdf (but really Adebayo et al.)
    - also look into Leander and Anna's version of this?

### $O(Training\ data)$ retraining
- **Leave-One-Out Test (Retraining: Multiple)**
    - Sort training data points by relevance for decision of labeling a test point $x$ with class label $c$. Go through training points $x_i$ and retrain model on all training points but $x_i$ and record the class $c$ logit for test point $x$. The resulting list should be monotonously increasing (highest relevance -> removing gives lowest logit), count the upsets.
    - Obviously computationally problematic because needs a lot of retraining. Could take a random subset, though probably still expensive.
    - Idea by Moritz (but also everyone)
- **Add-One-In Test (Retraining: Multiple but only partially)**
    - Retrain your model on the original training data set but with one of the training points added in twice. Addition of the most relevant training data points should be associated with the highest increase in logits.
    - Idea by Moritz
- **Label Flip Test (Retraining: Multiple)**
    - Flipping the label of highly relevant data points should decrease logits after retraining, flipping the label of highly negatively relevant data points (to the desired label) should increase the logits
    - Idea by Moritz
- **Check Std (Retraining: Multiple)**
    - Check standard deviations of relevances over multiple runs (e.g. with random batch order)?
    - akin to the Bayesian paper
- **Cumulative and individual add Batch-In Tests (Retraining: Multiple)**
    - Cumulative: Add 10% of data, record prediction probability for one data point, then 20% and so on
    - Cumulative: Do this once in the order of most relevant to least relevant and compare to random (or to least relevant to most relevant) and measure area between curves
    - Individual: Use 10% of data, record prediction probability for one data point, then next 20% and so on
    - Idea by https://www.scholar-inbox.com/papers/Wu2024ARXIV_On_the_Faithfulness_of.pdf, but for training data
- **Individual Leave-Batch-Out Tests (Retraining: Multiple)**
    - Leave first 10% of data out, record prediction probability for one data point, then the next 10%
    - Do this in order of relevance
    - Leave-One-Out is just Leave-Batch-Out with batch size 1

### Higher order retraining
- **Average Marginal Effect** 
    - Similar to Shapley, but as metric
    - $AME_{x_n} = \Sigma_{S \subset [N] \setminus x_n} \frac{1}{N}{ {N-1}\choose{|S|}}^{-1}(U(S \cup x_n)-U(S))$, where $U$ is some utility function (e.g. probability prediciton of correct class for a test data point)
    - Can also have non uniform sampling to not get Shapley value
    - Can be estimated in $O(N \log \log N)$ or $O(k \log N)$ if only $k$ data points have non-zero AME
    - Idea by https://proceedings.mlr.press/v162/lin22h/lin22h.pdf

### Unclear
- **Randomisation Test**
    - Perturbing the parameters of the model would work for DualView, but not methods which track the training process (e.g. TracIn), or for model-agnostic attribution methods
    - Idea from https://arxiv.org/pdf/2401.06465.pdf, needs to be adopted to training data attribution (by Moritz)

- **AOPC**
    - Idea by https://aclanthology.org/2020.acl-main.494.pdf



- Also watch this: https://www.youtube.com/watch?v=unDA9yPjG68
- And check what methods exist for feature attribution (e.g. in MetaQuantus), I asumme a lot of them can be translated to data attribution

- **Failure modes from MetaQuantus**
    - https://arxiv.org/pdf/2302.07265.pdf
    - Noise Resilience: A quality estimator should be resilient to minor perturbations of its input parameters.
        - Could here also be minor permutations of the training data?
    - A quality estimator should be reactive to disruptive perturbations of its input parameters.
    - Go through Table 1 and adapt the experiments to Data Attribution.
    - Sparseness: sparseness in number of training data points
    - Complexity: 
    - Faithfulness correlation:
    - Pixel-flipping: Label-flipping or cumulative leave-out
    - Pointing-Game: Identical class/subclass test
    - Relevance Mass Acc.: ??? (Ground-truth mask?)
    - Random Logit:
    - Model Parameter Randomisation:
    - Max-Sensitivity:
    - Local Lipschitz Estimation:
    - Top-K Intersection: Identical class/subclass test with top K instead of top attribution? (Is that still a reasonable ground truth mask?)
    - Relevance Rank Accuracy: ??? Ground-truth mask?