# TODOs

### Additional Models (Galip)

- [ ] Global (w.r.t models): TRAK
- [ ] Local (w.r.t. models): TracIn &rarr; needs retraining
- [ ] Updated IF approximation (EK-FAC approximation of Grosse et al.)
- [ ] Potentially others?

### Datasets (not distributed yet)

- [ ] Animals with attributes (can also be used for concepts)
- [ ] look for more (Galip + Mo)

### Metrics (Mo)

- [ ] Randomisation Sanity Test
- [ ] Check literature which other metrics are used
- [ ] Come up with new metrics
- [ ] Implement them

### Feature Attributions (stash)

- [ ] Heatmaps currently don't change for deep models
- [ ] Heatmap for all other kernel surrogates?
- [ ] Reformulate non-kernel surrogates as kernel surrogates if possible, or find other LRP rules

### Additionally (include now?)

- [ ] Experiment using only support vectors as training data and see how accuracy is reduced

### Additionally (next paper?)

- [ ] Relevance Vector Machine (Mo)
    - Paper: https://proceedings.neurips.cc/paper_files/paper/1999/file/f3144cefe89a60d6a1afaf7859c5076b-Paper.pdf
    - Package: https://github.com/JamesRitchie/scikit-rvm/tree/master?tab=readme-ov-file
- [ ] BiLRP (Read: Galip + Mo)
- [ ] Concepts (stash)
- [ ] Check how close surrogate models are to original model (Mo)
- [ ] Parallelisation? (Read: Galip + Mo)
- [ ] Make more clear how the algorithm works and what the correct way to interpret the relevances are (e.g. how do they relate to the support vectors? Why are support vectors "relevant"?) (Read on SVMs: Mo)
- [ ] Explain in more detail the difference to RP (Read paper: Mo)
- [ ] Theoretically motivate the surrogate model (e.g. a soft-margin linear SVM can be rephrased as a retraining of the final layer with a hinge loss and weight decay. Maybe we can not only empirically show that the result of the SVM is close to the weight vector of the original final layer, but also show theoretically that this retraining of the final layer will stay close to the original solution? Depending on how the solution for the SVM is implemented e.g. with a Stochastic Gradient Descent, we could also use as an initialisation the original weight vector form the last layer, maybe then the SVM will stick more closely to the orginial final layer) (Mo)