This repository implements the experiments for the paper DualView: **Runtime-Efficient and Sparse Training Data Attribution**

DualView provides a runtime-efficient method for Training Data Attribution that is also sparse by construction. In our paper, DualView has shown competitive performance against state-of-the-art methods while requiring orders of magnitude less computational time and memory.

<img src="img/runtime_vs_rank.png" alt="Average ranks plotted against total explanation time over seven evaluation metric for DualView and prominent approaches from the literature." width="800"/>

We further introduce DuaLRP, a novel method to obtain feature maps, which indicate why a certain training point is relevant for the prediction on a test point.

<img src="img/dualrp_overview_with_explanation_flattened.jpg" alt="DualLRP provides heatmaps on test-train sample pairs that indicate _why_ a certain training image is influential for the prediction on the test sample." width="800"/>


This repository contains the code used to generate and evaluate post-hoc local data attribution of torch models using seven evaluation criteria and against eight explanation methods. These include three different approximations of Influence Functions, TRAK, TracIn and Representer Point Selection. [`sklearn`](https://scikit-learn.org/) has been used for DualView explanations.

# Getting Started
To start explaining predictions with DualView, you first need to install the modified `sklearn` library by running

```bash
pip install ./scikit-learn-dual
```

Then, given a classification dataset `train` and a corresponding `model`, you can start attributing the model predictions on a test dataset `test` with DualView:
```python
from explainers.DualView

C = 0.001
device = "cuda"
cache_dir = "<cache_dir_for_dualview>"
features_dir = "<cache_dir_for_features>"
explainer = DualView(
    model,
    train,
    device=device,
    dir=cache_dir,
    features_dir=features_dir,
    C=C)
ldr = torch.utils.data.DataLoader(test, batch_size=True, shuffle=False)
for (x, y) in ldr:
    preds = model(x).argmax(dim=-1)
    attributions = explainer.explain(x, preds)

```

# Evaluation of TDA methods
In this section, we explain how to reproduce the results in our paper.

The evaluation is conducted in five steps:
1. Generate configuration files for the following steps
2. Train or download checkpoints 
3. Generate explainer caches
4. Generate explanations

These steps are done four times, using different modifications of the same datasets. These correspond to the following keywords in our code:
"std" for vanilla dataset, "group" for modified dataset with superclasses, "corrupt" for a dataset with corrupt labels and "mark" for a dataset with shortcut features.

Finally, we have the last step
5. Evaluate the explanations with metrics

which uses the correct dataset type and reads the correspoinding explanations for each metric.

## 1. Generating configuration files

All of our scripts work with .yaml files that specify the details of the required computation. These are generated with the ipynb files in the `config_files` directory. 

Open each notebook and replace the `<data_root>` placeholder with the directory where you keep the associated datasets. You can then run the notebook to generate configuration files.

## 2. Download checkpoints

To reproduce our results exactly, you are advised to download the pretrained checkpoints for MNIST, CIFAR-10 and AwA2 datasets with the following link:

** URL will be provided for the camera ready version. Unfortunately, the checkpoints are too big to host in an anonymous fashion. **

Alternatively, you can train new models using the training configuration files:

```bash
    cd src
    python train.py --config_file config_files/train/< dataset_name >/< configuration_file_name >
```
## 3. Generate Caches for Explainers
Except for LiSSA, our explainers require a cache. However, note that computing the cache for the TracIn method will compute the caches required for GradDot and GradCos. Finally, we use the `explain.py` script **with the caching configurations** to compute caches:

To start computing the caches, run

```bash
    cd src
    python explain.py --config_file config_files/cache/< dataset_name >/< configuration_file_name >
```

## 4. Generate Explanations
Once you have the checkpoints and required caches, you can start generating training data attributions:

```bash
    cd src
    python explain.py --config_file config_files/explain/< dataset_name >/< configuration_file_name >
```

Note that running the `explain.py` script for the `corrupt` dataset/model will compute self-attributions of training points, instead of attributing the test data predictions.

## 5. Evaluate Explanation Quality
Evaluation is done using the evaluate.py script and the results will be stored in the *results* folder.

```bash
    cd src
    python explain.py --config_file config_files/evaluate/< dataset_name >/< configuration_file_name >
```