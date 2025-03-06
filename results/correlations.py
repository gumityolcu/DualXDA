import torch
from torchmetrics.functional.regression import (
    kendall_rank_corrcoef,
    spearman_corrcoef,
)
import sys 
import argparse
import yaml
import os
import numpy as np
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import logging

matplotlib.interactive(False)

def explainer_corr(xpl_src_dir, save_dir = None, corr="spearman", device="cpu" , num_test_samples=None, color_map = "viridis"):
    tensors=dict()
    names=[]
    for dir in os.listdir(xpl_src_dir):
        if not os.path.isdir(os.path.join(xpl_src_dir,dir)):
            continue
        for f in os.listdir(os.path.join(xpl_src_dir,dir)):
            if "_all" in f:
                basename=f.split("_")[0]
                names.append(basename)
                tensors[basename]=torch.load(os.path.join(xpl_src_dir,dir,f), map_location=device)
                if num_test_samples is not None and num_test_samples<tensors[basename].shape[0]:
                    tensors[basename]=tensors[basename][:num_test_samples]
    corr_fns={"spearman":spearman_corrcoef, "kendall":kendall_rank_corrcoef}

    corr_fn=corr_fns[corr]
    corr_matrix=torch.zeros((len(tensors.keys()),len(tensors.keys())))


    # Use two for loops to create correlations between the explainers
    for i, (name1) in enumerate(names):
        tensor1=tensors[name1]
        for j, (name2) in enumerate(names):
            if i==j:
                corr_matrix[i,j]=1.0
            else:
                tensor2=tensors[name2]
                corr_matrix[i,j]=corr_fn(tensor1.T, tensor2.T).mean()
    fig, ax = plt.subplots(figsize = (8,6))
    im = ax.imshow(corr_matrix, cmap = color_map)

    # Take only the name of the explainer
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    ax.set_xticks(np.arange(len([n.replace("InfluenceFunction","").replace("Explainer","") for n in names])), labels=names)
    ax.set_yticks(np.arange(len(names)), labels=names)

    # Loop over data dimensions and create text annotations.
    for i in range(len(names)):
        for j in range(len(names)):
            text = ax.text(j, i, f"{corr_matrix[i, j]:.3f}",
                    ha="center", va="center", color="w")

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=10)

    plt.title(f'{corr} Matrix Heatmap', fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir,f"{corr}.jpg"))  # Save the image
    torch.save(os.path.join(save_dir,f"{corr}_correlations"))


if __name__ == "__main__":
    prsr=argparse.ArgumentParser()
    prsr.add_argument("--xpl_path", type=str, required=True)
    prsr.add_argument("--device", type=str, required=True)
    prsr.add_argument("--output_path", type=str, required=True)
    prsr.add_argument("--num_test_samples", type=int, default=None, required=False)
    prsr.add_argument("--corr", type=str, choices=["spearman", "kendall"], required=True)
    args=prsr.parse_args()
    explainer_corr(
                   xpl_src_dir=args.xpl_path,
                   save_dir=args.output_path,
                   device=args.device,
                   num_test_samples=args.num_test_samples,
                   corr=args.corr,
                   color_map = "bwr"
                   )