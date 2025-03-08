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
        xpl_root=os.path.join(xpl_src_dir,dir)

        if not os.path.isdir(xpl_root):
            continue
        file_list = [f for f in os.listdir(xpl_root) if ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and ("_tensor" not in f) and (".shark" not in f) and (".times" not in f)]
        file_root = file_list[0].split('_')[0]
        num_files=len(file_list)
        if os.path.isfile(os.path.join(xpl_root, f"{file_root}_all")):
            xpl_all = torch.load(os.path.join(xpl_root, f"{file_root}_all"))
        #merge all xpl
        else:
            xpl_all = torch.empty(0, device=device)
            for i in range(num_files):
                fname = os.path.join(xpl_root, f"{file_root}_{i:02d}")
                xpl = torch.load(fname, map_location=torch.device(device))
                xpl.to(device)
                xpl_all = torch.cat((xpl_all, xpl), 0)
            torch.save(xpl_all, os.path.join(xpl_root, f"{file_root}_all"))
        basename=dir
        names.append(basename)
        tensors[basename]=xpl_all
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

    ax.set_xticks(np.arange(len(names)), labels=names)
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
    torch.save(corr_matrix, os.path.join(save_dir,f"{corr}_correlations"))
    with open(os.path.join(save_dir,"order_of_explainers"), "w") as f:
        for i,n in enumerate(names):
            f.write(f"{n}")
            if i<len(names)-1:
                f.write("\n")
    

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