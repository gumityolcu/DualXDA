import torch
import os

import pandas as pd

path="/mnt/"

def l0_eps(xpl, eps=1e-6):
    # returns ratio of points whose influence (relative to the maximum influence) is smaller than eps
    xpl_norm = xpl / xpl.max(dim=1).values.unsqueeze(1)
    return (xpl_norm.abs() < eps).float().sum(dim=1).mean().item() / xpl.shape[1]

def hoyer_measure(xpl):
    """
    Calculate the Hoyer measure of sparsity for each row in a tensor.
    
    The Hoyer measure is defined as:
        (sqrt(n) - L1/L2) / (sqrt(n) - 1)
    
    where n is the dimension, L1 is the L1 norm, and L2 is the L2 norm.
    The measure ranges from 0 (completely dense) to 1 (completely sparse).
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [m, n]
        
    Returns:
        torch.Tensor: Tensor of shape [m] containing the Hoyer measure for each row
    """
    # Get the dimension of each row
    n = xpl.shape[1]
    
    # Calculate L1 norm for each row (sum of absolute values)
    l1_norm = torch.norm(xpl, p=1, dim=1)
    
    # Calculate L2 norm for each row (Euclidean norm)
    l2_norm = torch.norm(xpl, p=2, dim=1)
    
    # Handle zero vectors to avoid division by zero
    # When L2 norm is 0, the vector is all zeros, so sparsity should be 1
    mask = l2_norm == 0
    
    # Calculate Hoyer measure
    sqrt_n = torch.sqrt(torch.tensor(n, dtype=xpl.dtype))
    hoyer = torch.zeros_like(l1_norm)
    
    # Apply formula only for non-zero vectors
    non_zero_indices = ~mask
    hoyer[non_zero_indices] = (sqrt_n - l1_norm[non_zero_indices] / l2_norm[non_zero_indices]) / (sqrt_n - 1)
    
    # Set measure to 1 for all-zero vectors
    hoyer[mask] = 1.0
    
    return hoyer.mean().item()

def find_xpl_tensor(xpl_root):
    device= "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir(xpl_root):
        return None
    file_list = [f for f in os.listdir(xpl_root) if ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and ("_tensor" not in f) and (".shark" not in f) and (".times" not in f)]
    if len(file_list)==0:
        return None
    file_root = file_list[0].split('_')[0]
    num_files=len(file_list)
    xpl_all_path=os.path.join(xpl_root, f"{file_root}_all")
    if os.path.isfile(xpl_all_path):
        return xpl_all_path
    else:
        xpl_all = torch.empty(0, device=device)
        for i in range(num_files):
            fname = os.path.join(xpl_root, f"{file_root}_{i:02d}")
            xpl = torch.load(fname, map_location=torch.device(device))
            xpl.to(device)
            xpl_all = torch.cat((xpl_all, xpl), 0)
        torch.save(xpl_all, xpl_all_path)
    return xpl_all_path

xai_methods = ["input_similarity_dot", "input_similarity_cos", "input_similarity_l2",
               "feature_similarity_dot", "feature_similarity_cos", "feature_similarity_l2",
               "lissa", "arnoldi", "kronfluence",
               "tracin",
               "trak",
               "graddot",
               "gradcos",
               "representer",
               #"dualview_1e_06",
               "dualview_1e_05",
               #"dualview_0.0001",
               "dualview_0.001",
               #"dualview_0.01",
               "dualview_0.1",]

index_column=[
    "\\multirow{3}{*}{Input Similarity} & Dot",
    "& CosSim",
    "& $\\ell_2$",
    "\\hline\n\\multirow{3}{*}{Feature Similarity} & Dot",
    "& CosSim",
    "& $\\ell_2$",
    "\\hline\n\\multirow{3}{*}{Influence Function} & LiSSA",
    "& Arnoldi",
    "& EK-FAC",
    "\\hline\nTracIn &",
    "\\hline\nTRAK &",
    "\\hline\nGradDot &",
    "\\hline\nGradCos &",
    "\\hline\nRepresenter Points &",
    "\\hline\n\\multirow{3}{*}{DualView (ours)} & $C=10^{-5}$",
    "& $C=10^{-3}$",
    "& $C=10^{-1}$",
    ]

print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{cl|ll|ll|ll}")
print("\\toprule")
print("&& \\multicolumn{2}{c|}{MNIST} & \\multicolumn{2}{c|}{CIFAR} & \\multicolumn{2}{c}{AWA}\\\\")
print("&& $\\ell_0^{\\varepsilon}$ & Hoyer & $\\ell_0^{\\varepsilon}$ & Hoyer & $\\ell_0^{\\varepsilon}$ & Hoyer\\\\")
print("\\hline")
print("\\hline")

lines = []
ds_type='std'

for i, name in enumerate(xai_methods):
    xai_method = xai_methods[i]
    line = index_column[i]

    for ds in ['MNIST', 'CIFAR', 'AWA']:
        xpl_dir=os.path.join(path, "explanations", ds, ds_type, xai_method)
        xpl_path=find_xpl_tensor(xpl_dir)
        if xpl_path is not None:
            xpl=torch.load(xpl_path, map_location=torch.device('cpu'))
            l0=l0_eps(xpl)
            hoyer=hoyer_measure(xpl)
            line = line + f' & {l0:.3f} & {hoyer:.3f}'
        else:
            line = line + ' & - & -'
    line = line + ' \\\\'
    lines.append(line)

for l in lines:
    print(l)

print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Sparsity metrics for all data attribution explainers}")
print("\\label{tab:sparsity}")
print("\\end{table}")