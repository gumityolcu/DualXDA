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


def group_files_by_base_name(src_dir):
    grouped_files = defaultdict(list) # Create dict for the explainers

    # Walk through all directories and subdirectories in the source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # Check if 'explainer' is in the filename
            if 'explainer' in file.lower():
                # Split the filename at the underscore to get the base name
                base_name = file.split('_')[0]
                # Get the numerical part of the filename
                file_number = int(file.split('_')[1].split('.')[0])
                file_path = os.path.join(root, file)
                grouped_files[base_name].append((file_number, file_path)) # Create a tuple with file number and the file path
            

    # Sort the file paths numerically for each base name
    for base_name in grouped_files:
        grouped_files[base_name].sort() #Sorts the files based on the tuple's first element, which is the file number
        grouped_files[base_name] = [file_path for _, file_path in grouped_files[base_name]] # Remove the numerical part
            
    return grouped_files

# Load the tensors
def load_tensors(file_paths):
    tensors = []
    for file_path in file_paths:
        tensor = torch.load(file_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        tensors.append(tensor)
    return torch.cat(tensors,dim=0)

def explainer_corr(xpl_src_dir, corr_types = "spearman", save_dir = None, num_rows = None , color_map = "viridis"):
    
    # Initialize the correlation matrix
    if not isinstance(corr_types, list):
        corr_types = [corr_types] 
    corr_fns=[spearman_corrcoef if corr_t == "spearman" else kendall_rank_corrcoef for corr_t in corr_types ]

    # Group files by their base names
    grouped_files = group_files_by_base_name(xpl_src_dir)

    # Load tensors from the grouped files
    grouped_tensors = {base_name: load_tensors(file_list) for base_name, file_list in grouped_files.items()}

    min_num_of_test_samples = min([tensor.shape[0] for base_name, tensor in grouped_tensors.items()])
    
    num_rows=min_num_of_test_samples if num_rows is None else num_rows
    if num_rows<min_num_of_test_samples:
        num_rows=min_num_of_test_samples
        print("IGNORING num_rows BECAUSE IT IS BIGGER THAN MINIMUM TEST SAMPLE SIZE OF EXPLAINERS")
    

     
    corr_matrix_names = list(grouped_tensors.keys()) # Copy the base name list to correlation matrix names

    corr_matrices=[]
    for c, corr_fn in enumerate(corr_fns):
        corr_matrix=[]
        # Use two for loops to create correlations between the explainers
        for _, tensor1 in grouped_tensors.items():
            corr_row=[]
            for _, tensor2 in grouped_tensors.items():
                corr_output=corr_fn(tensor1.T, tensor2.T).mean()
                corr_row.append(corr_output.item())
            corr_matrix.append(corr_row)
        corr_matrix=torch.tensor(corr_matrix)
        corr_matrices.append(corr_matrix)
        if save_dir is not None:
            fig, ax = plt.subplots(figsize = (8,6))
            im = ax.imshow(corr_matrix, cmap = color_map)

            # Take only the name of the explainer
            rows = []
            columns = []
            for i in range(len(corr_matrix_names)):
                rows.append(corr_matrix_names[i].split("Explainer")[0])
                columns.append(corr_matrix_names[i].split("Explainer")[0])

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

            ax.set_xticks(np.arange(len(rows)), labels=rows)
            ax.set_yticks(np.arange(len(columns)), labels=columns)

            # Loop over data dimensions and create text annotations.
            for i in range(len(rows)):
                for j in range(len(columns)):
                    text = ax.text(j, i, f"{corr_matrix[i, j]:.3f}",
                            ha="center", va="center", color="w")

            cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=10)

            plt.title('SpearmanCorrCoef Matrix Heatmap', fontsize=14)
            fig.tight_layout()
            plt.savefig(os.path.join(save_dir,f"{corr_types[c]}.jpg"))  # Save the image
        else:
            plt.show()

    return corr_matrices[0] if len(corr_matrices)==1 else corr_matrices


if __name__ == "__main__":
    current = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(current)
    sys.path.append(current)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, "r") as stream:
       try:
           train_config = yaml.safe_load(stream)
       except yaml.YAMLError as exc:
           logging.info(exc)

    save_dir = f"{train_config['save_dir']}/{os.path.basename(config_file)[:-5]}"

    explainer_corr(xpl_src_dir=train_config["xpl_src_dir"],
                   corr_types = train_config["corr_types"],
                   save_dir = train_config["save_dir"],
                   num_rows = None , color_map = "viridis")