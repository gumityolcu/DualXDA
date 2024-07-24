import torch
from torchmetrics.regression import SpearmanCorrCoef
from torchmetrics.regression import KendallRankCorrCoef
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

src_dir = r'C:\Users\akbulut\Desktop\Deniz\Correlations'
save_path = r'C:\Users\akbulut\Desktop\Deniz\deneme'

def explainer_corr(src_dir, save_dir = True, save_image_to = None, spearman_corr = True, kendall_corr = True, num_row = None, color_map = "viridis"):
    
    # Initialize the correlation matrix
    spearman_corr_matrix = None
    kendall_corr_matrix = None 

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
        return tensors

    # Group files by their base names
    grouped_files = group_files_by_base_name(src_dir)

    # Load tensors from the grouped files
    grouped_tensors = {base_name: load_tensors(file_list) for base_name, file_list in grouped_files.items()}

    # Copy the grouped tensors to compare with concatenation operation
    compare_tensors = grouped_tensors.copy()

    # Print the explainer names
    for base_name, tensors in grouped_tensors.items():
        print(f"This explainer was found: {base_name}")

    min_arg = [] # Create list to find minimum row value in the explainer tensors
    max_arg_row = []
    # Concatenate the tensors
    for base_name, tensors in grouped_tensors.items(): # Tensor list
        grouped_tensors[str(base_name)] = torch.cat(grouped_tensors[str(base_name)], axis = 0)
        min_arg.append(grouped_tensors[str(base_name)].shape[0])
        max_arg_row.append(grouped_tensors[str(base_name)].shape[1]) # Take the minimum number of rows
   
    min_value = min(min_arg) # Take the minimum number of columns for correlations
    max_arg_row_val = min(max_arg_row)
    if num_row == None:
        num_row = max_arg_row_val

    # Compare the tensors and Concatanation Operation
    compare_list = []
    for base_name, tensors in grouped_tensors.items():
        gitfor = 0
        compare_list = []
        k = 0
        print(base_name)
        for i in range(len(compare_tensors[str(base_name)])):
            lenght = gitfor
            gitfor += compare_tensors[str(base_name)][k].shape[0]
            k += 1
            result = torch.eq(compare_tensors[str(base_name)][i], grouped_tensors[str(base_name)][lenght:gitfor, :])
            are_equal = result.all().item()
            compare_list.append(are_equal)

        print(set(compare_list))

    # Create the correlations
    if spearman_corr:

        base_name_list = [] # Initialize the base name list
        corr_matrix = []    # Initialize the correlation matrix list
        corr_dict = {}      # Initialize the correlation matrix dictionary
        corr_values = []    # Initialize the correlation values list
        spearman = SpearmanCorrCoef()

        for base_name, tensors in grouped_tensors.items():
            base_name_list.append(base_name) # Take the base names and fill the list

        corr_matrix_names = base_name_list.copy() # Copy the base name list to correlation matrix names

        # Use two for loops to create correlations between the explainers
        for base_name, tensors in grouped_tensors.items():
            for k in base_name_list:
                #print(f"{base_name} --- {k}")
                for i in range(num_row): # Compare the user defined row numbers
                    target = grouped_tensors[str(base_name)][i, :min_value]
                    preds = grouped_tensors[str(k)][i, :min_value]
                    corr_matrix.append(spearman(preds.T, target.T))
                    
                corr_matrix = np.array(corr_matrix, dtype = np.float32) # Turn the correlation list to numpy array
                mean = corr_matrix.mean() # Take the mean of the correlations
                corr_values.append(mean) 
                #print(mean)
                corr_matrix = [] # Initialize the correlation matrix list for the next loop
    
            corr_dict[base_name] = corr_values
            corr_values = [] # Initialize the correlation values list for the next loop
    

        matrix = []
        for explainers, values in corr_dict.items():
            matrix.append(values)
    
    
        matrix = np.array(matrix)
        print("----Spearman Correlation Matrix----")
        print(matrix) 

        if save_dir == True:

            fig, ax = plt.subplots(figsize = (8,6))
            im = ax.imshow(matrix, cmap = color_map)

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
                    text = ax.text(j, i, f"{matrix[i, j]:.3f}",
                            ha="center", va="center", color="w")

            cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=10)

            plt.title('SpearmanCorrCoef Matrix Heatmap', fontsize=14)
            fig.tight_layout()
            if save_image_to is None:
                plt.savefig("SpearmanCorrCoef.jpg")
            else:
                plt.savefig(f"{save_image_to}/SpearmanCorrCoef.jpg")  # Save the image
            

    if spearman_corr:
        spearman_corr_matrix = np.copy(matrix)


    if kendall_corr:

        base_name_list = []
        corr_matrix = []
        corr_dict = {}
        corr_values = []
        kendall = KendallRankCorrCoef()

        for base_name, tensors in grouped_tensors.items():
            base_name_list.append(base_name)

        corr_matrix_names = base_name_list.copy()

        for base_name, tensors in grouped_tensors.items():
            for k in base_name_list:
                #print(f"{base_name} --- {k}")
                for i in range(num_row):
                    target = grouped_tensors[str(base_name)][i, :min_value]
                    preds = grouped_tensors[str(k)][i, :min_value]
                    corr_matrix.append(kendall(preds.T, target.T))
                    
        
                corr_matrix = np.array(corr_matrix, dtype = np.float32)
                mean = corr_matrix.mean()
                corr_values.append(mean)
                #print(mean)
                corr_matrix = []
    
            corr_dict[base_name] = corr_values
            corr_values = []
            
        matrix = []
        for explainers, values in corr_dict.items():
            matrix.append(values)
    
    
        matrix = np.array(matrix)
        print("----Kendall Correlation Matrix----")
        print(matrix)

        if save_dir == True:
            
            fig, ax = plt.subplots(figsize = (8,6))
            im = ax.imshow(matrix, cmap = color_map)

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
                    text = ax.text(j, i, f"{matrix[i, j]:.3f}",
                            ha="center", va="center", color="w")

            cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=10)

            plt.title('KendallCorrCoef Matrix Heatmap', fontsize=14)

            fig.tight_layout()

            if save_image_to is None:
                plt.savefig("KendallCorrCoef.jpg")
            else:
                plt.savefig(f"{save_image_to}/KendallCorrCoef.jpg")  # Save the image to the certain directory

    if kendall_corr:
        kendall_corr_matrix = np.copy(matrix)

    return spearman_corr_matrix, kendall_corr_matrix

spearman, kendall = explainer_corr(src_dir, save_dir = True, save_image_to = save_path, spearman_corr = True, kendall_corr =True, num_row = 2, color_map="summer")
