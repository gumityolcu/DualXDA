import torch
from torch import tensor
from torchmetrics.regression import SpearmanCorrCoef
from torchmetrics.regression import KendallRankCorrCoef
import os
import numpy as np
from collections import defaultdict

def group_files_by_base_name(src_dir):
    grouped_files = defaultdict(list)
    
    # Walk through all directories and subdirectories in the source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # Check if 'explainer' is in the filename
            if 'explainer' in file.lower():
                # Split the filename at the underscore to get the base name
                base_name = file.split('_')[0]
                #numbers = file.split("_")[1]
                file_path = os.path.join(root, file)
                grouped_files[base_name].append(file_path)
                
    
    return grouped_files

def load_tensors(file_paths):
    tensors = []
    for file_path in file_paths:
        tensor = torch.load(file_path, map_location=torch.device('cpu'))
        tensors.append(tensor)
    return tensors

# Define the source directory (where to search for files)
source_directory = r'C:\Users\akbulut\Desktop\Deniz\Correlations'

# Group files by their base names
grouped_files = group_files_by_base_name(source_directory)

# Load tensors from the grouped files
grouped_tensors = {base_name: load_tensors(file_list) for base_name, file_list in grouped_files.items()}

#Copy the grouped tensors to compare with concatenation operation
compare_tensors = grouped_tensors.copy()

#Concatenate the tensors
for base_name, tensors in grouped_tensors.items(): #tensor list
   grouped_tensors[str(base_name)] = torch.cat(grouped_tensors[str(base_name)], axis = 0)


#Compare the tensors and concatanation operation
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

print("-----------------Spearman's Rank Correlation Coefficient----------------")

base_name_list = []
corr_matrix = []
sum_matrix = []
spearman = SpearmanCorrCoef()
list(corr_matrix)
for base_name, tensors in grouped_tensors.items():
    base_name_list.append(base_name)

for base_name, tensors in grouped_tensors.items():
    for k in base_name_list:
        print(f"{base_name} --- {k}")
        for i in range(100): #Determines how many rows will be compared
            target = grouped_tensors[str(base_name)][i, :4000]
            preds = grouped_tensors[str(k)][i, :4000]
            corr_matrix.append(spearman(preds.T, target.T))
            #print(spearman(preds, target))
        
        corr_matrix = np.array(corr_matrix, dtype = np.float32)
        #print(corr_matrix.shape)
        #print(corr_matrix)
        mean = corr_matrix.mean()
        print(mean)
        corr_matrix = []
    
    base_name_list.remove(str(base_name))


print("-----------------Kendall Rank Correlation Coefficient----------------")

base_name_list = []
corr_matrix = []
sum_matrix = []
kendall = KendallRankCorrCoef()
list(corr_matrix)
for base_name, tensors in grouped_tensors.items():
    base_name_list.append(base_name)
    
for base_name, tensors in grouped_tensors.items():
    for k in base_name_list:
        print(f"{base_name} --- {k}")
        for i in range(10): #Determines how many rows will be compared
            target = grouped_tensors[str(base_name)][i, :4000]
            preds = grouped_tensors[str(k)][i, :4000]
            corr_matrix.append(kendall(preds.T, target.T))
            
        
        corr_matrix = np.array(corr_matrix, dtype = np.float32)
        #print(corr_matrix.shape)
        #print(corr_matrix)
        mean = corr_matrix.mean()
        print(mean)
        corr_matrix = []
    
    base_name_list.remove(str(base_name))