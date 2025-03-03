import torch
import os
import json
from copy import deepcopy


def get_xai_names(file_names):
    return_list=[]
    for f in file_names:
        xai_name=f.split("_")[2]
        if xai_name=="dualview":
            xai_name=f"dualview-{f.split('_')[3]}"
        if not xai_name in return_list:
            return_list.append(xai_name)
    return return_list

def get_value(xai, metric, dic):
    return dic["avg_score"]

dataset_name="MNIST"
results_dir="/home/fe/yolcu/Documents/Code/DualView-wip/test_output/eval/"+dataset_name
metrics=["std", "stdk", "group", "groupk", "mark", "corrupt"]

file_names=[]
total_results=[]
for f in [x for x in os.listdir(results_dir) if x.endswith("json")]:
    with open(os.path.join(results_dir,f),"r") as file:
        diction=json.load(file)
    file_names.append(f)
    total_results.append((f, diction))

xai_names=get_xai_names(file_names)
xai_names=sorted(xai_names)



header="""\\begin{table*}[h]
\\caption{Evaluation results for MNIST and CIFAR-10 datasets. For the Domain Mismatch Metric on the CIFAR-10 dataset, the trained model was robust against our perturbations, and thus we were able to use only 72 test samples. For all other evaluations, 4000 test images have been used. Higher score is better for all metrics, best results are in bold. The expected score of a random attributor for each metric is given as RAND, assuming there is no class imbalance in the datasets, which is the case in our experiments.}
\\centering
\\label{table:eval_results}
\\begin{tabular}{"""
header=header+"c|"*(len(metrics)+2)+"}\n"
header=header+"\\cline{"+f"2-{len(metrics)+2}"+"}\n"
header=header+""" & \\textbf{Attribution Method}"""

for m in metrics:
    header=header+" & \\textbf{"+m+"}"
header=header+"\\\\\\hline"

dataset_header="""\\multicolumn{1}{|c|}{\\multirow{11}{*}{\\rotatebox[origin=c]{90}{\\textbf{MNIST}}}} & """

lines=[header, dataset_header]
for i, name in enumerate(xai_names):
    arr = [None for _ in metrics]
    for j,m in enumerate(metrics):
        for fname, dic in total_results:
            val=None
            if f"{dataset_name}_{m}_{name.replace('dualview-','dualview_')}" in fname:
                val=get_value(name,m,dic)
                assert arr[j] is None
                arr[j]=val

    line = "\\multicolumn{1}{|c|}{} & " if not i==0 else ""
    line = line+"\\textbf{"+name+"}"
    for j in range(len(metrics)):
        line=line+ " & "+("{val:.3f}".format(val=arr[j]) if arr[j] is not None else "-")
    line=line+ " \\\\\\cline{2-2}"
    lines.append(line)
    
for l in lines:
    print(l)