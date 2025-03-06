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
    keys=["avg_score", "auc_score"]
    if "avg_score" in dic.keys():
        return dic["avg_score"]
    elif "auc_score" in dic.keys():
        if "coefs_auc_score" in dic.keys():
            return dic["auc_score"], dic["coefs_auc_score"]
        return dic["auc_score"]
dataset_name="AWA"
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

header="\\begin{table*}[h]\n"
"\\caption{Evaluation results for the "+ dataset_name+ " dataset.}\n"
"\\centering\n"
"\\label{table:eval_results}\n"
"\\begin{tabular}{\n"
header=header+"|"+"c|"*(len(metrics)+1)+"}\n"
header=header+"\\hline\n"
header=header+"\\textbf{Attribution Method}"

for m in metrics:
    header=header+" & \\textbf{"+m+"}"
header=header+"\\\\ \\hline"

lines=[header]
for i, name in enumerate(xai_names):
    arr = [None for _ in metrics]
    for j,m in enumerate(metrics):
        for fname, dic in total_results:
            val=None
            if f"{dataset_name}_{m}_{name.replace('dualview-','dualview_')}" in fname:
                val=get_value(name,m,dic)
                assert arr[j] is None
                arr[j]=val

    line = "\\textbf{"+name+"}"
    for j in range(len(metrics)):
        if arr[j] is not None:
            if isinstance(arr[j], tuple):
                addition= " & {val1:.3f} ({val2:.3f})".format(val1=arr[j][0], val2=arr[j][1])
            else:
                addition= " & {val:.3f}".format(val=arr[j])
        else:
            addition=" & -"
        line=line+addition
    line=line+ " \\\\ %\\hline"
    lines.append(line)
    
for l in lines:
    print(l)