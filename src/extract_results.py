import os
import sys

tgz_path="/home/fe/yolcu/Documents/Code/DualView-wip/experiments/august"
ex_path="/home/fe/yolcu/Documents/Code/DualView-wip/explanations/MNIST"
list_of_xai_methods = []
list_of_ds_types = []
files=[f for f in os.listdir(tgz_path) if f.endswith(".tgz")]
for f in files:
    xai_method=f.replace(".yaml-output_data.tgz","").split("_")
    if xai_method[-2] in ["std", "mark", "corrupt", "group"]:
        ds_type=xai_method[-2]
        xai_method=xai_method[-1]
    else:
        ds_type=xai_method[-3]
        xai_method=f"{xai_method[-2]}_{xai_method[-1]}"
    list_of_ds_types.append(ds_type)
    if xai_method not in list_of_xai_methods:
        list_of_xai_methods.append(xai_method)
    final_path=os.path.join(ex_path, ds_type, xai_method)
    os.makedirs(final_path, exist_ok=True)
    os.system(f"tar -xvf {os.path.join(tgz_path,f)} -C {final_path}")

for m in list_of_xai_methods:
    for t in list_of_ds_types:
        os.system(f"mv {os.path.join(ex_path, t, m, 'outputs','*')} {os.path.join(ex_path, t, m)}")
        os.system(f"rm -r {os.path.join(ex_path, t, m, 'outputs')}")