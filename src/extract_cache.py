import os
import sys

tgz_path="/home/fe/yolcu/Documents/Code/DualView-wip/experiments/october_cache"
ex_path="/home/fe/yolcu/Documents/Code/DualView-wip/test_output/"
list_of_xai_methods = []
list_of_ds_types = []
files=[f for f in os.listdir(tgz_path) if f.endswith(".tgz")]
for f in files:
    os.system(f"tar -xvf {os.path.join(tgz_path,f)} -C {ex_path}")

os.system(f"mv {os.path.join(ex_path, 'outputs/*')} {os.path.join(ex_path)}")
os.system(f"rm -r {os.path.join(ex_path, 'outputs')}")