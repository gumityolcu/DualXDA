import torch
import numpy as np
import os

shortcut_train_1000 = "/home/weckbecker/coding/DualXDA/src/dataset/ag_news/train_shortcut_unified_1000_shortcut_index.npy"
shortcut_train_2000 = "/home/weckbecker/coding/DualXDA/src/dataset/ag_news/train_shortcut_unified_2000_shortcut_index.npy"
shortcut_test_1000 = "/home/weckbecker/coding/DualXDA/src/dataset/ag_news/test_unified_1000_shortcut_index.npy"

xpl_1000_folder = "/home/weckbecker/coding/DualXDA/explanations/ag_news/llama/shortcut/1000"
xpl_2000_folder = "/home/weckbecker/coding/DualXDA/explanations/ag_news/llama/shortcut/2000"

def analyse(xpl_path, shortcut_train_idx_path, shortcut_test_idx_path):
    print("-"*30)
    xpl = torch.load(xpl_path, map_location=torch.device('cpu'))
    shortcut_train_idx = np.load(shortcut_train_idx_path)
    shortcut_test_idx = np.load(shortcut_test_idx_path)
    print(len(shortcut_train_idx), len(shortcut_test_idx))
    not_shortcut_train_idx = [i for i in range(xpl.shape[1]) if i not in shortcut_train_idx]
    not_shortcut_test_idx = [i for i in range(xpl.shape[0]) if i not in shortcut_test_idx]
    print(len(not_shortcut_train_idx), len(not_shortcut_test_idx))
    shortcut_xpl_sum = xpl[shortcut_test_idx, :].sum(dim=0)
    print(shortcut_xpl_sum.shape)
    shortcut_sorted_idx = torch.argsort(shortcut_xpl_sum, dim=-1, descending=True, stable=False)
    not_shortcut_xpl_sum = xpl[not_shortcut_test_idx, :].sum(dim=0)
    not_shortcut_sorted_idx = torch.argsort(not_shortcut_xpl_sum, dim=-1, descending=True, stable=False)
    return sum([s in shortcut_train_idx for s in shortcut_sorted_idx[:10]]), sum([s in not_shortcut_train_idx for s in not_shortcut_sorted_idx[:10]])

def sv(xpl_path, shortcut_train_idx_path):
    xpl = torch.load(xpl_path, map_location=torch.device('cpu'))
    xpl_sum = xpl.sum(dim=0)
    #print(xpl_sum)
    not_sv = (torch.isclose(xpl_sum, torch.zeros(xpl_sum.shape), atol=1e-12))
    shortcut_train_idx = np.load(shortcut_train_idx_path)
    return (sum(not_sv).item(), sum([s in shortcut_train_idx for s in torch.where(not_sv)[0]]))


if __name__ == "__main__":
    for C in ["0.1", "0.001", "1e-05"]:
        # print(sv(os.path.join(xpl_1000_folder, f"dualda_{C}_v0/DualDAExplainer-{C}_all"), shortcut_train_1000))
        # print(sv(os.path.join(xpl_2000_folder, f"dualda_{C}_v0/DualDAExplainer-{C}_all"), shortcut_train_2000))

        print(analyse(os.path.join(xpl_1000_folder, f"dualda_{C}_v0/DualDAExplainer-{C}_all"), shortcut_train_1000, shortcut_test_1000))
        print(analyse(os.path.join(xpl_2000_folder, f"dualda_{C}_v0/DualDAExplainer-{C}_all"), shortcut_train_2000, shortcut_test_1000))