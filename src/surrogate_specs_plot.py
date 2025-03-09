import torch
from utils.data import load_datasets_reduced
from utils.models import load_model, clear_resnet_from_checkpoints
from scipy.stats import kendalltau
from matplotlib import pyplot as plt
import os
import argparse

def main():
    dataset_name="MNIST"
    device="cuda" if torch.cuda.is_available() else "cpu"
    n_cls={"MNIST":10, "CIFAR":10, "AWA":50}

    C_values=[1e-6,1e-5,0.0001,0.001,0.01,0.1,1.,10.,100.]
    num_classes=n_cls[dataset_name]
    root=f"/mnt/cache/{dataset_name}/std"
    outdir="/mnt/outputs"

    outname=f"{dataset_name}_K_plot"
    preactivations=torch.load(f"{root}/features/samples",map_location=device)
    labels=torch.load(f"{root}/features/labels",map_location=device)

    num_supvecs=[]
    train_times=[]
    test_accs=[]
    train_accs=[]

    ds_kwargs = {
        'data_root': "/mnt/dataset",
        'image_set': "test",
        'validation_size': 2000,
        "only_train": False,
        'testsplit': "test",
        'transform': None,
        'class_groups': None,
        'num_classes': 10 if dataset_name !="AWA" else 50
    }

    model_names={"MNIST":"basic_conv", "CIFAR":"resnet18", "AWA":"resnet50"}
    model_name=model_names[dataset_name]
    model_path=f'{root.replace("cache", "checkpoints")}/{dataset_name}_{model_name}_best'

    train, test = load_datasets_reduced(dataset_name, "std", ds_kwargs)
    model = load_model(model_name, dataset_name, ds_kwargs["num_classes"]).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint=clear_resnet_from_checkpoints(checkpoint)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(device)



    ld=torch.utils.data.DataLoader(test, 32, shuffle=False)
    sv_counts=[]

    for c in C_values:
        svs=[0. for _ in range(num_classes)]
        C=str(c)
        dirname=f"dualview_{C}"
        coefs=torch.load(f"{root}/{dirname}/coefficients",map_location=device)
        for i in range(coefs.shape[0]):
            if (coefs[i]!=0).sum()!=0:
                for j in range(num_classes):
                    if coefs[i, j]!=0.:
                        svs[j]+=1
        sv_counts.append(svs)
        print(f"{dirname}")
        weight=torch.load(f"{root}/{dirname}/weights",map_location=device)
        pred=torch.matmul(preactivations, weight.T).argmax(dim=1)
        train_accs.append((pred==labels).float().mean().item())
        _test_accs=[]
        for i,(x,y) in enumerate(iter(ld)):
            if i>=100:
                break
            x=x.to(device)
            y=y.to(device)
            feat=model.features(x)
            pred=torch.matmul(feat, weight.T).argmax(dim=1)
            _test_accs.append((pred==y).float().mean().item())
        test_accs.append(torch.tensor(_test_accs, device=device).mean().item())
        train_time=torch.load(f"{root}/{dirname}/train_time", map_location=device)
        train_times.append(train_time.item())

    sv_counts=torch.tensor(sv_counts, device=device)

    x_axis=[i-6 for i in range(len(C_values))]
    plt.rcParams['text.usetex'] = True
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    fontdict={"size": 15}

    # plot 1
    fig, ax1=plt.subplots(figsize=(8,6))
    ax1.plot(x_axis, train_accs, label="Train accuracy", color="red")
    #ax1.plot(x_axis, test_accs, label="Test accuracy", color="blue")
    ax1.set_xlabel("$log_{10}K$", fontdict=fontdict)
    ax1.set_xticks(x_axis)
    ax1.set_ylabel("Accuracy", fontdict=fontdict)
    ax2=ax1.twinx()
    ax2.set_ylabel("Train time (s)", fontdict=fontdict)
    ax2.plot(x_axis, train_times, label="Train time", color="black")

    ax1.legend(loc=0, fontsize=12)
    ax2.legend(loc=4, fontsize=12)

    fig.tight_layout()
    plt.savefig(os.path.join(outdir, f"{outname}_1.pdf"))

    # plot 2
    fig, ax1=plt.subplots(figsize=(8,6))

    for i in range(len(C_values)):
        ax1.plot(x_axis,sv_counts.T[i], label=train.class_labels[i])
    ax1.set_xlabel("$log_{10}K$", fontdict=fontdict)
    ax1.set_xticks(x_axis)
    ax1.set_ylabel("Accuracy", fontdict=fontdict)
    ax2=ax1.twinx()
    ax2.set_ylabel("Train time (s)", fontdict=fontdict)
    ax2.plot(x_axis, train_times, label="Train time", color="black")

    ax1.legend(loc=0, fontsize=12)
    ax2.legend(loc=4, fontsize=12)

    fig.tight_layout()
    plt.savefig(os.path.join(outdir, f"{outname}_2.pdf"))

if __name__=="__main__":
    main()
    