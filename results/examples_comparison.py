import argparse
import yaml
import sys
sys.path.append("../src")
from matplotlib.gridspec import GridSpec
from utils.data import load_datasets
from utils.models import clear_resnet_from_checkpoints, compute_accuracy, load_model
import logging
from tqdm import tqdm
import os
from metrics import *


def evaluate(method_name, dataset_name, mispredicted):

    #hyper params
    offset = 0
    page_size = 4
    pages=5
    T=5 #number of most influential samples
    root = "/home/fe/yolcu/Documents/Code/DualView-wip"
    data_root = "/home/fe/yolcu/Documents/Code/DualView-wip/../Datasets"
    save_dir = f"/home/fe/yolcu/Documents/Code/DualView-wip/test_output/examples/{dataset_name}"
    dataset_type = "std"
    device = "cpu"
    method_names= [method_name]#["dualview_0.1", "dualview_0.001", "dualview_1e_05", "lissa", "arnoldi", "representer", "tracin", "gradcos", "graddot"]
    method_title_names= [""]#["dualview_0.1", "dualview_0.001", "dualview_1e_05", "lissa", "arnoldi", "representer", "tracin", "gradcos", "graddot"]
    fontsize=16

    ##################
    model_names={
        "MNIST":"basic_conv",
        "CIFAR":"resnet18",
        "AWA": "resnet50"
    }
    num_classes={
        "MNIST":10,
        "CIFAR":10,
        "AWA":50
    }
    model_name = model_names[dataset_name]
    model_path = os.path.join(root, "checkpoints",dataset_name,"std", f"{dataset_name}_{model_name}_best")
    num_classes = num_classes[dataset_name]
    validation_size = 2000
    xpl_roots = [
        os.path.join(root,"explanations",dataset_name,"std",name) for name in method_names
    ]


    if not torch.cuda.is_available():
        device = "cpu"
    ds_kwargs = {
        'data_root': data_root,
        'class_groups': None,
        'image_set': "test",
        'validation_size': validation_size,
        'only_train': False,
        'num_classes':num_classes,
        'transform': None,
    }
    train, test = load_datasets(dataset_name, dataset_type, **ds_kwargs)
    canonizer = None
    model = load_model(model_name, dataset_name, num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint=clear_resnet_from_checkpoints(checkpoint)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    # if dataset_name == "CIFAR":
    #     offset = 5

    acc, misclassified_indices = compute_accuracy(model, test, device=device)
    if mispredicted:
        all_indices = misclassified_indices[:100]
    else:
        all_indices = list(range(100))
        
    for root in xpl_roots:
        os.makedirs(root, exist_ok=True)
    # if dataset_name == "CIFAR":
    #     all_indices = [all_indices[i] for i in [12, 13, 15, 16, 18, 25, 30, 36, 39, 47]]
    file_lists = [[f for f in os.listdir(xpl_root) if
                   ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and ("tensor" not in f) and (".shark" not in f)] for xpl_root in
                  xpl_roots]
    # file_root = file_list[0].split('_')[0]
    cumul_xpl = [torch.empty(0, len(train), dtype=torch.float32) for _ in xpl_roots]
    for i in range(len(cumul_xpl)):
        cur_index = 0
        filename_root = file_lists[i][0].split('_')[0]
        xpls=[]
        if f"{filename_root}_all" in file_lists[i]:
            file_name = os.path.join(xpl_roots[i], f"{filename_root}_all")
            xpl = torch.load(file_name, map_location=torch.device("cpu"))
            xpls.append(xpl)
        else:
            for j in range(len(file_lists[i])):
                file_name = os.path.join(xpl_roots[i], f"{filename_root}_{j:02d}")
                xpl = torch.load(file_name, map_location=torch.device("cpu"))
                xpls.append(xpl)
        for xpl in xpls:
            xpl = (1 / xpl.abs().max(dim=-1)[0][:, None]) * xpl
            len_xpl = xpl.shape[0]
            for k in range(len_xpl):
                if cur_index + k in all_indices:
                    cumul_xpl[i] = torch.concat((cumul_xpl[i], xpl[None, k]))
            cur_index = cur_index + len_xpl

    for p in range(pages):
        start_ind = offset + p * page_size
        indices = all_indices[start_ind:start_ind + page_size]
        xpl_tensors = [c[start_ind:start_ind + page_size] for c in cumul_xpl]
        fname = f"xpl_comparison_{p}"
        if len(method_names)==1.:
            fname=f"{method_names[0]}_{p}"
        else:
            fname=f"{len(method_names)}_explainers_{p}"
        #generate_comparison_explanations_horizontal_with_small_spaces(model, train, test, xpl_tensors, method_title_names,indices, save_dir, f"{dataset_name}_{'correct' if not mispredicted else 'mispredicted'}_{fname}", T, device, fontsize=fontsize, start_ind=start_ind - offset)
        _2x2exampleplot(model, train, test, xpl_tensors, method_title_names,indices, save_dir, f"{dataset_name}_{'correct' if not mispredicted else 'mispredicted'}_{fname}", T, device, fontsize=fontsize, start_ind=start_ind - offset)
#        generate_comparison_explanations_vertical(model, train, test, xpl_tensors, method_names, indices, save_dir, f"V_{fname}", T, device, start_ind=start_ind-offset)
        


def _2x2exampleplot(model, train, test, xpl_tensors, method_names,
                                                                  indices, save_dir, fname, T, device, fontsize, start_ind):
    assert len(method_names)==1
    assert len(xpl_tensors)==1
    indices=indices[:4]
    buffer = 2
    x, y = test[0]
    samples = torch.empty((0, x.shape[0], x.shape[1], x.shape[2]))
    labels = torch.empty((0,), dtype=torch.int)
    # plt.tight_layout()
    # N = 5  # number of explanations
    # T = 5  # number of train samples
    for ind in indices:
        x, y = test[ind]
        if isinstance(y, tuple):
            y = y[1]
        y = torch.tensor([y], dtype=torch.int)
        samples = torch.cat((samples, torch.unsqueeze(x, dim=0)), dim=0)
        labels = torch.cat((labels, y), dim=0)
    with torch.no_grad():
        preds = torch.argmax(model(samples.to(device)), dim=1)


    persquare = 6
    space = 2
    sep_space=4
    if samples.shape[1] == 1:
        samples = torch.transpose(samples, 1, 2)
        samples = torch.transpose(samples, 2, 3)
    dx=2
    fig = plt.figure(figsize=(26,8))
                     #(dx*(2*(persquare + len(xpl_tensors) * (T * persquare+space)-space)+sep_space),0.09*dx*(4*(persquare + buffer)-buffer)))
    gs = GridSpec(nrows=4*(persquare + buffer)-buffer,
                    ncols=2*(persquare + len(xpl_tensors) * (T * persquare+space)-space)+sep_space)
    gs.tight_layout(fig)
    for i in range(4):
        influence_sort_ids = [torch.argsort(cumul_xpl[i]) for cumul_xpl in xpl_tensors]
        col = i%2
        
        row = int(i/2)
        slice1=(row*2*(persquare+buffer)+int(persquare/2+buffer/2),row*2*(persquare+buffer)+int(3*persquare/2+buffer/2))
        slice2=(col*(len(xpl_tensors)*(T * persquare + space)-space+sep_space+persquare),col*(len(xpl_tensors)*(T * persquare + space)-space+sep_space+persquare)+persquare)
        ax = fig.add_subplot(gs[slice1[0]:slice1[1], slice2[0]:slice2[1]])
        lab=str(train.class_labels[preds[i]])
        if len(lab)>5:
            lab=lab[:5]+"."
        ax.set_title(f'Pred.: {lab}',fontdict={"size":fontsize})

        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,
            left=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
        )
        sample_img = train.inverse_transform(samples[i])
        if samples.shape[-1] == 1:
            sample_img=1.-sample_img
            sample_img = torch.concat((sample_img, sample_img, sample_img), dim=-1)
        sample_img = torch.clip(sample_img, min=0., max=1.)
        # assert sample_img.min() >= 0. and sample_img.max() <= 1.
        if sample_img.shape[0] == 3:
            sample_img = sample_img.transpose(0, 1)
            sample_img = sample_img.transpose(1, 2)
        ax.imshow(sample_img)

        lab=str(train.class_labels[labels[i]])
        if len(lab)>5:
            lab=lab[:5]+"."
        ax.set_ylabel(f'Label: {lab}',fontdict={"size":fontsize})
        for k in range(len(xpl_tensors)):
            for j in range(T):
                x, y = train[influence_sort_ids[k][j]]
                x = train.inverse_transform(x)
                x = torch.transpose(x, 0, 1)
                x = torch.transpose(x, 1, 2)
                if x.shape[-1] == 1:
                    x=1.-x
                    x = torch.concat((x, x, x), dim=-1)

                slice1=(row*2*(persquare+buffer),row*2*(persquare+buffer)+persquare)
                slice2=(col*((T * persquare + space) * len(xpl_tensors)-space+sep_space+(T * persquare + space) *k+persquare) + j * persquare + persquare,col*((T * persquare + space) * len(xpl_tensors)-space+sep_space+persquare) + j * persquare + (T * persquare + space) * k + 2 * persquare)
                ax = fig.add_subplot(gs[slice1[0]:slice1[1], slice2[0]:slice2[1]], facecolor="#ff8080")
 #               ax.patch.set_facecolor()

                if j == T - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Negative',fontdict={"size":fontsize})
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                # axes[1 + 3 * i, 0].set_ylabel('Positive Influence')
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Negative')
                x = torch.clip(x, min=0., max=1.)
                
                ax.imshow(x)
                lab=str(train.class_labels[y])
                if len(lab)>5:
                    lab=lab[:5]+"."
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][j]]:.2f},{lab}",fontdict={"size":fontsize})
                x, y = train[influence_sort_ids[k][-(j + 1)]]
                x = train.inverse_transform(x)
                if x.shape[0] == 1:
                    x = torch.transpose(x, 0, 1)
                    x = torch.transpose(x, 1, 2)
                    x=1.-x
                    x = torch.concat((x, x, x), dim=-1)
                slice1=((row*2+1)*(persquare+buffer),(row*2+1)*(persquare+buffer)+persquare)

                ax = fig.add_subplot(gs[slice1[0]:slice1[1], slice2[0]:slice2[1]])
                ax.patch.set_facecolor("#99ffbb")
                if j == T - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Positive',fontdict={"size":fontsize})
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Positive')
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                x = torch.clip(x, min=0., max=1.)
                assert x.min() >= 0. and x.max() <= 1.0
                if x.shape[0] == 3:
                    x = x.transpose(0, 1)
                    x = x.transpose(1, 2)
                ax.imshow(x)
                lab=str(train.class_labels[y])
                if len(lab)>5:
                    lab=lab[:5]+"."
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][-(j + 1)]]:.2f},{lab}",fontdict={"size":fontsize})
    plt.show(block=True)
    exit()
    #fig.savefig(f"{os.path.join(save_dir, fname)}.png")
    fig.savefig(f"{os.path.join(save_dir, fname)}.pdf")
    os.system(f"pdfcrop {os.path.join(save_dir, fname)}.pdf")
    os.system(f"rm {os.path.join(save_dir, fname)}.pdf")
    os.system(f"mv {os.path.join(save_dir, fname)}-crop.pdf {os.path.join(save_dir, fname)}.pdf")

def generate_comparison_explanations_horizontal_with_small_spaces(model, train, test, xpl_tensors, method_names,
                                                                  indices, save_dir, fname, T, device, fontsize, start_ind):
    buffer = 2
    x, y = test[0]
    samples = torch.empty((0, x.shape[0], x.shape[1], x.shape[2]))
    labels = torch.empty((0,), dtype=torch.int)
    # plt.tight_layout()
    # N = 5  # number of explanations
    # T = 5  # number of train samples
    for ind in indices:
        x, y = test[ind]
        if isinstance(y, tuple):
            y = y[1]
        y = torch.tensor([y], dtype=torch.int)
        samples = torch.cat((samples, torch.unsqueeze(x, dim=0)), dim=0)
        labels = torch.cat((labels, y), dim=0)
    with torch.no_grad():
        preds = torch.argmax(model(samples.to(device)), dim=1)

    N = len(indices)

    # ax0 = fig.add_subplot(gs[0, 0])
    # ax0.plot(time, height)
    # ax1 = fig.add_subplot(gs[1, 0])
    # ax1.plot(time, weight)
    # ax2 = fig.add_subplot(gs[:, 1])
    # ax2.plot(time, score)
    # ax3 = fig.add_axes([0.6, 0.6, 0.2, 0.2])
    # ax3.hist(distribution)
    persquare = 6
    space = 2
    if samples.shape[1] == 1:
        samples = torch.transpose(samples, 1, 2)
        samples = torch.transpose(samples, 2, 3)
    fig = plt.figure(figsize=(((T + 1) * len(xpl_tensors)) * 3, N*7.4))
    gs = GridSpec(nrows=2 * N*(persquare + buffer)-buffer,
                    ncols=(persquare + len(xpl_tensors) * T * persquare + (len(xpl_tensors) - 1) * space))
    gs.tight_layout(fig)
    for k in range(len(xpl_tensors)):
        ax = fig.add_subplot(gs[0:(persquare + buffer)*2*N-buffer,
                                persquare + k * (T * persquare + space):T * persquare + k * (
                                            T * persquare + space) + persquare])
        # ax.yaxis.set_label_position("right")
        ax.set_title(method_names[k],fontdict={"size":fontsize})
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,
            left=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    for i in range(N):
        influence_sort_ids = [torch.argsort(cumul_xpl[i]) for cumul_xpl in xpl_tensors]
        ax = fig.add_subplot(gs[i*2*(persquare+buffer)+int(persquare/2+buffer/2):i*2*(persquare+buffer)+int(3*persquare/2+buffer/2), 0:persquare])
        lab=str(train.class_labels[preds[i]])
        if len(lab)>5:
            lab=lab[:5]+"."
        ax.set_title(f'Pred.: {lab}',fontdict={"size":fontsize})
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,
            left=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
        )
        sample_img = train.inverse_transform(samples[i])
        if samples.shape[-1] == 1:
            sample_img=1.-sample_img
            sample_img = torch.concat((sample_img, sample_img, sample_img), dim=-1)
        sample_img = torch.clip(sample_img, min=0., max=1.)
        # assert sample_img.min() >= 0. and sample_img.max() <= 1.
        if sample_img.shape[0] == 3:
            sample_img = sample_img.transpose(0, 1)
            sample_img = sample_img.transpose(1, 2)
        ax.imshow(sample_img)
        lab=str(train.class_labels[labels[i]])
        if len(lab)>5:
            lab=lab[:5]+"."
        ax.set_ylabel(f'Label: {lab}',fontdict={"size":fontsize})
        for k in range(len(xpl_tensors)):
            for j in range(T):
                x, y = train[influence_sort_ids[k][j]]
                x = train.inverse_transform(x)
                x = torch.transpose(x, 0, 1)
                x = torch.transpose(x, 1, 2)
                if x.shape[-1] == 1:
                    x=1.-x
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[i*2*(persquare+buffer):i*2*(persquare+buffer)+persquare, (T * persquare + space) * k + j * persquare + persquare:(
                                                                                                                         T * persquare + space) * k + j * persquare + 2 * persquare])
                if j == T - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Negative',fontdict={"size":fontsize})
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                # axes[1 + 3 * i, 0].set_ylabel('Positive Influence')
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Negative')
                x = torch.clip(x, min=0., max=1.)
                ax.imshow(x)
                lab=str(train.class_labels[y])
                if len(lab)>5:
                    lab=lab[:5]+"."
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][j]]:.2f},{lab}",fontdict={"size":fontsize})
                x, y = train[influence_sort_ids[k][-(j + 1)]]
                x = train.inverse_transform(x)
                if x.shape[0] == 1:
                    x = torch.transpose(x, 0, 1)
                    x = torch.transpose(x, 1, 2)
                    x=1.-x
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[(i*2+1)*(persquare+buffer):(2*i+1) * (persquare+buffer)+persquare,
                                     (T * persquare + space) * k + j * persquare + persquare:(T * persquare + space) * k + j * persquare + 2 * persquare])
                if j == T - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Positive',fontdict={"size":fontsize})
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Positive')
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                x = torch.clip(x, min=0., max=1.)
                assert x.min() >= 0. and x.max() <= 1.0
                if x.shape[0] == 3:
                    x = x.transpose(0, 1)
                    x = x.transpose(1, 2)
                ax.imshow(x)
                lab=str(train.class_labels[labels[i]])
                if len(lab)>5:
                    lab=lab[:5]+"."
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][-(j + 1)]]:.2f},{lab}",fontdict={"size":fontsize})

    fig.savefig(f"{os.path.join(save_dir, fname)}.png")
    fig.savefig(f"{os.path.join(save_dir, fname)}.pdf")
    os.system(f"pdfcrop {os.path.join(save_dir, fname)}.pdf")
    os.system(f"rm {os.path.join(save_dir, fname)}.pdf")
    os.system(f"mv {os.path.join(save_dir, fname)}-crop.pdf {os.path.join(save_dir, fname)}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--mispredicted", action="store_true")
    args=parser.parse_args()
    method_name = args.method_name
    dataset_name = args.dataset_name
    mispredicted = args.mispredicted
    evaluate(method_name, dataset_name, mispredicted)




def generate_comparison_explanations_vertical(model, train, test, xpl_tensors, method_names, indices, save_dir,
                                              base_fname, T, device, start_ind):
    x, y = test[0]
    samples = torch.empty((0, x.shape[0], x.shape[1], x.shape[2]))
    labels = torch.empty((0,), dtype=torch.int)
    # plt.tight_layout()
    # plt.tight_layout()
    # N = 5  # number of explanations
    # T = 5  # number of train samples
    for ind in indices:
        x, y = test[ind]
        if isinstance(y, tuple):
            y = y[1]
        y = torch.tensor([y], dtype=torch.int)
        samples = torch.cat((samples, torch.unsqueeze(x, dim=0)), dim=0)
        labels = torch.cat((labels, y), dim=0)
    with torch.no_grad():
        preds = torch.argmax(model(samples.to(device)), dim=1)

    N = len(indices)

    # ax0 = fig.add_subplot(gs[0, 0])
    # ax0.plot(time, height)
    # ax1 = fig.add_subplot(gs[1, 0])
    # ax1.plot(time, weight)
    # ax2 = fig.add_subplot(gs[:, 1])
    # ax2.plot(time, score)
    # ax3 = fig.add_axes([0.6, 0.6, 0.2, 0.2])
    # ax3.hist(distribution)

    if samples.shape[1] == 1:
        samples = torch.transpose(samples, 1, 2)
        samples = torch.transpose(samples, 2, 3)
    for i in range(N):
        fig = plt.figure(figsize=((T + 1) * 1.7, 3.5 * len(xpl_tensors)))
        gs = GridSpec(nrows=2 * len(xpl_tensors), ncols=T + 1)
        gs.tight_layout(fig)
        influence_sort_ids = [torch.argsort(cumul_xpl[i]) for cumul_xpl in xpl_tensors]
        ax = fig.add_subplot(gs[0:1, 0:1])
        ax.set_title(f'Pred.: {train.class_labels[preds[i]]}')
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,
            left=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
        )
        sample_img = train.inverse_transform(samples[i])
        if samples.shape[-1] == 1:
            sample_img = torch.concat((sample_img, sample_img, sample_img), dim=-1)
        sample_img = torch.clip(sample_img, min=0., max=1.)
        # assert sample_img.min() >= 0. and sample_img.max() <= 1.
        if sample_img.shape[0] == 3:
            sample_img = sample_img.transpose(0, 1)
            sample_img = sample_img.transpose(1, 2)
        ax.imshow(sample_img)
        ax.set_ylabel(f'Label: {train.class_labels[labels[i]]}')
        for k in range(len(xpl_tensors)):
            ax = fig.add_subplot(gs[2 * k:2 * k + 2, 1:T + 1])
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(method_names[k], fontdict={'size': 12})
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            for j in range(T):
                x, y = train[influence_sort_ids[k][j]]
                x = train.inverse_transform(x)
                x = torch.transpose(x, 0, 1)
                x = torch.transpose(x, 1, 2)
                if x.shape[-1] == 1:
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[2 * k, j + 1])
                if j == 0:
                    ax.set_ylabel('Negative')
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                # axes[1 + 3 * i, 0].set_ylabel('Positive Influence')
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Negative')
                x = torch.clip(x, min=0., max=1.)
                ax.imshow(x)
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][j]]:.2f},  {train.class_labels[y]}")
                x, y = train[influence_sort_ids[k][-(j + 1)]]
                x = train.inverse_transform(x)
                if x.shape[0] == 1:
                    x = torch.transpose(x, 0, 1)
                    x = torch.transpose(x, 1, 2)
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[2 * k + 1, j + 1])
                if j == 0:
                    ax.set_ylabel('Positive')
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Positive')
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                x = torch.clip(x, min=0., max=1.)
                assert x.min() >= 0. and x.max() <= 1.0
                if x.shape[0] == 3:
                    x = x.transpose(0, 1)
                    x = x.transpose(1, 2)
                ax.imshow(x)
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][-(j + 1)]]:.2f},  {train.class_labels[y]}")
        fname = f"{base_fname}-{i}"
        fig.savefig(f"{os.path.join(save_dir, fname)}.pdf")
        os.system(f"pdfcrop {os.path.join(save_dir, fname)}.pdf")
        os.system(f"rm {os.path.join(save_dir, fname)}.pdf")
        os.system(f"mv {os.path.join(save_dir, fname)}-crop.pdf {os.path.join(save_dir, fname)}.pdf")

def mix(x, xpl):
    return .5 * (x + xpl)


def generate_comparison_explanations_horizontal(model, train, test, xpl_tensors, method_names, indices, save_dir,
                                                base_fname, T, device, start_ind):
    x, y = test[0]
    samples = torch.empty((0, x.shape[0], x.shape[1], x.shape[2]))
    labels = torch.empty((0,), dtype=torch.int)
    # plt.tight_layout()
    # N = 5  # number of explanations
    # T = 5  # number of train samples
    for ind in indices:
        x, y = test[ind]
        if isinstance(y, tuple):
            y = y[1]
        y = torch.tensor([y], dtype=torch.int)
        samples = torch.cat((samples, torch.unsqueeze(x, dim=0)), dim=0)
        labels = torch.cat((labels, y), dim=0)
    with torch.no_grad():
        preds = torch.argmax(model(samples.to(device)), dim=1)

    N = len(indices)

    # ax0 = fig.add_subplot(gs[0, 0])
    # ax0.plot(time, height)
    # ax1 = fig.add_subplot(gs[1, 0])
    # ax1.plot(time, weight)
    # ax2 = fig.add_subplot(gs[:, 1])
    # ax2.plot(time, score)
    # ax3 = fig.add_axes([0.6, 0.6, 0.2, 0.2])
    # ax3.hist(distribution)

    if samples.shape[1] == 1:
        samples = torch.transpose(samples, 1, 2)
        samples = torch.transpose(samples, 2, 3)
    for i in range(N):
        fig = plt.figure(figsize=(((T) * len(xpl_tensors) + 1) * 3.4, 7.4))
        gs = GridSpec(nrows=2, ncols=len(xpl_tensors) * (T) + 1)
        gs.tight_layout(fig)
        influence_sort_ids = [torch.argsort(cumul_xpl[i]) for cumul_xpl in xpl_tensors]
        ax = fig.add_subplot(gs[0:1, 0:1])
        ax.set_title(f'Pred.: {train.class_labels[preds[i]]}')
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,
            left=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
        )
        sample_img = train.inverse_transform(samples[i])
        if samples.shape[-1] == 1:
            sample_img = torch.concat((sample_img, sample_img, sample_img), dim=-1)
        sample_img = torch.clip(sample_img, min=0., max=1.)
        # assert sample_img.min() >= 0. and sample_img.max() <= 1.
        if sample_img.shape[0] == 3:
            sample_img = sample_img.transpose(0, 1)
            sample_img = sample_img.transpose(1, 2)
        ax.imshow(sample_img)
        ax.set_ylabel(f'Label: {train.class_labels[labels[i]]}')
        for k in range(len(xpl_tensors)):
            ax = fig.add_subplot(gs[0:2, 1 + k * (T):T + k * (T) + 1])
            # ax.yaxis.set_label_position("right")
            ax.set_title(method_names[k], fontdict={'size': 16}, )
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            for j in range(T):
                x, y = train[influence_sort_ids[k][j]]
                x = train.inverse_transform(x)
                x = torch.transpose(x, 0, 1)
                x = torch.transpose(x, 1, 2)
                if x.shape[-1] == 1:
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[0, (T) * k + j + 1])
                if j == T - 1 and k == len(xpl_tensors) - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Negative')
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                # axes[1 + 3 * i, 0].set_ylabel('Positive Influence')
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Negative')
                x = torch.clip(x, min=0., max=1.)
                ax.imshow(x)
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][j]]:.2f},{train.class_labels[y]}")
                x, y = train[influence_sort_ids[k][-(j + 1)]]
                x = train.inverse_transform(x)
                if x.shape[0] == 1:
                    x = torch.transpose(x, 0, 1)
                    x = torch.transpose(x, 1, 2)
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[1, (T) * k + j + 1])
                if j == T - 1 and k == len(xpl_tensors) - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Positive')
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Positive')
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                x = torch.clip(x, min=0., max=1.)
                assert x.min() >= 0. and x.max() <= 1.0
                if x.shape[0] == 3:
                    x = x.transpose(0, 1)
                    x = x.transpose(1, 2)
                ax.imshow(x)
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][-(j + 1)]]:.2f},{train.class_labels[y]}")
        fname = f"{base_fname}-{i}"

        fig.savefig(f"{os.path.join(save_dir, fname)}.pdf")
        os.system(f"pdfcrop {os.path.join(save_dir, fname)}.pdf")
        os.system(f"rm {os.path.join(save_dir, fname)}.pdf")
        os.system(f"mv {os.path.join(save_dir, fname)}-crop.pdf {os.path.join(save_dir, fname)}.pdf")


def generate_comparison_explanations_horizontal_with_spaces(model, train, test, xpl_tensors, method_names, indices,
                                                            save_dir, base_fname, T, device, start_ind):
    fontsize = 15
    x, y = test[0]
    samples = torch.empty((0, x.shape[0], x.shape[1], x.shape[2]))
    labels = torch.empty((0,), dtype=torch.int)
    # plt.tight_layout()
    # N = 5  # number of explanations
    # T = 5  # number of train samples
    for ind in indices:
        x, y = test[ind]
        if isinstance(y, tuple):
            y = y[1]
        y = torch.tensor([y], dtype=torch.int)
        samples = torch.cat((samples, torch.unsqueeze(x, dim=0)), dim=0)
        labels = torch.cat((labels, y), dim=0)
    with torch.no_grad():
        preds = torch.argmax(model(samples.to(device)), dim=1)

    N = len(indices)

    # ax0 = fig.add_subplot(gs[0, 0])
    # ax0.plot(time, height)
    # ax1 = fig.add_subplot(gs[1, 0])
    # ax1.plot(time, weight)
    # ax2 = fig.add_subplot(gs[:, 1])
    # ax2.plot(time, score)
    # ax3 = fig.add_axes([0.6, 0.6, 0.2, 0.2])
    # ax3.hist(distribution)

    if samples.shape[1] == 1:
        samples = torch.transpose(samples, 1, 2)
        samples = torch.transpose(samples, 2, 3)
    for i in range(N):
        fig = plt.figure(figsize=(((T + 1) * len(xpl_tensors)) * 1.7, 3.7))
        gs = GridSpec(nrows=2, ncols=len(xpl_tensors) * (T + 1))
        gs.tight_layout(fig)
        influence_sort_ids = [torch.argsort(cumul_xpl[i]) for cumul_xpl in xpl_tensors]
        ax = fig.add_subplot(gs[0:1, 0:1])
        ax.set_title(f'Pred.: {train.class_labels[preds[i]]}', fontdict={"size": fontsize})
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,
            left=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
        )
        sample_img = train.inverse_transform(samples[i])
        if samples.shape[-1] == 1:
            sample_img = torch.concat((sample_img, sample_img, sample_img), dim=-1)
        sample_img = torch.clip(sample_img, min=0., max=1.)
        # assert sample_img.min() >= 0. and sample_img.max() <= 1.
        if sample_img.shape[0] == 3:
            sample_img = sample_img.transpose(0, 1)
            sample_img = sample_img.transpose(1, 2)
        ax.imshow(sample_img)
        ax.set_ylabel(f'Label: {train.class_labels[labels[i]]}', fontdict={'size': fontsize})
        for k in range(len(xpl_tensors)):
            ax = fig.add_subplot(gs[0:2, 1 + k * (T + 1):T + k * (T + 1) + 1])
            # ax.yaxis.set_label_position("right")
            ax.set_title(method_names[k], fontdict={"size": fontsize})
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            for j in range(T):
                x, y = train[influence_sort_ids[k][j]]
                x = train.inverse_transform(x)
                x = torch.transpose(x, 0, 1)
                x = torch.transpose(x, 1, 2)
                if x.shape[-1] == 1:
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[0, (T + 1) * k + j + 1])
                if j == T - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Negative', fontdict={"size": fontsize})
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                # axes[1 + 3 * i, 0].set_ylabel('Positive Influence')
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Negative')
                x = torch.clip(x, min=0., max=1.)
                ax.imshow(x)
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][j]]:.2f},{train.class_labels[y]}",
                              fontdict={"size": fontsize})
                x, y = train[influence_sort_ids[k][-(j + 1)]]
                x = train.inverse_transform(x)
                if x.shape[0] == 1:
                    x = torch.transpose(x, 0, 1)
                    x = torch.transpose(x, 1, 2)
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[1, (T + 1) * k + j + 1])
                if j == T - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Positive', fontdict={"size": fontsize})
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Positive')
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                x = torch.clip(x, min=0., max=1.)
                assert x.min() >= 0. and x.max() <= 1.0
                if x.shape[0] == 3:
                    x = x.transpose(0, 1)
                    x = x.transpose(1, 2)
                ax.imshow(x)
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][-(j + 1)]]:.2f},  {train.class_labels[y]}",
                              fontdict={"size": fontsize})
        fname = f"{base_fname}-{i}"

        fig.savefig(f"{os.path.join(save_dir, fname)}.pdf")
        os.system(f"pdfcrop {os.path.join(save_dir, fname)}.pdf")
        os.system(f"rm {os.path.join(save_dir, fname)}.pdf")
        os.system(f"mv {os.path.join(save_dir, fname)}-crop.pdf {os.path.join(save_dir, fname)}.pdf")
