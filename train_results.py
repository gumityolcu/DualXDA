import torch
from os import listdir, path, system
from matplotlib import pyplot as plt

def mvdir(p):
    system(f"mv {path.join(p,'outputs','*')} {p}")
    for l in listdir(p):
        if not "best" in l:
            system(f"rm -r {path.join(p,l)}")

def visualize(lists, split, caption="", save_path=None, text_clr="white", cmap="inferno", block=True):
    L=[]
    row_labels=[]
    col_labels=[]
    for label, loss, leng in lists:
        rowlbl=", ".join(label[0:split+1])
        row_labels.append(rowlbl)
        collbl="_".join(label[split+1:])
        col_labels.append(collbl)
        L.append((rowlbl,collbl,loss,leng))
    row_labels=sorted(list(set(row_labels)))
    col_labels=sorted(list(set(col_labels)))
    print(row_labels)    
    print(col_labels)
    row_labels_dict={row_labels[i]:i for i in range(len(row_labels))}
    col_labels_dict={col_labels[i]:i for i in range(len(col_labels))}
    res=torch.empty((len(row_labels),len(col_labels)))
    lengths=torch.zeros_like(res)
    cnt=torch.zeros_like(res)

    fig, ax=plt.subplots(figsize=(10,20))
    print(lengths)
    ax.set_xticks(list(range(len(col_labels))), labels=col_labels)
    ax.set_yticks(list(range(len(row_labels))), labels=row_labels)

    for rowl,coll, loss, leng in L:
        rid=row_labels_dict[rowl]
        cid=col_labels_dict[coll]
        cnt[rid,cid] = cnt[rid,cid] + 1
        res[rid,cid] = loss
        lengths[rid,cid]=leng
        ax.text(cid,rid,f"{loss:.3f}, {leng}",ha="center", va="center", color=text_clr)

    ax.imshow(res, cmap=cmap)
    ax.set_title(caption)
    if save_path is not None:
        plt.savefig(str(save_path)+".png")    
        plt.savefig(str(save_path)+".pdf")    
    plt.show(block=block)



def run(init_dir, split=1, text_clr="black", cmap="inferno"):
    list_of_ckpts=[]
    replace_strs=["MNIST-MNIST_", "CIFAR-CIFAR_", \
                  "std_", "group_", "corrupt_", "mark_", \
                   "0.001_", "0.005_", "0.0001_", \
                     ".yaml-output_data", "_sgd_constant_cross_entropy"]
    dirlist=sorted([f for f in listdir(init_dir) if (".png" not in f) and (".pdf" not in f)])
    for l in dirlist:
        if path.isdir(path.join(init_dir,l, "outputs")):
            mvdir(path.join(init_dir,l))  #moves files out of the outputs folder and deletes everything other than the best ckpt
        label=l
        for rep in replace_strs:
            label=label.replace(rep, "")
            label=label.replace(rep, "")
            label=label.replace(rep, "")
        # here we split : [param1, param2, param3 , ...]
        label=label.split("_")
        assert len(listdir(path.join(init_dir,l)))==1
        fil=listdir(path.join(init_dir,l))[0]
        val_losses=torch.load(path.join(init_dir,l,fil), map_location="cpu")["validation_accuracy"]
        
        list_of_ckpts.append((label, val_losses[-1], len(val_losses)))
    # by giving split=1 we tell it that the rows will be pairs of (param1, param2),
    # splitting the list from index 1, the columns will be n-tuples of params with ids >1
    # you should change split such that the columns are combinations of different augmentations

    visualize(list_of_ckpts, split=split, caption=init_dir.replace("test_output/",""), save_path=path.join(init_dir,"grid"), text_clr=text_clr, cmap=cmap)

if __name__=="__main__":

    # The strings in this list will be deleted from the folder name
    # What we want in the end is:
    # param1_param2_param3_param4_param5
    params_list=[
        ("1e3", "MNIST", 1),
        ("5e3", "MNIST", 1),
        ("1e4", "CIFAR", 1),
        ("5e3", "CIFAR", 1),
    ]
    for lr, ds, i in params_list:
        for typ in ["std", "group", "corrupt", "mark"]:
            init_dir=f"/media/yolcu/DualView/test_output/{ds}/{typ}/{lr}"
            if path.isdir(init_dir):
                print(f"running on {init_dir}")
                run(init_dir, split=i)
            else:
                print(f" not running on {init_dir}")







