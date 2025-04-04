import json
import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns

xai_methods=[
            'lissa',
            'arnoldi',
            'kronfluence',
            'graddot',
            'tracin',
            'trak',
            'representer',
            'dualview_0.1',
            'dualview_0.001',
            'dualview_1e_05',
        ]
xai_legends=[
    "LISSA",
    "Arnoldi",
    "EK-FAC",
    "TracIn",
    "TRAK",
    "GradDot",
    "Representer Points",
    #"DualView 1e-06",
    "DualView $C=10^{-5}$",
    #"DualView 0.0001",
    "DualView $C=10^{-3}$",
    #"DualView 0.01",
    "DualView $C=10^{-1}$",
]
datasets=["MNIST", "CIFAR", "AWA"]
ds_names=["MNIST", "CIFAR-10", "AWA"]
dark_grid_params = {
    'axes.facecolor': '#d6d6f5',  # Darker background
    'axes.grid': True,
    'grid.linestyle': '-',
    'grid.linewidth': 0.8
}

plot_indices=list(range(len(xai_methods)))
plot_indices=plot_indices[-3:]+plot_indices[:-3]
sns.set_theme()
plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
fontdict={"size": 15}
p=f"/home/fe/yolcu/Documents/Code/DualView-wip/test_output/eval/"
os.makedirs(os.path.join(p,"corrupt_plots"),exist_ok=True)
fig=plt.figure(figsize=(16,16))

for i, dsname in enumerate(datasets):
    # Apply the custom style
    sns.set_context("paper")
    #sns.set_theme(style="whitegrid")
    #sns.set_style("darkgrid")
    #fig.subplots_adjust(top=.95)
    plt.subplots_adjust(hspace=.25)
    ax=fig.add_subplot(2,2,i+1)
    sns.color_palette("colorblind")

    plt.xlabel('Ratio of Controlled Images')
    plt.ylabel('Ratio of Detected Poisoned Samples')
    palette=sns.color_palette(n_colors=len(xai_methods)+1)
    for k,j in enumerate(plot_indices):
        if dsname=="AWA" and xai_methods[j]=="trak":
            continue
        if xai_methods[j] not in ["kronfluence", "lissa"]:
            with open(os.path.join(p,dsname, f"{dsname}_corrupt_{xai_methods[j]}_eval_results.json")) as file:
                res = json.load(file)
                arr=res['label_flipping_curve']
                x=torch.range(1,len(arr))/len(arr)
                if "dualview" in xai_methods[j]:
                    plt.plot(x,arr,label=xai_legends[j], color=palette[k])
                else:
                    plt.plot(x,arr,label=xai_legends[j], color=palette[k+1])
    plt.plot([0.,1.],[0.,1.],linestyle="dashed",color="black",label="RAND")
    plt.plot([0.,res['num_corrupt_samples']/len(arr),1.],[0.,1.,1.],color="gray", linestyle="dashed")
    plt.plot([0.,1-res['num_corrupt_samples']/len(arr),1.],[0.,0.,1.],color="gray", linestyle="dashed")
    _,x_lim=plt.xlim()
    _,y_lim=plt.ylim()
    plt.xlim((0.,x_lim))
    plt.ylim((0.,y_lim))
    plt.grid(True, which="major", linestyle='-', linewidth=.8)
    plt.title(f"{ds_names[i]} Dataset", fontsize=15)
    if i==0:
        #### UPPER LEGEND
        #plt.legend(handleheight=1, ncol=4, loc="upper center",
        #           bbox_to_anchor=(2.15, 1.25))
        ### LEFT LEGEND
        reorder = lambda l, nc: sum((l[i::nc] for i in range(nc)), [])
        h, l = ax.get_legend_handles_labels()
        #plt.legend(reorder(h, 3), reorder(l, 3), handleheight=1, ncol=3, loc="upper left",
        #          bbox_to_anchor=(2.15, 1.25))
        plt.legend(handleheight=2, ncol=len(xai_legends), loc="upper center", fontsize=20)
        sns.move_legend(ax, "lower right", bbox_to_anchor=(1.8, -1.), frameon=False,prop={'size': 16})

plt.savefig(os.path.join(p,'corrupt_plots', "label_posioning_curve.pdf"))
os.system(f"pdfcrop {os.path.join(p, 'corrupt_plots', 'label_posioning_curve.pdf')}")
os.system(f"rm {os.path.join(p, 'corrupt_plots', 'label_posioning_curve.pdf')}")
os.system(f"mv {os.path.join(p, 'corrupt_plots', 'label_posioning_curve-crop.pdf')} {os.path.join(p, 'corrupt_plots', 'label_posioning_curve.pdf')}")