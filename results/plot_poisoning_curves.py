import json
import torch
import matplotlib.pyplot as plt
import os

methods=[
            ('input_similarity_dot', None, 'Inp. Sim. Dot'),
            ('feature_similarity_dot', None, 'Feat. Sim. Dot'),
            ('arnoldi', 'red', 'Arnoldi IF'),
            ('representer', 'cyan', 'RP'),
            ('graddot','yellow','GradDot'),
            ('tracin','blue','TracIn'),
            ('trak','blue','TRAK'),
            ('dualview_0.1','black','DV 0.1'),
            ('dualview_0.001','black','DV 0.001'),
            ('dualview_1e_05','black','DV $10^{-5}$'),
        ]
import seaborn as sns
plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
fontdict={"size": 15}
p=f"/home/fe/yolcu/Documents/Code/DualView-wip/test_output/eval/"
os.makedirs(os.path.join(p,"corrupt_plots"),exist_ok=True)
for dsname in ["MNIST","CIFAR","AWA"]:
    sns.set_theme("paper", style="whitegrid")
    plt.figure(figsize=(8,6))
    plt.xlabel('Ratio of Controlled Images')
    plt.ylabel('Ratio of Detected Poisoned Samples')
    for m,c,n in methods:
        if dsname=="AWA" and m=="trak":
            continue
        with open(os.path.join(p,dsname, f"{dsname}_corrupt_{m}_eval_results.json")) as file:
            res = json.load(file)
        arr=res['label_flipping_curve']
        x=torch.range(1,len(arr))/len(arr)
        plt.plot(x,arr,label=n)
    plt.plot([0.,1.],[0.,1.],linestyle="dashed",color="black",label="RAND")
    plt.plot([0.,res['num_corrupt_samples']/len(arr),1.],[0.,1.,1.],color="gray", linestyle="dashed")
    plt.plot([0.,1-res['num_corrupt_samples']/len(arr),1.],[0.,0.,1.],color="gray", linestyle="dashed")
    _,x_lim=plt.xlim()
    _,y_lim=plt.ylim()
    plt.xlim((0.,x_lim))
    plt.ylim((0.,y_lim))
    plt.legend(loc="lower right")
    #plt.show()
    #continue
    plt.savefig(os.path.join(p,'corrupt_plots', f"{dsname}_label_posioning_curve.pdf"))
    os.system(f"pdfcrop {os.path.join(p, 'corrupt_plots', f'{dsname}_label_posioning_curve.pdf')}")
    os.system(f"rm {os.path.join(p, 'corrupt_plots', f'{dsname}_label_posioning_curve.pdf')}")
    os.system(f"mv {os.path.join(p, 'corrupt_plots', f'{dsname}_label_posioning_curve-crop.pdf')} {os.path.join(p, 'corrupt_plots', f'{dsname}_label_posioning_curve.pdf')}")
