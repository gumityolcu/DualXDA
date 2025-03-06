import json
import torch
import matplotlib.pyplot as plt
import os

methods=[
            #('dualview_0.001','black','DV 1e-3'),
            #('lissa', 'green', 'LiSSA IF'),
            #('kronfluence', 'orange', 'LiSSA IF'),
            #('arnoldi', 'red', 'Arnoldi IF'),
            #('representer', 'cyan', 'RP'),
            ('graddot','yellow','GradDot'),
            ('tracin','blue','TracIn'),
        ]

plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
fontdict={"size": 15}

for dsname in ["AWA"]:
    p=f"/home/fe/yolcu/Documents/Code/DualView-wip/test_output/eval/{dsname}/"
    plt.figure(figsize=(8,6))
    plt.xlabel('Ratio of Controlled Images')
    plt.ylabel('Ratio of Detected Poisoned Samples')
    for m,c,n in methods:
        with open(os.path.join(p,f"{dsname}_corrupt_{m}_eval_results.json")) as file:
            res = json.load(file)
        arr=res['label_flipping_curve']
        x=torch.range(1,len(arr))/len(arr)
        plt.plot(x,arr,c,label=n)
    plt.plot([0.,1.],[0.,1.],linestyle="dashed",color="black",label="RAND")
    plt.plot([0.,res['num_corrupt_samples']/len(arr),1.],[0.,1.,1.],color="gray", linestyle="dashed")
    plt.plot([0.,1-res['num_corrupt_samples']/len(arr),1.],[0.,0.,1.],color="gray", linestyle="dashed")
    _,x_lim=plt.xlim()
    _,y_lim=plt.ylim()
    plt.xlim((0.,x_lim))
    plt.ylim((0.,y_lim))
    plt.legend(loc="lower right")
    plt.show()
    continue
    plt.savefig(os.path.join(p,f"{dsname}_label_posioning_curve.pdf"))
    os.system(f"pdfcrop {os.path.join(p,f'{dsname}_label_posioning_curve.pdf')}")
    os.system(f"rm {os.path.join(p,f'{dsname}_label_posioning_curve.pdf')}")
    os.system(f"mv {os.path.join(p,f'{dsname}_label_posioning_curve-crop.pdf')} {os.path.join(p,f'{dsname}_label_posioning_curve.pdf')}")
