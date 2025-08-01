# imports
import torch

from zennit.composites import EpsilonGammaBox, EpsilonPlusFlat, EpsilonPlus, EpsilonAlpha2Beta1
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient
from zennit.image import imgify

import matplotlib.patches as patches

import matplotlib.pyplot as plt

from utils.data import load_datasets
from utils.models import load_model

def upscale(img,factor=4):
    newimg=torch.zeros((factor*img.shape[0],factor*img.shape[1],3))
    for i in range(28):
        for j in range(28):
                newimg[i*factor:(i+1)*factor,j*factor:(j+1)*factor]=img[i,j]
    return newimg

def add_black_frame(ax):
    """ Adds a black square frame around each subplot """
    rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

def evaluate_sample(model, train, t):
    probs = torch.nn.functional.softmax(model(train.transform(t.numpy()).unsqueeze(dim=0))[0])
    return (probs.argmax().item(), probs.max().item())

def round_zeros(input, eps = 1e-4):
    return input * (input.abs() > eps)

def display_img(ax, input, dataset, channel_dimension=3):
    img = torch.clip(dataset.inverse_transform(input[0].clone().detach()), min=0., max=1.).squeeze()
    if channel_dimension == 3:
        img.permute(1,2,0)
        ax.imshow(img)
    else:
        ax.imshow(img, cmap='gray_r')
    return None

def lrp(test_input, model, class_to_explain = None, composite = EpsilonPlus()):
    num_classes = model.classifier.out_features

    if class_to_explain == None:
        class_to_explain = test_input[1]

    with Gradient(model=model, composite=composite) as attributor:
        out, relevance = attributor(test_input[0].unsqueeze(dim=0), torch.eye(num_classes)[[class_to_explain]])

    relevance = relevance[0].sum(0).detach().cpu()

    return relevance

def xda_heatmap(train_input, test_input, attribution, model, mode = "train", composite = EpsilonPlus()):    

    with torch.no_grad():
        train_features = model.features(train_input[0].unsqueeze(dim = 0))
        test_features = model.features(test_input[0].unsqueeze(dim = 0))
        attr_output = test_features * train_features * (attribution / (train_features @ test_features.T))

    to_attribute = train_input[0] if mode == "train" else test_input[0]

    with Gradient(model=model.features, composite=composite) as attributor:
        out, relevance = attributor(to_attribute.unsqueeze(dim=0), attr_output)

    relevance = relevance[0].sum(0).detach().cpu()

    return relevance

def neuron_xda_all_neurons(train_input, test_input, attribution, model, mode = "train", class_to_explain = None, composite = EpsilonPlus()):    

    if class_to_explain == None:
        class_to_explain = test_input[1]

    relevance_vec = []

    with torch.no_grad():
        train_features = model.features(train_input[0].unsqueeze(dim = 0))
        test_features = model.features(test_input[0].unsqueeze(dim = 0))
        attr_output = test_features * train_features * (attribution / (train_features @ test_features.T))
        to_attribute = train_input[0] if mode == "train" else test_input[0]

    for i in range(train_features.shape[1]):
        attr_output_masked = attr_output * torch.eye(train_features.shape[1])[i]

        with Gradient(model=model.features, composite=composite) as attributor:
            out, relevance = attributor(to_attribute.unsqueeze(dim=0), attr_output_masked)

        relevance = relevance[0].sum(0).detach().cpu()
        relevance_vec.append(relevance)

    return relevance_vec

def neuron_xda(train_input, test_input, neuron_idx, attribution, model, mode = "train", class_to_explain = None, composite = EpsilonPlus()):    

    if class_to_explain == None:
        class_to_explain = test_input[1]

    with torch.no_grad():
        train_features = model.features(train_input[0].unsqueeze(dim = 0))
        test_features = model.features(test_input[0].unsqueeze(dim = 0))
        attr_output = test_features * train_features * (attribution / (train_features @ test_features.T))
        to_attribute = train_input[0] if mode == "train" else test_input[0]
        attr_output_masked = attr_output * torch.eye(train_features.shape[1])[neuron_idx]

    with Gradient(model=model.features, composite=composite) as attributor:
        out, relevance = attributor(to_attribute.unsqueeze(dim=0), attr_output_masked)

    relevance = relevance[0].sum(0).detach().cpu()

    return relevance


# CANONIZERS FOR RESNET
# taken from https://github.com/frederikpahde/xai-canonization

from zennit import canonizers as zcanon
from zennit import torchvision as ztv

class ResNetCanonizer(zcanon.CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''
    def __init__(self):
        super().__init__((
            ztv.ResNetBottleneckCanonizer(),
            ztv.ResNetBasicBlockCanonizer(),
        ))

class ResNetBNCanonizer(zcanon.CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''
    def __init__(self):
        super().__init__((
            ztv.SequentialMergeBatchNorm(),
            ztv.ResNetBottleneckCanonizer(),
            ztv.ResNetBasicBlockCanonizer(),
        ))


###################
### ART HELPERS ###
###################
        
import numpy as np
import matplotlib.transforms
import matplotlib.path
from matplotlib.collections import LineCollection
import matplotlib.colors as colors

def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    '''truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)'''    
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))
    return new_cmap

# from https://stackoverflow.com/questions/47163796/using-colormap-with-annotate-arrow-in-matplotlib
def colourgradarrow(ax, start, end, cmap="viridis", n=50, lw=3):
    #cmap = plt.get_cmap(cmap, n)
    cmap = truncate_colormap(cmap, 0.0, 0.7, n)
    # Arrow shaft: LineCollection
    x = np.linspace(start[0],end[0],n)
    y = np.linspace(start[1],end[1],n)
    points = np.array([x,y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1],points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=lw)
    lc.set_array(np.linspace(0,1,n))
    ax.add_collection(lc)
    # Arrow head: Triangle
    tricoords = [(0,-0.9),(0.9,0),(0,0.9),(0,-0.9)]
    angle = np.arctan2(end[1]-start[1],end[0]-start[0])
    rot = matplotlib.transforms.Affine2D().rotate(angle)
    tricoords2 = rot.transform(tricoords)
    tri = matplotlib.path.Path(tricoords2, closed=True)
    ax.scatter(end[0],end[1], c=1, s=(2*lw)**2, marker=tri, cmap=cmap,vmin=0)
    ax.autoscale_view()

#######################
### CUSTOM LRP RULE ###
#######################
    
# Use a MixedComposite, using EpsilonPlusFlat except for the first n_flat layers
from torch.nn import ReLU
from zennit.composites import NameMapComposite, MixedComposite
from zennit.rules import Flat


def flatten_epsilonplusflat(n_flat=10):
    flat_layer_names_model = [f"features.{i}" for i in range(n_flat) if not isinstance(model.features[i], ReLU)]
    flat_layer_names_features = [f"{i}" for i in range(n_flat) if not isinstance(model.features[i], ReLU)]

    flatcomposite_model = NameMapComposite(
        name_map = [(flat_layer_names_model, Flat(zero_params='bias'))]
    )

    flatcomposite_features = NameMapComposite(
        name_map = [(flat_layer_names_features, Flat(zero_params='bias'))]
    )   

    flattened_epsilonplusflat = MixedComposite([flatcomposite_model, flatcomposite_features, EpsilonPlusFlat(zero_params='bias')])
    return flattened_epsilonplusflat

#####################
### CONFIG LOADER ###
#####################

import yaml
import warnings

REQUIRED_KW = ["xpl_path", "dataset_name", "data_root", "model_name", "num_classes", "validation_size", "test_idx"]


def load_from_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

        missing_keys = [k for k in REQUIRED_KW if k not in config]
        if missing_keys:
            raise ValueError(f"Missing required keys in config: {', '.join(missing_keys)}")
        if "device" not in config:
            warnings.warn("Device not specified in config, defaulting to 'cpu'.", UserWarning)
        if "nrows" not in config:
            warnings.warn("Number of proponents/opponents to display not specified in config, defaulting to 5.", UserWarning)
        if "composite" not in config:
            warnings.warn("Composite not specified in config, defaulting to EpsilonPlusFlat.", UserWarning)
        if "save_path" not in config:
            warnings.warn(f"Save path not specified in config, defaulting to './xda/{config['dataset_name']}'.", UserWarning)

        device = torch.device(config.get('device', 'cpu'))
        xpl = torch.load(config['xpl_path'], map_location=device)
        dataset_name = config['dataset_name']
        data_root = config['data_root']
        model_name = config['model_name']
        num_classes = config['num_classes']
        validation_size = config['validation_size']
        test_idx = config['test_idx']
        save_path = config.get('save_path', f'./xda/{dataset_name}')
        save_path = os.path.join(save_path, dataset_name)
        ds_kwargs = {
            'data_root': data_root, 
            'class_groups': None,
            'num_classes': num_classes,
            'image_set': "test",
            'validation_size': validation_size,
            'only_train': False,
            'transform': None
        }
        train, test = load_datasets(dataset_name, "std", **ds_kwargs)
        model = load_model(model_name, dataset_name, num_classes).to(device)
        model.eval()

        nrows = config.get('nrows', 5)
        composite_name = config.get('composite', 'EpsilonPlusFlat')
        if composite_name == 'EpsilonPlusFlat':
            composite = EpsilonPlusFlat(zero_params='bias')
        elif composite_name == 'EpsilonPlus':
            composite = EpsilonPlus(zero_params='bias')
        elif composite_name == 'EpsilonAlpha2Beta1':
            composite = EpsilonAlpha2Beta1(zero_params='bias')
        elif composite_name == 'EpsilonGammaBox':
            composite = EpsilonGammaBox(zero_params='bias')
        elif composite_name == 'Flatten_EpsilonPlusFlat':
            composite = flatten_epsilonplusflat(n_flat=10)
        else:
            raise ValueError(f"Unknown composite: {composite_name}")
        return xpl, train, test, model, test_idx, nrows, composite, save_path

####################
### XDA FUNCTION ###
####################

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
import os

matplotlib.pyplot.rcParams.update({
    "text.usetex": True,
    #'text.latex.preamble': r'\usepackage{xcolor}'
})


def xda(xpl, train, test, model, test_idx, nrows, composite, save_path):
    size=2
    test_sample = test[test_idx]
    train_indices_most = torch.topk(xpl[test_idx], nrows).indices
    train_indices_least = torch.topk(-xpl[test_idx], nrows).indices
    channel_dimension = test_sample[0].shape[0]


    # Create figure with a specific size ratio to keep squares
    fig = plt.figure(figsize=((2*nrows+2)*size, 4*size))

    # Create a custom grid
    gs = gridspec.GridSpec(6, 2*nrows+2+1+2, figure=fig)

    # Set spacing between subplots
    gs.update(wspace=0.05, hspace=0.2)

    # Big square 2x2 in the middle
    ax_big = fig.add_subplot(gs[1:3, nrows+1:nrows+2+1])
    display_img(ax_big, test_sample, test, channel_dimension)
    #ax_big.set_title("Test Sample", fontsize=20)
    # Add black frame but still hide axes ticks
    ax_big.axis('on')
    ax_big.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # Set spines (borders) to black
    for spine in ax_big.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    # Big LRP square 2x2 in the middle
    ax_big = fig.add_subplot(gs[1:3, nrows+1+2:nrows+2+1+2])
    relevance = lrp(test_sample, model, composite=composite)
    img = imgify(relevance, cmap='bwr', symmetric = True)
    display_img(ax_big, test_sample, test, channel_dimension)
    ax_big.imshow(img, alpha=.9)
    #ax_big.set_title("Test Sample", fontsize=20)
    # Add black frame but still hide axes ticks
    ax_big.axis('on')
    ax_big.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # Set spines (borders) to black
    for spine in ax_big.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2)

    # Middle row: Picture of proponents / opponents
    for i in range(nrows):
        # Proponent
        ax = fig.add_subplot(gs[2, nrows+2+i+1+2])
        display_img(ax, train[train_indices_most[i]], test, channel_dimension)
        # Add black frame but still hide axes ticks
        ax.axis('on')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # Set spines (borders) to black
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2)
        # Opponent
        ax = fig.add_subplot(gs[2, nrows-1-i+1])
        display_img(ax, train[train_indices_least[i]], test, channel_dimension)
        # Add black frame but still hide axes ticks
        ax.axis('on')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # Set spines (borders) to black
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2)

    # Middle row: Add Relevance and label
    for i in range(nrows):
        # Proponent
        ax = fig.add_subplot(gs[1, nrows+2+i+1+2])
        ax.axis('off')
        # Add title lower in the cell (at y=0.3 instead of 0.5)
        ax.text(0.5, 0.05, f"Relevance: {xpl[test_idx, train_indices_most[i]]:.2f},\nLabel: {train.class_name_dict[train[train_indices_most[i]][1]] if hasattr(train, "class_name_dict") else train[train_indices_most[i]][1]}", ha='center', va='center', fontsize=9)
        # Opponent
        ax = fig.add_subplot(gs[1, nrows-1-i+1])
        ax.axis('off')
        # Add title lower in the cell (at y=0.3 instead of 0.5)
        ax.text(0.5, 0.05, f"Relevance: {xpl[test_idx, train_indices_least[i]]:.2f},\nLabel: {train.class_name_dict[train[train_indices_least[i]][1]] if hasattr(train, "class_name_dict") else train[train_indices_most[i]][1]}", ha='center', va='center', fontsize=9)

    # Middle row: Add titles (proponents, opponents) 
    # Proponent
    ax = fig.add_subplot(gs[1, nrows+2+1+2:2*nrows+2+1+2])
    plt.text(0.5, 0.8, '\\textbf{POSITIVELY} relevant training samples', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20)
    ax.axis('off')
    # Opponent
    ax = fig.add_subplot(gs[1, 0+1:nrows+1])
    plt.text(0.5, 0.8, '\\textbf{NEGATIVELY}  relevant training samples', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20)
    ax.axis('off')

    # Add hline between Row 1 and 2 and between Row 3 and 4
    ax = fig.add_subplot(gs[0:2, 0:2*nrows+2+1+2])
    ax.axhline(y=1/2, xmin=0.04, xmax=1, linestyle = "-", linewidth=2, color='black')
    ax.axis('off')

    ax = fig.add_subplot(gs[2:4, 0:2*nrows+2+1+2])
    ax.axhline(y=1/2, xmin=0.04, xmax=1, linestyle = "-", linewidth=2, color='black')
    ax.axis('off')

    # Row 1 + Row 4: 'XDA' title
    ax = fig.add_subplot(gs[0, nrows+1:nrows+2+1+2])
    ax.text(0.5, 0.5, '\\textbf{XDA}', fontsize=30, ha='center', va='center')
    ax.axis('off')
    ax = fig.add_subplot(gs[3, nrows+1:nrows+2+1+2])
    ax.text(0.5, 0.5, '\\textbf{XDA}', fontsize=30, ha='center', va='center')
    ax.axis('off')

    # Row 1 + Row 5: XDA
    for i in range(nrows):
        # Proponents
        # Train
        ax = fig.add_subplot(gs[0, nrows+2+i+1+2])
        relevance = xda_heatmap(train[train_indices_most[i]], test_sample, xpl[test_idx, train_indices_most[i]], model, mode = 'train', composite = composite)
        img = imgify(relevance, cmap='bwr', symmetric = True)
        display_img(ax, train[train_indices_most[i]], test, channel_dimension)
        ax.imshow(img, alpha=.9)
        ax.axis('on')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2)
        # Test
        ax = fig.add_subplot(gs[3, nrows+2+i+1+2])
        relevance = xda_heatmap(train[train_indices_most[i]], test_sample, xpl[test_idx, train_indices_most[i]], model, mode = 'test', composite = composite)
        img = imgify(relevance, cmap='bwr', symmetric = True)
        display_img(ax, test_sample, test, channel_dimension)
        ax.imshow(img, alpha=.9)
        ax.axis('on')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2)
        
        # Opponents
        # Train
        ax = fig.add_subplot(gs[0, nrows-1-i+1])
        relevance = xda_heatmap(train[train_indices_least[i]], test_sample, xpl[test_idx, train_indices_least[i]], model, mode = 'train', composite = composite)
        img = imgify(relevance, cmap='bwr', symmetric = True)
        display_img(ax, train[train_indices_least[i]], test, channel_dimension)
        ax.imshow(img, alpha=.9)      
        ax.axis('on')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2)
        # Test
        ax = fig.add_subplot(gs[3, nrows-1-i+1])
        relevance = xda_heatmap(train[train_indices_least[i]], test_sample, xpl[test_idx, train_indices_least[i]], model, mode = 'test', composite = composite)
        img = imgify(relevance, cmap='bwr', symmetric = True)
        display_img(ax, test_sample, test, channel_dimension)
        ax.imshow(img, alpha=.9)
        ax.axis('on')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2)

        # Add proponent arrow (red)
        ax = fig.add_subplot(gs[1, nrows+2+1+2:2*nrows+2+1+2])
        colourgradarrow(ax, (0,.5), (nrows,.5), cmap='Reds_r', n=100, lw=10)
        ax.axis('off')

        # Add opponent arrow (blue)
        ax = fig.add_subplot(gs[1, 0+1:nrows+1])
        colourgradarrow(ax, (nrows,.5), (0,.5), cmap='Blues_r', n=100, lw=10)
        ax.axis('off')

        # Add train/text
        ax = fig.add_subplot(gs[0, 0])
        ax.text(0.8, 0.5, 'Train', rotation=90, fontsize=30, ha='center', va='center')
        ax.axis('off')

        ax = fig.add_subplot(gs[3, 0])
        ax.text(0.8, 0.5, 'Test', rotation=90, fontsize=30, ha='center', va='center')
        ax.axis('off')
    
        # Add vertical line
        ax = fig.add_subplot(gs[0:4, 0:2])
        ax.axvline(x=1/2, ymin=0., ymax=1, linestyle = "-", linewidth=2, color='black')
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{test_idx}.png"), dpi=300, bbox_inches='tight')

###################
### CLI SUPPORT ###
###################
import argparse
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()
    config_file = args.config_file
    if not os.path.exists(config_file):
        raise FileNotFoundError("Config file does not exist. Please provide a valid path.")

    xpl, train, test, model, test_idx, nrows, composite, save_path = load_from_config(config_file)
    xda(xpl, train, test, model, test_idx, nrows, composite, save_path)