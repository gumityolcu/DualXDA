from torch.nn import Module, Sequential, Conv2d, ReLU, Linear, Flatten, LeakyReLU, MaxPool2d
import os
import torch

"""

Basic Model:
Only linear
Conv+linear
With and without biases

VGG with and without biases

"""


class BasicModel(Module):
    empty = {
        'num': 0,
        'padding': 0,
        'kernel': 0,
        'stride': 0,
        'features': 0
    }

    def __init__(self, input_shape, num_classes, convs=None, fc=None, bias=True, leaky=False):
        super(BasicModel, self).__init__()
        if convs is None:
            convs = BasicModel.empty
        if fc is None:
            fc = BasicModel.empty
        x = torch.zeros(size=input_shape)

        if isinstance(convs['kernel'], int):
            convs['kernel'] = [convs['kernel'] for _ in range(convs['num'])]
        if isinstance(convs['padding'], int):
            convs['padding'] = [convs['padding'] for _ in range(convs['num'])]
        if isinstance(convs['stride'], int):
            convs['stride'] = [convs['stride'] for _ in range(convs['num'])]
        if isinstance(convs['features'], int):
            convs['features'] = [convs['features'] for _ in range(convs['num'])]
        assert convs['num'] == len(convs['kernel'])
        assert convs['num'] == len(convs['padding'])
        assert convs['num'] == len(convs['stride'])
        assert convs['num'] == len(convs['features'])
        assert fc['num'] == len(fc['features'])
        activation_class = LeakyReLU if leaky else ReLU
        self.convs = convs
        self.fc = fc
        self.bias = bias
        self.leaky = leaky
        self.features = Sequential()
        for c in range(convs['num']):
            module = Conv2d(x.shape[0], convs['features'][c], kernel_size=convs['kernel'][c],
                            padding=convs['padding'][c], stride=convs['stride'][c], bias=bias)
            with torch.no_grad():
                x = module(x)
            self.features.add_module(name=f'conv-{c}',
                                     module=module)
            self.features.add_module(name=f"relu-{c}", module=activation_class())
        self.features.add_module(name='flatten', module=Flatten())
        x = torch.flatten(x)
        last_features = x.shape[0]
        for i in range(fc['num']):
            self.features.add_module(name=f'fc-{i}',
                                     module=Linear(in_features=last_features, out_features=fc['features'][i],
                                                   bias=bias))
            last_features = fc['features'][i]
            self.features.add_module(name=f"relu-{convs['num'] + i}", module=activation_class())
        self.classifier = Linear(in_features=last_features, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def influence_named_parameters(self):
       return [("classifier.weight", self.classifier.weight)]

    def sim_parameters(self):
        return self.parameters()


class BasicConvModel(BasicModel):
    default_convs = {
        'num': 3,
        'padding': 0,
        'kernel': 3,
        'stride': 1,
        'features': [5, 10, 5]
    }
    default_fc = {
        'num': 2,
        'features': [500, 100]
    }

    def __init__(self, input_shape, num_classes, convs=None, fc=None, leaky=False):
        if convs is None:
            convs = BasicConvModel.default_convs
        if fc is None:
            fc = BasicConvModel.default_fc

        super(BasicConvModel, self).__init__(
            num_classes=num_classes,
            convs=convs,
            fc=fc,
            leaky=leaky,
            input_shape=input_shape
        )

