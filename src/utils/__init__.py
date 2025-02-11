from time import time as time
from abc import ABC, abstractmethod
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
import json
from zennit.composites import EpsilonPlus, EpsilonAlpha2Beta1
from zennit.attribution import Gradient
from zennit.image import imgify
import itertools




def zennit_inner_product_explanation(model, train, test, composite_cls=EpsilonAlpha2Beta1, canonizer=None,
                                     mode="train",cmap_name="bwr"):
    with torch.no_grad():
        train_features = model.features(train)
        test_features = model.features(test)
        init = test_features if mode == "train" else train_features
        # init=init/test_features.abs().max()
    if canonizer is not None:
        attributor = Gradient(model.features, composite_cls(canonizers=[canonizer]))
    else:
        attributor = Gradient(model.features, composite_cls())

    input = train if mode == "train" else test

    output, attr = attributor(input, init)
    attr = attr.cpu()
    img = imgify(attr.sum(1), symmetric=True, cmap=cmap_name)

    sensible_palette = {img.palette.colors[color]: torch.tensor(color, dtype=torch.float32) for color in
                        img.palette.colors.keys()}

    tensor_img = torch.tensor(img.getdata()).resize(1, img.size[0], img.size[1])
    rgb_img = torch.empty((tensor_img.shape[1], tensor_img.shape[2], 3), dtype=torch.float32)
    for i in range(tensor_img.shape[1]):
        for j in range(tensor_img.shape[2]):
            rgb_img[i, j] = sensible_palette[int(tensor_img[0, i, j])]
    # return img, attr.sum(1)
    return rgb_img / 255., attr.sum(1)


def xplain(model, train, test, device, explainer_cls, batch_size, kwargs, num_batches_per_file, save_dir,
           start_file, num_files, graddot=False, self_influence=False):
    torch.manual_seed(42)
    # the graddot parameter indicates if we are generating graddot attributions
    # if it is true, graddot.explain will be called a second time with normalize_train=True to generate gradcos along the way
    print("LOG: XPLAIN INTRO")
    explainer = explainer_cls(model=model, dataset=train, device=device, **kwargs)
    explainer.train()
    if self_influence:
        print("LOG: SELF INFLUENCES")
        explainer.self_influences()
        exit()  

    name=explainer_cls.name
    if "DualView" in name:
        name=explainer.get_name()
    test_ld = DataLoader(test, batch_size=batch_size, shuffle=False)
    explanations = torch.empty((0, len(train)), device=device)
    if graddot:
        gradcos_explanations = torch.empty((0, len(train)), device=device)
    i = 0
    j = start_file
    file_indices = torch.zeros(int(len(test) / batch_size) + 1, dtype=torch.int)
    file_indices[start_file * num_batches_per_file:(start_file + num_files) * num_batches_per_file] = 1
    iter_loader = itertools.compress(test_ld, file_indices)
    for u, (x, y) in enumerate(iter_loader):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            preds = torch.argmax(model(x), dim=1)
        xpl = explainer.explain(x=x, xpl_targets=preds)
        explanations = torch.cat((explanations, xpl), dim=0)
        if graddot:
            xpl = explainer.explain(x=x, xpl_targets=preds, normalize_train=True)
            gradcos_explanations = torch.cat((gradcos_explanations, xpl), dim=0)
        #xpl = explainer.explain(x=x, xpl_targets=y) to explain actual labels
        i = i + 1
        if i == num_batches_per_file:
            i = 0
            torch.save(explanations, os.path.join(save_dir, f"{name}_{j:02d}"))
            explanations = torch.empty((0, len(train)), device=device)
            if graddot:
                torch.save(gradcos_explanations, os.path.join(save_dir, f"{explainer_cls.gradcos_name}_{j:02d}"))
                gradcos_explanations = torch.empty((0, len(train)), device=device)
            print(f"Finished file {j:02d}")
            j = j + 1

    if not i == 0:
        torch.save(explanations, os.path.join(save_dir, f"{name}_{j:02d}"))
        if graddot:
            torch.save(gradcos_explanations, os.path.join(save_dir, f"{explainer_cls.gradcos_name}_{j:02d}"))
        print(f"Finished file {j:02d}")

    return explanations

#below function is outdated, copy xplain() from above and include time measurement

# def xplain_to_compute_time(model, train, test, device, explainer_clses, kwargses,
#                            save_dirs, err, start_page, num_pages, page_size, skip=0):
#     explainers = [explainer_clses[i](model=model, dataset=train, device=device, **(kwargses[i])) for i in
#                   range(len(kwargses))]
#     test_ld = DataLoader(test, batch_size=1, shuffle=False)
#     i = 0
#     for page in range(num_pages):
#         xpl_timeses = [[] for t in range(len(explainers))]
#         reses = [{} for t in range(len(explainers))]
#         xpl_instance_ids = [skip + start_page * page_size + page * page_size + i for i in range(page_size)]
#         selection_tensor = torch.zeros(len(test_ld))
#         selection_tensor[xpl_instance_ids] = 1.
#         iter_loader = itertools.compress(test_ld, selection_tensor)
#         for u, (x, y) in enumerate(iter_loader):
#             with torch.no_grad():
#                 x = x.to(device)
#                 y = y.to(device)
#                 preds = torch.argmax(model(x), dim=1)
#             for t in range(len(explainers)):
#                 explainer = explainers[t]
#                 save_dir = save_dirs[t]
#                 if u == 0:
#                     train_time = explainer.train()
#                     reses[t] = {'training': train_time}
#                 print(f"Starting page {page} with page_size={page_size}")
#                 last_time = time()
#                 xpl = explainer.explain(x=x, xpl_targets=preds)
#                 xpl_timeses[t].append(time() - last_time)
#                 # print(xpl_times)
#                 # explanations=torch.cat((explanations, xpl), dim=0)
#                 i = i + 1
#                 reses[t]["xpl"] = xpl_timeses[t]
#                 with open(os.path.join(save_dir, f"resources_page_{page}"), 'w', encoding='utf-8') as f:
#                     json.dump(reses[t], f, ensure_ascii=False, indent=4)
#             print(f"Ended page {page} with page_size={page_size}")


class Metric(ABC):
    name = "BaseMetricClass"

    @abstractmethod
    def __init__(self, train, test):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_result(self, dir):
        pass

    def write_result(self, resdict, dir, file_name):
        with open(f"{dir}/{file_name}", 'w', encoding='utf-8') as f:
            json.dump(self.to_float(resdict), f, ensure_ascii=False, indent=4)
        print(resdict)

    @staticmethod
    def to_float(results):
        if isinstance(results, dict):
            return {key: Metric.to_float(r) for key, r in results.items()}
        elif isinstance(results, str):
            return results
        else:
            if isinstance(results, torch.Tensor):
                results=results.cpu()
            return np.array(results).astype(float).tolist()
