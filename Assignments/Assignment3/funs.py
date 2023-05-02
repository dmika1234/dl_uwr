import collections
import logging
import os
import re

import httpimport
import numpy as np
import PIL
import scipy.io
import scipy.ndimage

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.utils.data as data


def generate_prob_heatmaps(vgg, img, id, x_dim, y_dim):
  heatmap = np.zeros((x_dim[1] - x_dim[0], y_dim[1] - y_dim[0]))
  arg_prob_heatmap = np.zeros((x_dim[1] - x_dim[0], y_dim[1] - y_dim[0]))
  occlusion_iter = OcclusionIterator(img, x_dim=x_dim, y_dim=y_dim, occlusion_size=1)
  occlusion_iter_obj = iter(occlusion_iter)
  for i in range(y_dim[1] - y_dim[0]):
      for j in range(x_dim[1] - x_dim[0]):
          occluded_img = to_tensor(next(occlusion_iter_obj))
          probs = to_np(vgg.probabilities(occluded_img))[0]
          heatmap[i, j] += probs[id]
          arg_prob_heatmap[i, j] += np.argmax(probs)
          # print(np.argmax(probs))
  heatmap /= heatmap.max()
  arg_prob_heatmap /= arg_prob_heatmap.max()
  return heatmap, arg_prob_heatmap


def generate_heatmap(vgg, img, x_dim, y_dim, layer_name, map_index):
    heatmap = np.zeros((x_dim[1] - x_dim[0], y_dim[1] - y_dim[0]))
    occlusion_iter = OcclusionIterator(img, x_dim=x_dim, y_dim=y_dim, occlusion_size=1)
    occlusion_iter_obj = iter(occlusion_iter)
    for i in range(y_dim[1] - y_dim[0]):
        for j in range(x_dim[1] - x_dim[0]):
            occluded_img = to_tensor(next(occlusion_iter_obj))
            occluded_activation = vgg.layer_activations(occluded_img, layer_name)
            heatmap[i, j] += to_np(occluded_activation[0, map_index].sum())
    heatmap /= heatmap.max()
    return heatmap


class OcclusionIterator(data.IterableDataset):
    def __init__(self, img, x_dim=(0,0), y_dim=(0,0), occlusion_size=1, occlusion_step=1, gray_value=128):
        super().__init__()
        self.img = img
        self.start_x, self.end_x = x_dim
        self.start_y, self.end_y = y_dim
        self.occlusion_size = occlusion_size
        self.occlusion_step = occlusion_step
        self.height, self.width = img.shape[0:2]
        self.end_x = np.minimum(self.width, self.end_x)
        self.end_y = np.minimum(self.height, self.end_y)
        self.gray_value = gray_value
        
    def __iter__(self):
      for y in range(self.start_y, self.end_y, self.occlusion_step):
        for x in range(self.start_x, self.end_x, self.occlusion_step):
          img_copy = np.copy(self.img)
          img_copy[y:(y+self.occlusion_size), x:(x+self.occlusion_size):,  :] = self.gray_value
          yield img_copy


class VGGPreprocess(torch.nn.Module):
    """Pytorch module that normalizes data for a VGG network
    """

    # These values are taken from http://pytorch.org/docs/master/torchvision/models.html
    RGB_MEANS = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
    RGB_STDS = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

    def forward(self, x):
        """Normalize a single image or a batch of images
        
        Args:
            x: a pytorch Variable containing and float32 RGB image tensor with 
              dimensions (batch_size x width x heigth x RGB_channels) or 
              (width x heigth x RGB_channels).
        Returns:
            a torch Variable containing a normalized BGR image with shape 
              (batch_size x BGR_channels x width x heigth)
        """
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        # x is batch * width * heigth *channels,
        # make it batch * channels * width * heigth
        if x.size(3) == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        means = self.RGB_MEANS
        stds = self.RGB_STDS
        if x.is_cuda:
            means = means.cuda()
            stds = stds.cuda()
        x = (x - Variable(means)) / Variable(stds)
        return x


class VGG(torch.nn.Module):
    """Wrapper around a VGG network allowing convenient extraction of layer activations.
    
    """

    FEATURE_LAYER_NAMES = {
        "vgg16": [
            "conv1_1",
            "relu1_1",
            "conv1_2",
            "relu1_2",
            "pool1",
            "conv2_1",
            "relu2_1",
            "conv2_2",
            "relu2_2",
            "pool2",
            "conv3_1",
            "relu3_1",
            "conv3_2",
            "relu3_2",
            "conv3_3",
            "relu3_3",
            "pool3",
            "conv4_1",
            "relu4_1",
            "conv4_2",
            "relu4_2",
            "conv4_3",
            "relu4_3",
            "pool4",
            "conv5_1",
            "relu5_1",
            "conv5_2",
            "relu5_2",
            "conv5_3",
            "relu5_3",
            "pool5",
        ],
        "vgg19": [
            "conv1_1",
            "relu1_1",
            "conv1_2",
            "relu1_2",
            "pool1",
            "conv2_1",
            "relu2_1",
            "conv2_2",
            "relu2_2",
            "pool2",
            "conv3_1",
            "relu3_1",
            "conv3_2",
            "relu3_2",
            "conv3_3",
            "relu3_3",
            "conv3_4",
            "relu3_4",
            "pool3",
            "conv4_1",
            "relu4_1",
            "conv4_2",
            "relu4_2",
            "conv4_3",
            "relu4_3",
            "conv4_4",
            "relu4_4",
            "pool4",
            "conv5_1",
            "relu5_1",
            "conv5_2",
            "relu5_2",
            "conv5_3",
            "relu5_3",
            "conv5_4",
            "relu5_4",
            "pool5",
        ],
    }

    def __init__(self, model="vgg19"):
        super(VGG, self).__init__()
        all_models = {
            "vgg16": torchvision.models.vgg16,
            "vgg19": torchvision.models.vgg19,
        }
        vgg = all_models[model](pretrained=True)

        self.preprocess = VGGPreprocess()
        self.features = vgg.features
        self.classifier = vgg.classifier
        self.softmax = torch.nn.Softmax(dim=-1)

        self.feature_names = self.FEATURE_LAYER_NAMES[model]

        assert len(self.feature_names) == len(self.features)

    def forward(self, x):
        """ Return pre-softmax unnormalized logits. 
        """
        x = self.preprocess(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def probabilities(self, x):
        """Return class probabilities.
        """
        logits = self.forward(x)
        return self.softmax(logits)

    def layer_activations(self, x, layer_name):
        """Return activations of a selected layer.
        """
        x = self.preprocess(x)
        for name, layer in zip(self.feature_names, self.features):
            x = layer(x)
            if name == layer_name:
                return x
        raise ValueError("Layer %s not found" % layer_name)

    def multi_layer_activations(self, x, layer_names):
        """Return activations of all requested layers.
        """
        pass # TODO implement me

    def predict(self, x, k=1):
        """Return predicted class IDs.
        """
        probabilities = self.probabilities(x)
        top_prob, top_pred = torch.topk(probabilities, k=k)
        return top_pred


class ILSVRC2014Sample(object):
    """Mapper from numerical class IDs to their string LABELS and DESCRIPTIONS.
    
    Please use the dicts:
    - id_to_label and label_to_id to convert string labels and numerical ids
    - label_to_desc to get a textual description of a class label
    - id_to_desc to directly get descriptions for numerical IDs
    
    """

    def load_image(self, path):
        img = PIL.Image.open(path)
        img = img.convert("RGB")
        for t in self.transforms:
            img = t(img)
        return np.asarray(img).astype("float32") / 255.0

    def __init__(self, num=100):
        self.transforms = [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
        ]

        base_dir = "ilsvrc_subsample/"
        devkit_dir = base_dir

        meta = scipy.io.loadmat(devkit_dir + "/meta.mat")
        imagenet_class_names = []
        self.label_to_desc = {}
        for i in range(1000):
            self.label_to_desc[meta["synsets"][i][0][1][0]] = meta["synsets"][i][0][2][
                0
            ]
            imagenet_class_names.append(meta["synsets"][i][0][1][0])

        img_names = sorted(os.listdir(base_dir + "/img"))[:num]
        img_ids = {int(re.search("\d{8}", name).group()) for name in img_names}
        with open(devkit_dir + "/ILSVRC2012_validation_ground_truth.txt", "r") as f:
            self.labels = [
                imagenet_class_names[int(line.strip()) - 1]
                for i, line in enumerate(f)
                if i + 1 in img_ids
            ]
        self.data = [self.load_image(base_dir + "/img/" + name) for name in img_names]

        self.id_to_label = sorted(self.label_to_desc.keys())
        self.label_to_id = {}
        self.id_to_desc = []
        for id_, label in enumerate(self.id_to_label):
            self.label_to_id[label] = id_
            self.id_to_desc.append(self.label_to_desc[label])


def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x, requires_grad=False):
    x = torch.from_numpy(x)
    if CUDA:
        x = x.cuda()
    # return torch.tensor(x, **kwargs)
    if requires_grad:
        return x.clone().contiguous().detach().requires_grad_(True)
    else:
        return x.clone().contiguous().detach()