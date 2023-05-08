import collections
import logging
import os
import re

import httpimport
import numpy as np
import PIL
import scipy.io
import scipy.ndimage
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.utils.data as data


class FGSM_Attack:
    def __init__(self, model, cuda=True):
        self.model = model
        self.cuda = cuda
    
    def generate(self, img, target_class, epsilon=0.1, patch_loc=(0, 0), patch_size=(224, 224)):
        if self.cuda:
            self.model.cuda().eval()
        else:
            self.model.eval()
        # Convert the image to a tensor
        img = to_tensor(img, requires_grad=True, cuda=True)
        patch_img = torch.zeros_like(img)
        patch_img[patch_loc[0]:patch_loc[0]+patch_size[0]-1, patch_loc[0]:patch_loc[1]+patch_size[1]-1] = 1
        # Get the model's output for the image
        output = self.model(img.unsqueeze(0))

        # Calculate the loss between the output and the target class
        loss = F.cross_entropy(output, to_tensor(np.array([target_class]), cuda=True))

        # Calculate gradients of the loss w.r.t. the image tensor
        self.model.zero_grad()
        loss.backward()
        gradient = img.grad.data

        # Create adversarial image by adding a perturbation to the original image
        perturbed_image = img + epsilon * gradient.sign() * patch_img

        # Clip the pixel values of the adversarial image to the valid range [0,1]
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        # Convert the adversarial image back to a numpy array
        self.perturbed_image_np = to_np(perturbed_image)
        # self.plot()
        return self.perturbed_image_np

    def plot(self):
        plt.axis("off")
        plt.imshow(self.perturbed_image_np)



class DeepDreamViz():
    def __init__(self, model, size=64,
                 upscaling_factor=1.2,
                 blur=1,
                 cuda=True):
        self.size = size
        self.upscaling_factor = upscaling_factor
        self.model = model
        self.cuda = cuda
        self.blur=torchvision.transforms.GaussianBlur(blur)
        self.transforms = [
            torchvision.transforms.RandomCrop(224)
        ]
        
    def load_image(self, path):
        img = PIL.Image.open(path)
        img = img.convert("RGB")
        return np.asarray(img).astype("float32") / 255.0    
    
    def preprocess(self,x):
        if x.shape[1] < 224:
            x = torchvision.transforms.Resize((224,224))(x)
        for t in self.transforms:
            x = t(x)
        return x

    
    def generate_imgs(self, img, upsc_img_num):
        imgs = [img]
        for _ in range(upsc_img_num - 1):
            new_size = int(imgs[-1].shape[1]*self.upscaling_factor)
            imgs.append(torchvision.transforms.Resize((new_size, new_size))(imgs[-1]))
        return imgs
        
    def visualize(self, path, layer_name, filter, upsc_img_num, lr=0.01, opt_steps=20):
        if self.cuda:
            self.model.cuda().eval()
        else:
            self.model.eval()
        img = self.load_image(path)
        img = to_tensor(change_axis_totensor(img), cuda=self.cuda)
        img = torchvision.transforms.Resize((self.size, self.size))(img)
        imgs = self.generate_imgs(img, upsc_img_num)
        
        change = to_tensor(np.zeros_like(to_np(imgs[-1])), cuda=self.cuda)
        for _, img_tens in enumerate(imgs[::-1]):
            img_tens_cp = to_tensor(to_np(img_tens).copy(), cuda=self.cuda)
            with torch.no_grad():
                img_tens += torchvision.transforms.Resize((img_tens.shape[1:2]))(change)
            img_tens.requires_grad = True
            adam = torch.optim.Adam([img_tens], lr=lr, weight_decay=0)
            for n in range(opt_steps):
                adam.zero_grad()
                img_trans = self.preprocess(img_tens).cuda()
                activations = self.model.layer_activations(img_trans, layer_name)[0]
                loss = -torch.mean(activations[filter])
                loss.backward()
                adam.step()
                with torch.no_grad():
                    torch.clip(self.blur(img_tens),0,1)
            change = img_tens - img_tens_cp
            assert np.any(to_np(change) != 0)
                
        return change_axis_tonp(to_np(img_tens))


class FilterVisualizer():
    def __init__(self, model, size=64,
                 upscaling_steps=10,
                 upscaling_factor=1.2,
                 blur=1,
                 cuda=True):
        self.size = size
        self.upscaling_steps = upscaling_steps
        self.upscaling_factor = upscaling_factor
        self.model = model
        self.cuda = cuda
        if self.cuda:
            self.model.cuda().eval()
        self.blur=torchvision.transforms.GaussianBlur(blur)
        for param in self.model.parameters():
            param.requires_grad = False

        self.transforms = [
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomCrop(224)
        ]

    def preprocess(self,x):
        for t in self.transforms:
            x = t(x)
        return x
    
    def scaleup(self, img, sz):
        tensor = torchvision.transforms.Resize((sz, sz))(img)
        tensor = to_tensor(to_np(tensor), requires_grad=True, cuda=self.cuda)
        return tensor
    
    def visualize(self, layer_name, filter, lr=0.01, opt_steps=20):
        if self.cuda:
            self.model.cuda().eval()
        else:
            self.model.eval()
        sz = self.size
        img = np.array(np.random.rand(sz, sz, 3), dtype=np.float32)
        img_tens = to_tensor(change_axis_totensor(img), requires_grad=True, cuda=self.cuda)

        for _ in range(self.upscaling_steps):
            adam = torch.optim.Adam([img_tens], lr=lr, weight_decay=1e-6)
            for n in range(opt_steps):
                img_var = self.preprocess(img_tens)
                adam.zero_grad()
                activations = self.model.layer_activations(img_var, layer_name)[0]
                loss = -torch.mean(activations[filter])
                loss.backward()
                adam.step()
            sz = int(self.upscaling_factor * sz)
            img_tens = self.scaleup(self.blur(img_tens), sz)

        self.output = change_axis_tonp(to_np(img_tens))
        self.plot()
    
    def plot(self):
        plt.axis("off")
        plt.imshow(self.output)


def change_axis_tonp(array):
    return np.moveaxis(array,0, 2)


def change_axis_totensor(array):
    return np.moveaxis(array, 2, 0)


def train(
    model, data_loaders, optimizer, criterion, train_transform=None, num_epochs=1, log_every=100, cuda=True
):
    if cuda:
        model.cuda()

    iter_ = 0
    epoch = 0
    best_params = None
    best_val_err = np.inf
    history = {"train_losses": [], "train_errs": [], "val_errs": []}
    print("Training the model!")
    print("You can interrupt it at any time.")
    try:
        while epoch < num_epochs:
            model.train()
            # model.train_mode()
            epoch += 1
            for x, y in data_loaders["train"]:
                if train_transform:
                    x = train_transform(x)
                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                iter_ += 1

                optimizer.zero_grad()
                out = model.forward(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                _, predictions = out.max(dim=1)
                err_rate = 100.0 * (predictions != y).sum() / out.size(0)

                history["train_losses"].append(loss.item())
                history["train_errs"].append(err_rate.item())

                                
                if iter_ % log_every == 0:
                    print(
                        "Minibatch {0: >6}  | loss {1: >5.2f} | err rate {2: >5.2f}%".format(
                            iter_, loss.item(), err_rate
                        )
                    )

            val_err_rate = compute_error_rate(model, data_loaders["test"], cuda)
            history["val_errs"].append((iter_, val_err_rate))

            if val_err_rate < best_val_err:

                                
                best_epoch = epoch
                best_val_err = val_err_rate

                
            m = "After epoch {0: >2} | valid err rate: {1: >5.2f}% | doing {2: >3} epochs".format(
                epoch, val_err_rate, num_epochs
            )
            print("{0}\n{1}\n{0}".format("-" * len(m), m))
    except KeyboardInterrupt:
        pass
    if best_params is not None:
        print("\nLoading best params on validation set (epoch %d)\n" % (best_epoch))
        model.parameters = best_params
    plot_history(history)


# Train only the classifier!
def compute_error_rate(model, data_loader, cuda=True):
    model.eval()
    num_errs = 0.0
    num_examples = 0
    for x, y in data_loader:
        if cuda:
            x = x.cuda()
            y = y.cuda()

        with torch.no_grad():
            outputs = model.forward(x)
            _, predictions = outputs.max(dim=1)
            num_errs += (predictions != y).sum().item()
            num_examples += x.size(0)
    return 100.0 * num_errs / num_examples


def plot_history(history):
    # figsize(16, 4)
    plt.subplot(1, 2, 1)
    train_loss = np.array(history["train_losses"])
    plt.semilogy(np.arange(train_loss.shape[0]), train_loss, label="batch train loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    train_errs = np.array(history["train_errs"])
    plt.plot(np.arange(train_errs.shape[0]), train_errs, label="batch train error rate")
    val_errs = np.array(history["val_errs"])
    plt.plot(val_errs[:, 0], val_errs[:, 1], label="validation error rate", color="r")
    plt.ylim(0, 20)
    plt.legend()


class SubsampledImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indexes, transform, **kwargs):
        super(SubsampledImageDataset, self).__init__(**kwargs)
        self.dataset = dataset
        self.indexes = indexes
        self.transform = transform

    def __getitem__(self, i):
        img, label = self.dataset[self.indexes[i]]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.indexes)


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def obscured_imgs(img, boxsize=8, bsz=64, stride=4, grey_val = 0.51):
    h, w, _ = img.shape
    batches = []

    for i in range(0, h - boxsize, stride):
      for j in range(0, w - boxsize, stride):
        img_cp = img.copy()
        img_cp[i:i+boxsize, j:j+boxsize, :] = grey_val
        batches.append(img_cp)
    
    for i in range(0, len(batches), bsz):
        yield np.array(batches[i: min(i + bsz, len(batches))])


def generate_prob_heatmaps(vgg, img, id, x_dim, y_dim, gray_value=0.255, cuda=False):
  heatmap = np.zeros((x_dim[1] - x_dim[0], y_dim[1] - y_dim[0]))
  arg_prob_heatmap = np.zeros((x_dim[1] - x_dim[0], y_dim[1] - y_dim[0]))
  occlusion_iter = OcclusionIterator(img, x_dim=x_dim, y_dim=y_dim, occlusion_size=1, gray_value=gray_value)
  occlusion_iter_obj = iter(occlusion_iter)
  for i in range(y_dim[1] - y_dim[0]):
      for j in range(x_dim[1] - x_dim[0]):
          occluded_img = to_tensor(next(occlusion_iter_obj), cuda=cuda)
          probs = to_np(vgg.probabilities(occluded_img))[0]
          heatmap[i, j] += probs[id]
          arg_prob_heatmap[i, j] += np.argmax(probs)
          # print(np.argmax(probs))
  heatmap /= heatmap.max()
  arg_prob_heatmap /= arg_prob_heatmap.max()
  return heatmap, arg_prob_heatmap


def generate_heatmap(vgg, img, x_dim, y_dim, layer_name, map_index, gray_value=0.255, cuda=False):
    heatmap = np.zeros((x_dim[1] - x_dim[0], y_dim[1] - y_dim[0]))
    occlusion_iter = OcclusionIterator(img, x_dim=x_dim, y_dim=y_dim, occlusion_size=1, gray_value=gray_value)
    occlusion_iter_obj = iter(occlusion_iter)
    for i in range(y_dim[1] - y_dim[0]):
        for j in range(x_dim[1] - x_dim[0]):
            occluded_img = to_tensor(next(occlusion_iter_obj), cuda=cuda)
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
        x = self.preprocess(x)
        activations = []
        for name, layer in zip(self.feature_names, self.features):
            x = layer(x)
            if name in layer_names:
                activations.append(x)
        return activations

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


def to_tensor(x, requires_grad=False, cuda=False):
    x = torch.from_numpy(x)
    if cuda:
        x = x.cuda()
    # return torch.tensor(x, **kwargs)
    if requires_grad:
        return x.clone().contiguous().detach().requires_grad_(True)
    else:
        return x.clone().contiguous().detach()