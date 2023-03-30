import os
import time

from IPython.display import clear_output
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.datasets
from torch import nn


def compute_error_rate(model, data_loader, device="cpu"):
    """Evaluate model on all samples from the data loader.
    """
    # Put the model in eval mode, and move to the evaluation device.
    model.eval()
    model.to(device)
    if isinstance(data_loader, InMemDataLoader):
        data_loader.to(device)

    num_errs = 0.0
    num_examples = 0
    # we don't need gradient during eval!
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model.forward(x)
            _, predictions = outputs.data.max(dim=1)
            num_errs += (predictions != y.data).sum().item()
            num_examples += x.size(0)
    return num_errs / num_examples


def plot_history(history):
    """Helper to plot the trainig progress over time."""
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    train_loss = np.array(history["train_losses"])
    plt.semilogy(np.arange(train_loss.shape[0]), train_loss, label="batch train loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    train_errs = np.array(history["train_errs"])
    plt.plot(np.arange(train_errs.shape[0]), train_errs, label="batch train error rate")
    val_errs = np.array(history["val_errs"])
    plt.plot(val_errs[:, 0], val_errs[:, 1], label="validation error rate", color="r")
    plt.ylim(0, 0.20)
    plt.legend()


class InMemDataLoader(object):
    """
    A data loader that keeps all data in CPU or GPU memory.
    """

    __initialized = False

    def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            drop_last=False,
    ):
        """A torch dataloader that fetches data from memory."""
        batches = []
        for i in tqdm(range(len(dataset))):
            batch = [torch.tensor(t) for t in dataset[i]]
            batches.append(batch)
        tensors = [torch.stack(ts) for ts in zip(*batches)]
        dataset = torch.utils.data.TensorDataset(*tensors)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler option is mutually exclusive "
                    "with batch_size, shuffle, sampler, and "
                    "drop_last"
                )
            self.batch_size = None
            self.drop_last = None

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with " "shuffle")

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.RandomSampler(dataset)
                else:
                    sampler = torch.utils.data.SequentialSampler(dataset)
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size, drop_last
            )

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.__initialized = True

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ("batch_size", "sampler", "drop_last"):
            raise ValueError(
                "{} attribute should not be set after {} is "
                "initialized".format(attr, self.__class__.__name__)
            )

        super(InMemDataLoader, self).__setattr__(attr, val)

    def __iter__(self):
        for batch_indices in self.batch_sampler:
            yield self.dataset[batch_indices]

    def __len__(self):
        return len(self.batch_sampler)

    def to(self, device):
        self.dataset.tensors = tuple(t.to(device) for t in self.dataset.tensors)
        return self


def l2_norm(w):
    return (w ** 2).sum() / 2


def create_mnist_loaders(batch_size=128, data_path="./data", download=True):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    _test = torchvision.datasets.MNIST(
        data_path, train=False, download=download, transform=transform
    )

    # Load training data, split into train and valid sets
    _train = torchvision.datasets.MNIST(
        data_path, train=True, download=download, transform=transform
    )
    _train.data = _train.data[:50000]
    _train.targets = _train.targets[:50000]

    _valid = torchvision.datasets.MNIST(
        data_path, train=True, download=download, transform=transform
    )
    _valid.data = _valid.data[50000:]
    _valid.targets = _valid.targets[50000:]

    mnist_loaders = {
        "train": InMemDataLoader(_train, batch_size=batch_size, shuffle=True),
        "valid": InMemDataLoader(_valid, batch_size=batch_size, shuffle=False),
        "test": InMemDataLoader(_test, batch_size=batch_size, shuffle=False),
    }

    return mnist_loaders


##############################################################################################


def SGD(
        model,
        data_loaders,
        alpha=1e-4,
        epsilon=0.0,
        lr_schedule=(None, None),
        decay=0.0,
        num_epochs=1,
        max_num_epochs=np.nan,
        patience_expansion=1.5,
        log_every=100,
        device="cpu",
        verbose=False
):
    alpha0 = alpha
    lr_schedule, lr_schedule_type = lr_schedule
    # Put the model in train mode, and move to the evaluation device.
    model.train()
    model.to(device)
    for data_loader in data_loaders.values():
        if isinstance(data_loader, InMemDataLoader):
            data_loader.to(device)

    #
    # TODO for Problem 1.3: Initialize momentum variables
    # Hint: You need one velocity matrix for each parameter
    velocities = [None for _ in model.parameters()]
    #
    iter_ = 0
    epoch = 0
    best_params = None
    best_val_err = np.inf
    history = {"train_losses": [], "train_errs": [], "val_errs": []}
    print("Training the model!")
    print("Interrupt at any time to evaluate the best validation model so far.")
    try:
        tstart = time.time()
        siter = iter_
        while epoch < num_epochs:
            model.train()
            epoch += 1
            if epoch > max_num_epochs:
                break
            #
            # TODO: You can implement learning rate control here (it is updated
            # once per epoch), or below in the loop over minibatches.
            if lr_schedule_type == "epochs":
                alpha = lr_schedule(alpha0, epoch)

            for x, y in data_loaders["train"]:
                x = x.to(device)
                y = y.to(device)
                iter_ += 1
                # This calls the `forward` function: https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html
                out = model(x)
                loss = model.loss(out, y)
                loss.backward()
                _, predictions = out.max(dim=1)
                batch_err_rate = (predictions != y).sum().item() / out.size(0)

                history["train_losses"].append(loss.item())
                history["train_errs"].append(batch_err_rate)

                # disable gradient computations - we do not want torch to
                # backpropagate through the gradient application!
                with torch.no_grad():
                    for (name, p), v in zip(model.named_parameters(), velocities):
                        if "weight" in name:
                            #
                            # TODO for Problem 1.3: Implement weight decay (L2 regularization
                            # on weights by changing the gradients
                            p.grad += decay * p


                        #
                        # TODO for Problem 1.2: Implement a learning rate schedule
                        # Hint: You can use the iteration or epoch counters
                        if lr_schedule_type == "batch_iters":
                            alpha = lr_schedule(alpha0, iter_)

                        #
                        # TODO for Problem 1.1: If needed, implement here a momentum schedule
                        # epsilon = TODO
                        #

                        #
                        # TODO for Problem 1.1: Implement velocity updates for momentum
                        # lease make sure to modify the contents of v, not the v pointer!!!
                        if v is None:
                            param_grad = -alpha * p.grad
                        else:
                            param_grad = epsilon * v - alpha * p.grad
                            v[...] = param_grad

                        #
                        # TODO for Problem 1: Set a more sensible learning rule here,
                        #       using your learning rate schedule and momentum
                        #
                        p += param_grad

                        # Zero gradients for the next iteration
                        p.grad.zero_()

                if iter_ % log_every == 0 and verbose:
                    num_iter = iter_ - siter + 1
                    print(
                        "Minibatch {0: >6}  | loss {1: >5.2f} | err rate {2: >5.2f}%, steps/s {3: >5.2f}".format(
                            iter_,
                            loss.item(),
                            batch_err_rate * 100.0,
                            num_iter / (time.time() - tstart),
                        )
                    )
                    tstart = time.time()

            val_err_rate = compute_error_rate(model, data_loaders["valid"], device)
            history["val_errs"].append((iter_, val_err_rate))

            if val_err_rate < best_val_err:
                # Adjust num of epochs
                num_epochs = int(np.maximum(num_epochs, epoch * patience_expansion + 1))
                best_epoch = epoch
                best_val_err = val_err_rate
                best_params = [p.detach().cpu() for p in model.parameters()]
            clear_output(True)
            os.system('cls')
            m = "After epoch {0: >2} | valid err rate: {1: >5.2f}% | doing {2: >3} epochs".format(
                epoch, val_err_rate * 100.0, num_epochs
            )
            print("{0}\n{1}\n{0}".format("-" * len(m), m))

    except KeyboardInterrupt:
        pass

    if best_params is not None:
        print("\nLoading best params on validation set (epoch %d)\n" % (best_epoch))
        with torch.no_grad():
            for param, best_param in zip(model.parameters(), best_params):
                param[...] = best_param
    plot_history(history)
    return val_err_rate

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        self.layers = nn.Sequential(*args, **kwargs)

    def init_params_norm(self, mean=0, sd=0.5):
        with torch.no_grad():
            # Initialize parameters
            for name, p in self.named_parameters():
                if "weight" in name:
                    p.normal_(mean, sd)
                elif "bias" in name:
                    p.zero_()
                else:
                    raise ValueError('Unknown parameter name "%s"' % name)
                
    # def init_params_xavier(self, gain=1):
    #     with torch.no_grad():
    #         # Initialize parameters
    #         for name, p in self.named_parameters():
    #             if "weight" in name:
    #                 fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(p)
    #                 std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    #                 a = np.sqrt(3.0) * std
    #                 nn.init.uniform_(p, -a, a)
    #             elif "bias" in name:
    #                 nn.init.zeros_(p)
    
    def init_params_xavier(self, gain=np.sqrt(2)):
        with torch.no_grad():
            # Initialize parameters
            for name, p in self.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(p, gain=gain)
                elif "bias" in name:
                    nn.init.zeros_(p)

    def forward(self, X):
        X = X.view(X.size(0), -1)
        return self.layers.forward(X)

    def loss(self, Out, Targets):
        return F.cross_entropy(Out, Targets)


# Rate schedule
def exp_schedule(alpha0, iter, beta=0.9, warmups=5):
    if iter <= warmups:
        alpha = iter * alpha0
    elif iter > warmups:
        alpha = alpha0 * beta ** (iter - warmups)
    return alpha


def batch_schedule(alpha0, iter, threshold=10000):
    if iter > threshold:
        alpha = alpha0 / 2
    else: 
        alpha = alpha0
    return alpha
