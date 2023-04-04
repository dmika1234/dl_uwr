import os
import time

from IPython.display import clear_output
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np
import random
import copy

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
        train_transform=None,
        norm_threshold=np.inf,
        pruned=False,
        patience_expansion=1.5,
        log_every=100,
        device="cpu",
        verbose=False,
        full_silent=False
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
    velocities = [torch.zeros_like(p) for p in model.parameters()]
    #
    iter_ = 0
    epoch = 0
    best_params = None
    best_val_err = np.inf
    history = {"train_losses": [], "train_errs": [], "val_errs": []}
    if not full_silent:
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
                if train_transform:
                    x = train_transform(x)
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
                        # Fro prunning
                        if pruned:
                            zero_mask = p == 0

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
                        param_grad = epsilon * v + (1-epsilon) * p.grad
                        v[...] = param_grad

                        #
                        # TODO for Problem 1: Set a more sensible learning rule here,
                        #       using your learning rate schedule and momentum
                        p -= alpha * v

                        # Norm Constraint:
                        if "weight" in name:
                            row_norms = torch.norm(p, p=2, dim=1)
                            mask = row_norms >= norm_threshold
                            scaled_rows = p[mask] / row_norms[mask, None]
                            scaled_rows *= norm_threshold
                            p[mask] = scaled_rows

                        if pruned:
                            p *= zero_mask.float()
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
            if not full_silent:
                print("{0}\n{1}\n{0}".format("-" * len(m), m))

    except KeyboardInterrupt:
        pass

    if best_params is not None:
        if not full_silent:
            print("\nLoading best params on validation set (epoch %d)\n" % (best_epoch))
        with torch.no_grad():
            for param, best_param in zip(model.parameters(), best_params):
                param[...] = best_param
    if not full_silent:            
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


def div_schedule(alpha0, iter, threshold=10):
    if iter > threshold:
        alpha = alpha0 / 2
    else: 
        alpha = alpha0
    return alpha


def div_schedule2(alpha0, iter, threshold=10):
  exp = np.floor(( iter - threshold) / threshold + 1)
  alpha = alpha0 / (2 ** exp)
  return alpha


# Misc
def check_if_best(val_err, best_val_err, alpha, epsilon, decay, max_num_epochs, lr_schedule, hidden_neurons, gain):
  if val_err <= best_val_err:
    msg = f"| Old best validation set error bitten!| Old error: {best_val_err * 100.0:.3f}% | New best error: {val_err * 100.0:.3f}% | Saving hyperparameters!|"
    print("{0}\n{1}\n{0}".format("-" * len(msg), msg))
    best_hypers = {'alpha': alpha, 'epsilon': epsilon, 'decay': decay, 'max_num_epochs': max_num_epochs,
                'lr_schedule': lr_schedule, 'hidden_neurons': hidden_neurons, 'gain': gain}
    return best_hypers



class Dropout(torch.nn.Module):
    def __init__(self, dropout_prob):
        super(Dropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x):
        if not self.training:
            return x
        mask = torch.rand(x.shape) > self.dropout_prob
        mask = mask.float().to(x.device)
        x = x * mask / (1 - self.dropout_prob)
        return x


class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x):
        if self.training:
            # Compute the mean and variance along each channel
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
        
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            y = self.gamma * x_hat + self.beta
            
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            y = self.gamma * x_hat + self.beta
        
        return y


def train_model(model, mnist_loaders, alpha, epsilon, lr_schedule, decay,
                 max_num_epochs, train_transform=None, norm_threshold=np.inf, pruned=False, device='cpu'):
    t_start = time.time()
    val_err = SGD(model, mnist_loaders, alpha=alpha, epsilon=epsilon, lr_schedule=lr_schedule,
                   decay=decay, max_num_epochs=max_num_epochs, train_transform=train_transform,
                     norm_threshold=norm_threshold, pruned=pruned, device=device)
    test_err_rate = compute_error_rate(model, mnist_loaders["test"])
    m = (
        f"Test error rate: {test_err_rate * 100.0:.3f}%, "
        f"training took {time.time() - t_start:.0f}s."
    )
    print("{0}\n{1}\n{0}".format("-" * len(m), m))
    return val_err


def hyperparameter_tuner(hyperparams, max_epochs=30, num_trials=100, loaders=None, device='cpu'):
  # Initialize the best validation accuracy and hyperparameters
  best_val_err = 1
  best_hyperparams = None

  # Perform random search
  for i in tqdm(range(num_trials)):
    # Randomly sample hyperparameters
    hyperparam_sample = {k: random.choice(v) for k, v in hyperparams.items()}
    
    # Train the model with the sampled hyperparameters
    model = Model(nn.Linear(28 * 28, hyperparam_sample['num_neurons']),
            nn.ReLU(),
            nn.Linear(hyperparam_sample['num_neurons'], hyperparam_sample['num_neurons']),
            nn.ReLU(),
            nn.Linear(hyperparam_sample['num_neurons'], 10))
    model.init_params_xavier(gain=hyperparam_sample['gain'])
    val_err = SGD(model, loaders, alpha=hyperparam_sample['lr'], epsilon=hyperparam_sample['momentum'], lr_schedule=hyperparam_sample['lr_schedule'],
                  decay=hyperparam_sample['decay'], max_num_epochs=max_epochs,
                    norm_threshold=hyperparam_sample['norm_threshold'], device=device, full_silent=True)

      
      # Record the hyperparameters if they gave the best validation accuracy so far
    if val_err < best_val_err:
        best_val_err = val_err
        best_hyperparams = hyperparam_sample
    
    print(f'Trial {i+1} obtained: val_err={100 * val_err:.2f}%, best_val_err={100 * best_val_err:.2f}%')
  return best_val_err, best_hyperparams


def prune_model(model, prune_perc):
    pruned_model = copy.deepcopy(model)
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                qt = torch.quantile(torch.abs(module.weight.data), prune_perc)
                mask = torch.abs(module.weight.data) < qt
                module.weight.data *= mask.float()
                if module.bias is not None:
                    module.bias.data *= mask[:, 0].float()
    return pruned_model

class ELM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights = torch.randn(input_size, hidden_size, requires_grad=False)
        self.biases = torch.randn(hidden_size, requires_grad=False)
        self.output_weights = torch.randn(hidden_size, output_size, requires_grad=False)
        self.init_params_xavier()

    def forward(self, x):
        x = x.view(x.size(0), -1).float()
        hidden = x @ self.weights + self.biases
        output = hidden @ self.output_weights
        return output
    
    def calculate_output_weights(self, x, y):
        x = x.view(x.size(0), -1).float()
        y = y.float()
        hidden = x @ self.weights + self.biases
        xtx = torch.matmul(hidden.t(), hidden)
        self.output_weights[...] = (torch.inverse(xtx) @ hidden.t() @ y).view(self.hidden_size, self.output_size)

    def init_params_xavier(self, gain=np.sqrt(2)):
        with torch.no_grad():
          nn.init.xavier_uniform_(self.weights, gain=gain)
          nn.init.zeros_(self.biases)
