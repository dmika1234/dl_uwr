
import os
import time
from IPython.display import clear_output
from tqdm.auto import tqdm
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets
from torch import nn
from funs import compute_error_rate, plot_history, create_mnist_loaders, SGD, Model

os.chdir('Assignments/Assignment2')

def exp_schedule(alpha0, iter, beta=0.9, warmups=5):
    if iter <= warmups:
        alpha = iter * alpha0
    elif iter > warmups:
        alpha = alpha0 * beta ** (iter - warmups)
    return alpha

############# Load the data
batch_size = 128
data_path = "./data"
mnist_loaders = create_mnist_loaders(batch_size=batch_size, data_path=data_path, download=False)
############# 

############# Create model
# model = Model(nn.Linear(28 * 28, 50),
#               nn.ReLU(),
#               nn.Linear(50, 10))
model = Model(nn.Linear(28 * 28, 10))
model.init_params_norm()
############# 

# On GPU enabled devices set device='cuda' else set device='cpu'
lr_schedule = partial(exp_schedule, beta=0.9, warmups=6)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t_start = time.time()
SGD(model, mnist_loaders, alpha=1e-1, lr_schedule=lr_schedule, epsilon=0.9, max_num_epochs=30, device=device)


test_err_rate = compute_error_rate(model, mnist_loaders["test"])
m = (
    f"Test error rate: {test_err_rate * 100.0:.3f}%, "
    f"training took {time.time() - t_start:.0f}s."
)
print("{0}\n{1}\n{0}".format("-" * len(m), m))
