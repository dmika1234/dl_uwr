import os
import time
from IPython.display import clear_output
from tqdm.auto import tqdm
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.datasets
from torch import nn
from funs import compute_error_rate, plot_history, create_mnist_loaders, SGD, Model, exp_schedule, batch_schedule

os.chdir('Assignments/Assignment2')

np.random.random(2023)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
############# Load the data
batch_size = 128
data_path = "./data"
mnist_loaders = create_mnist_loaders(batch_size=batch_size, data_path=data_path, download=False)
############# 


############# Create model
model = Model(nn.Linear(28 * 28, 100),
              nn.ReLU(),
              nn.Linear(100, 10))

# model = Model(nn.Linear(28 * 28, 10))
# model.init_params_norm()
model.init_params_xavier()
############# 

# On GPU enabled devices set device='cuda' else set device='cpu'
lr_schedule = (partial(exp_schedule, beta=0.9, warmups=6), "epochs")
# lr_schedule = (batch_schedule, "batch_iters")
# lr_schedule = (None, None)

val_err = SGD(model, mnist_loaders,
    alpha=1e-1, epsilon=0.9, lr_schedule=lr_schedule, decay=0.2,
    max_num_epochs=30, device=device)
##