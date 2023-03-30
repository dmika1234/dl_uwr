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
from funs import compute_error_rate, plot_history, create_mnist_loaders, SGD, Model, exp_schedule, div_schedule, div_schedule2, check_if_best


os.chdir('Assignments/Assignment2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
############# Load the data
batch_size = 128
data_path = "./data"
mnist_loaders = create_mnist_loaders(batch_size=batch_size, data_path=data_path, download=False)
############# 
best_val_err = 1.00
############# Hyperparams setting
alpha = 0.25
epsilon = 0.9
decay=0.0
max_num_epochs = 30
hidden_neurons = 800
gain = 0.1
# lr_schedule = (partial(exp_schedule, beta=0.9, warmups=3), "epochs")
# lr_schedule = (partial(div_schedule, threshold=12), "epochs")
lr_schedule = (partial(div_schedule2, threshold=10), "epochs")
# lr_schedule = (div_schedule, "batch_iters")
# lr_schedule = (None, None)
############# Create model
torch.manual_seed(2137)
model = Model(nn.Linear(28 * 28, hidden_neurons),
              nn.ReLU(),
              nn.Linear(hidden_neurons, hidden_neurons),
              nn.ReLU(),
              nn.Linear(hidden_neurons, 10))
model.init_params_xavier(gain=gain)
############# 

# On GPU enabled devices set device='cuda' else set device='cpu'
lr_schedule = (partial(exp_schedule, beta=0.9, warmups=6), "epochs")
# lr_schedule = (div_schedule, "batch_iters")
# lr_schedule = (None, None)

t_start = time.time()
val_err = SGD(model, mnist_loaders,
    alpha=alpha, epsilon=epsilon, lr_schedule=lr_schedule, decay=decay,
    max_num_epochs=max_num_epochs, device=device)
##
best_hypers = check_if_best(val_err, best_val_err, alpha, epsilon, decay, max_num_epochs,
                            lr_schedule, hidden_neurons, gain)

##
test_err_rate = compute_error_rate(model, mnist_loaders["test"])
m = (
    f"Test error rate: {test_err_rate * 100.0:.3f}%, "
    f"training took {time.time() - t_start:.0f}s."
)
print("{0}\n{1}\n{0}".format("-" * len(m), m))
##
