
import os
import time
import pickle
import yaml
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.optim import Adam, ClippedAdam
import numpy as np
from dataloader import *
from shutil import copyfile

print('Reading config')

# Run options
with open('config.yaml') as file:
    config = yaml.safe_load(file)

# Model architecture
model_params = config['model_params']
MODEL_NAME = model_params['model_name']
LATENT_DIM = model_params['latent_dim']
HIDDEN_DIM1 = model_params['hidden_dim1']
HIDDEN_DIM2 = model_params['hidden_dim2']
HIDDEN_DIM3 = model_params['hidden_dim3']

# Data
data_params = config['data_params']
COUNTS_PATH = data_params['counts_path']
CENTROIDS_PATH = data_params['centroids_path']
METADATA_PATH = data_params['metadata_path']
N_NEIGHBORS = data_params['n_neighbors']
NORM_METHOD = data_params['norm_method']
SCALE_FACTOR = data_params['scale_factor']
AXES = data_params['centroids'].split(',')

# Experiment parameters
exp_params = config['exp_params']
BATCH_SIZE = exp_params['batch_size']
TRAIN_SPLIT = exp_params['train_split']
LOSS = exp_params['loss']
OPTIMIZER = exp_params['optimizer']
LEARNING_RATE = exp_params['learning_rate']
USE_CUDA = exp_params['use_cuda']
NUM_EPOCHS = exp_params['num_epochs']
TEST_FREQUENCY = exp_params['test_frequency']
SAVE_FREQUENCY = exp_params['save_frequency']
SAVE_DIR = exp_params['save_dir']

# Route results to dir
CURR_TIME = time.strftime("%Y%m%d-%H%M%S")
SAVE_DIR = os.path.join(SAVE_DIR, MODEL_NAME + '_' + CURR_TIME)
os.mkdir(SAVE_DIR)
copyfile('./config.yaml', os.path.join(SAVE_DIR, 'config.yaml'))

if __name__ == "__main__":
    assert pyro.__version__.startswith("1.7.0")

    print('Initializing dataset')
    AXES = None if AXES == 'None' else AXES
    dset = SpatialDataset(
        counts_path=COUNTS_PATH,
        centroids_path=CENTROIDS_PATH,
        metadata_path=METADATA_PATH,
        label='Cell_class',
        n_neighbors=N_NEIGHBORS,
        axes=AXES
    )

    NORM_METHOD = None if NORM_METHOD == 'None' else NORM_METHOD
    if NORM_METHOD is not None:
        print('Normalizing dataset')
        if NORM_METHOD not in ['CountNorm', 'LogNorm']:
            raise ValueError('Invaild normalization method')
        dset.normalize_data(method = NORM_METHOD, scale_factor = SCALE_FACTOR)

    print('Initializing loaders')
    train_loader, val_loader, train_indices, val_indices = data_loader(
        dataset=dset,
        batch_size=BATCH_SIZE,
        train_split=TRAIN_SPLIT
    )

    # clear param store
    pyro.clear_param_store()

    # setup the VAE
    print('Initializing model')
    n_input = dset.n_features
    n_class = dset.n_class

    # vae
    if MODEL_NAME == 'ZINBVAE':
        from models.vae import *
        print('Training ZINB VAE')
        vae = ZINBVAE(
            n_input=n_input,
            z_dim=LATENT_DIM,
            hidden_dim1=HIDDEN_DIM1,
            hidden_dim2=HIDDEN_DIM2,
            hidden_dim3=HIDDEN_DIM3,
            use_cuda=USE_CUDA
        )

    elif MODEL_NAME == 'NBVAE':
        from models.vae import *
        print('Training NB VAE')
        vae = NBVAE(
            n_input=n_input,
            z_dim=LATENT_DIM,
            hidden_dim1=HIDDEN_DIM1,
            hidden_dim2=HIDDEN_DIM2,
            hidden_dim3=HIDDEN_DIM3,
            use_cuda=USE_CUDA
        )

    # conditional vae
    elif MODEL_NAME == 'CVAE':
        from models.cvae import *
        print('Training ZINB CVAE')
        vae = CVAE(
            n_input=n_input,
            z_dim=LATENT_DIM,
            hidden_dim1=HIDDEN_DIM1,
            hidden_dim2=HIDDEN_DIM2,
            hidden_dim3=HIDDEN_DIM3,
            use_cuda=USE_CUDA
        )

    elif MODEL_NAME == 'NBCVAE':
        from models.nbcvae import *
        print('Training NB VAE')
        vae = CVAE(
            n_input=n_input,
            z_dim=LATENT_DIM,
            hidden_dim1=HIDDEN_DIM1,
            hidden_dim2=HIDDEN_DIM2,
            hidden_dim3=HIDDEN_DIM3,
            use_cuda=USE_CUDA
        )

    elif MODEL_NAME == 'LabelVAE':
        from models.labelvae import *
        print('Training LabelVAE')
        vae = LabelVAE(
            n_genes=n_input,
            n_class=n_class,
            z_dim=LATENT_DIM,
            hidden_dim1=HIDDEN_DIM1,
            hidden_dim2=HIDDEN_DIM2,
            hidden_dim3=HIDDEN_DIM3,
            use_cuda=USE_CUDA
        )

    else:
        raise ValueError('Invalid model name')

    print(vae)

    # setup the optimizer
    adam_args = {"lr": LEARNING_RATE}
    if OPTIMIZER == 'Adam':
        optimizer = Adam(adam_args)
    elif OPTIMIZER == 'ClippedAdam':
        optimizer = ClippedAdam(adam_args)

    if LOSS == 'ELBO':
        loss = Trace_ELBO()
    elif LOSS == 'MeanField':
        loss = TraceMeanField_ELBO()

    # setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss=loss)

    train_elbo = []
    val_elbo = []
    # training loop
    print('Commence training')
    for epoch in range(NUM_EPOCHS):

        total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA)
        train_elbo.append(-total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch > 0 and epoch % TEST_FREQUENCY == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(svi, val_loader, use_cuda=USE_CUDA)
            val_elbo.append(-total_epoch_loss_test)
            print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

            prev = val_elbo[-1]
            curr = -total_epoch_loss_test
            if curr > prev:
                break

        if epoch > 0 and epoch % SAVE_FREQUENCY == 0:
            file_name = 'model_epochs{}.pt'.format(epoch)
            torch.save(vae.state_dict(), os.path.join(SAVE_DIR, file_name))

    train_elbo_file = 'train_elbo.pkl'
    val_elbo_file = 'val_elbo.pkl'
    with open(os.path.join(SAVE_DIR, train_elbo_file), "wb") as tfile:
        pickle.dump(train_elbo, tfile)
    with open(os.path.join(SAVE_DIR, val_elbo_file), "wb") as vfile:
        pickle.dump(val_elbo, vfile)
