
import os
import pandas as pd
import numpy as np
import pickle as pkl
import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

## temp code
def get_latent(model, model_name, data_loader, n_neighbors=20, file = None):
    latent = []
    sample_id = []
    cell_type = []
    cell_locs = []
    for idx, sample in enumerate(data_loader):

        if model_name == 'LabelVAE':
            x = sample['cell_counts'].float()
            y = sample['cell_labels'].float()
            latent += [model.encoder(x,y)[0].cpu()]

        elif model_name == 'NeighborVAE':
            x = sample['cell_counts'].float()
            y = sample['neighbor_counts'].float()
            latent += [model.encoder(x,y)[0].cpu()]

        elif model_name == 'ZINBCVAE':
            x = sample['neighbor_counts'].float()
            y = sample['cell_counts'].float()
            y = y.expand(-1, n_neighbors, -1)
            latent += [model.prior(x,y)[0].cpu()]

        else:
            x = sample['cell_counts'].float()
            latent += [model.encoder(x)[0].cpu()]

        sample_id.append(sample['sample_id'])
        cell_type.extend(sample['cell_type'])
        cell_locs.append(sample['cell_locs'])

    latent=torch.cat(latent).detach().numpy()
    sample_id=torch.cat(sample_id).numpy()
    cell_type=np.array(cell_type)
    cell_locs=torch.cat(cell_locs).numpy()

    latent = latent[np.argsort(sample_id)]
    if file is not None:
        df = pd.DataFrame(latent)
        df.to_csv(file, index = False)
    return latent

def get_recon(model, model_name, data_loader, distribution, file=None, seed=42):
    pyro.set_rng_seed(seed)
    recon = []
    sample_id = []
    for idx, sample in enumerate(data_loader):
        sample_id.append(sample['sample_id'])
        x = sample['cell_counts'].float()
        # Encode
        if model_name == 'LabelVAE':
            y = sample['cell_labels'].float()
            z_mean, z_var = model.encoder(x,y)
        elif model_name == 'NeighborVAE':
            y = sample['neighbor_counts'].float()
            z_mean, z_var = model.encoder(x,y)
        else:
            z_mean, z_var = model.encoder(x)
        # Sample from encoder
        z = pyro.sample('', dist.Normal(z_mean, z_var))
        # Decode
        if model_name == 'LabelVAE' or model_name == 'NeighborVAE':
            scale, theta, dropout = model.decoder(z,y)
        else:
            if distribution == 'NB':
                scale, theta = model.decoder(z)
            elif distribution == 'ZINB':
                scale, theta, dropout = model.decoder(z)
        rate = scale * x.sum(dim=2)
        nb_logits = (rate + 1e-4).log() - (theta + 1e-4).log()
        # Sample
        if distribution == 'NB':
            xhat = pyro.sample('', dist.NegativeBinomial(
                total_count=theta,
                logits=nb_logits))
        elif distribution == 'ZINB':
            xhat = pyro.sample('', dist.ZeroInflatedNegativeBinomial(
                total_count=theta,
                logits=nb_logits,
                gate_logits=dropout))
        recon.append(xhat)

    counts = torch.cat(recon).detach().numpy()
    sample_id = torch.cat(sample_id).numpy()
    counts = counts[np.argsort(sample_id)]
    if file is not None:
        df = pd.DataFrame(counts)
        df.to_csv(file, index = False)
    return counts

def cvae_evaluation(cvae, data_loader, out_dir=None, seed=42):
    pyro.set_rng_seed(seed)
    counts = []
    latent = []
    recon = []
    sample_id = []

    for idx, sample in enumerate(data_loader):
        x = sample['neighbor_counts'].float()
        n_neighbors = x.shape[1]
        n_input = x.shape[2]
        y = sample['cell_counts'].float()
        counts.append(y[-1].mean(dim=0).numpy())
        y = y.expand(-1, n_neighbors, -1)
        z_mean, z_var = cvae.prior(x, y)
        latent.append(z_mean.mean(dim=0).detach().numpy())
        z = pyro.sample("", dist.Normal(z_mean, z_var))
        # decode the latent code z
        scale, theta, dropout = cvae.generation(z)
        rate = scale * x.reshape(-1,n_input).sum(dim=1)[:,None]
        nb_logits = (rate + 1e-4).log() - (theta + 1e-4).log()
        yhat = pyro.sample("", dist.ZeroInflatedNegativeBinomial(
                    total_count=theta,
                    logits=nb_logits,
                    gate_logits=dropout))
        yhat = yhat.mean(dim=0).numpy()
        recon.append(yhat)
        sample_id.append(sample['sample_id'].numpy())

    recon = np.array(recon)
    latent = np.array(latent)
    sample_id = np.array(sample_id).flatten()
    recon = recon[np.argsort(sample_id)]
    latent = latent[np.argsort(sample_id)]
    if dir is not None:
        latent_path = os.path.join(out_dir, 'latent.csv')
        df1 = pd.DataFrame(latent)
        df1.to_csv(latent_path, index = False)
        recon_path = os.path.join(out_dir, 'recon.csv')
        df2 = pd.DataFrame(recon)
        df2.to_csv(recon_path, index = False)
    return dict(latent=latent, recon=recon)

def load_pkl(file):
    with open(file, "rb") as f:
        x = pkl.load(f)
    return x

CELL_TYPE = [
    'Ambiguous',
    'OD Mature 2',
    'Endothelial 3',
    'Endothelial 2',
    'Astrocyte',
    'OD Mature 3',
    'OD Immature 1',
    'Pericytes',
    'Microglia',
    'Excitatory',
    'OD Mature 4',
    'Ependymal',
    'Inhibitory',
    'OD Mature 1',
    'OD Immature 2',
    'Endothelial 1'
    ]

PALETTE = [
    '#F2F3F4',
    '#222222',
    '#F3C300',
    '#875692',
    '#F38400',
    '#A1CAF1',
    '#BE0032',
    '#C2B280',
    '#848482',
    '#008856',
    '#E68FAC',
    '#0067A5',
    '#F99379',
    '#604E97',
    '#F6A600',
    '#B3446C',
    '#DCD300',
    '#882D17',
    '#8DB600',
    '#654522',
    '#E25822',
    '#2B3D26'
    ]

COLOR_MAP = dict(zip(CELL_TYPE, PALETTE))
