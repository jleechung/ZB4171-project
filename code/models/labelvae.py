
## Semi-supervised Variational Autoencoder Model
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim1, hidden_dim2, hidden_dim3, n_genes, n_class):
        super().__init__()
        self.n_input = n_genes + n_class
        self.fc1 = nn.Linear(self.n_input, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.mean_encoder = nn.Linear(hidden_dim3, z_dim)
        self.var_encoder = nn.Linear(hidden_dim3, z_dim)
        # setup the non-linearities
        self.relu = nn.ReLU()

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=2)
        xy = xy.reshape(-1, self.n_input)
        hidden = self.relu(self.fc1(xy))
        hidden = self.relu(self.fc2(hidden))
        hidden = self.relu(self.fc3(hidden))
        mean = self.mean_encoder(hidden)
        var = torch.exp(self.var_encoder(hidden))
        return mean, var

class ZINBDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim1, hidden_dim2, hidden_dim3, n_genes, n_class):
        super().__init__()
        self.n_class = n_class
        self.z_dim = z_dim + n_class
        self.fc1 = nn.Linear(self.z_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.scale = nn.Linear(hidden_dim3, n_genes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.log_theta = torch.nn.Parameter(torch.randn(n_genes))
        self.dropout = nn.Linear(hidden_dim3, n_genes)

    def forward(self, z, y):
        y = y.reshape(-1, self.n_class)
        zy = torch.cat([z,y], dim=1)
        hidden = self.relu(self.fc1(zy))
        hidden = self.relu(self.fc2(hidden))
        hidden = self.relu(self.fc3(hidden))
        scale = self.softmax(self.scale(hidden))
        theta = torch.exp(self.log_theta)
        dropout = self.dropout(hidden)
        return scale, theta, dropout


class LabelVAE(nn.Module):
    def __init__(self, n_genes, n_class, z_dim=10, hidden_dim1=128, hidden_dim2=128, hidden_dim3=128, use_cuda=False):
        super().__init__()
        self.n_genes = n_genes
        self.encoder = Encoder(z_dim, hidden_dim1, hidden_dim2, hidden_dim3, n_genes, n_class)
        self.decoder = ZINBDecoder(z_dim, hidden_dim1, hidden_dim2, hidden_dim3, n_genes, n_class)

        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x, y):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_mean = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_var = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_mean, z_var).to_event(1))
            # decode the latent code z
            scale, theta, dropout = self.decoder(z, y)
            rate = scale * x.sum(dim=2)
            nb_logits = (rate + 1e-4).log() - (theta + 1e-4).log()
            pyro.sample("obs", dist.ZeroInflatedNegativeBinomial(
                total_count=theta,
                logits=nb_logits,
                gate_logits=dropout).to_event(1), obs=x.reshape(-1, self.n_genes))

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, y):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_mean, z_var = self.encoder(x, y)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_mean, z_var).to_event(1))

def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for sample in train_loader:
        # if on GPU put mini-batch into CUDA memory
        x = sample['cell_counts'].float()
        y = sample['cell_labels'].float()
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x, y)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for sample in test_loader:
        # if on GPU put mini-batch into CUDA memory
        x = sample['cell_counts'].float()
        y = sample['cell_labels'].float()
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x, y)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test
