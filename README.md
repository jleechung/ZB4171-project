# ZB4171-project

## Introduction
Spatially-resolved omics allow single-cell transcriptomic profiling while retaining spatial information of cells in their native microenvironment. Integration of spatial information with gene expression data remains challenging, but could potentially yield significant biological insights into how cells function in their microenvironment. Here, we explore probabilistic generative modeling for jointly modeling expression and spatial information from spatial transcriptomics (ST) data.  

## Code usage

1. Clone the repository.

```
git clone https://github.com/jleechung/ZB4171-project
cd ZB4171-project
```

2. Create a virtual environment.

```
conda create --name project python=3.8
activate project
```

3. Install requirements.

```
pip3 install -r requirements.txt
```

## Running the models

We implement four variants of the VAE with PyTorch and Pyro:

- NBVAE: samples the parameters of the Negative Binomial distribution
- ZINBVAE: samples the parameters of the Zero Inflated Negative Binomial distribution
- LabelVAE: ZINB + conditioned on cell type labels
- CVAE: ZINB + conditioned on spatial neighbors

To run the models,

1. Specify data, model and experiment configurations in `config.yml`
2. Execute with `python main.py`

## Data sources

- MERFISH mouse hypothalamus: https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248
- ST mouse olfactory bulb: https://github.com/JEFworks-Lab/MERINGUE/tree/master/data
- seqFISH+: https://github.com/CaiGroup/seqFISH-PLUS

## Related Work
Probabilistic modeling for scRNA-seq and/or ST:
- [SPICEMIX: Latent variable model for jointly modeling spatial information and expression for cell type inference](https://www.biorxiv.org/content/10.1101/2020.11.29.383067v2)
- [gimVI: A joint model for unpaired scRNA-seq and spatial transcriptomics data for imputation](https://arxiv.org/abs/1905.02269)
- [scVI: Deep generative modeling for scRNA-seq](https://www.nature.com/articles/s41592-018-0229-2)
- [BISCUIT: Normalization and clustering of single-cell data with Dirichlet process mixture model](http://proceedings.mlr.press/v48/prabhakaran16.pdf)

Methods developed for ST:
- [SpatialDE: Identification of spatially variable genes](https://www.nature.com/articles/nmeth.4636)
- [Hidden Markov Random Fields for clustering expression data with spatially-coherent genes](https://www.nature.com/articles/nbt.4260)
- [BayesSpace: resolution enhancement and clustering for ST](https://www.nature.com/articles/s41587-021-00935-2)

Generative modeling for spatial data:
- [Augmenting correlation structures in spatial data using generative models](https://arxiv.org/abs/1905.09796)
