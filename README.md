# Grammar Variational Autoencoder with Gaussian Hamiltonian Monte Carlo Sampling

This branch is based on the Grammar-VAE paper available [here](https://arxiv.org/abs/1703.01925) and based on the implementation work of Tim Schneider available [here](https://github.com/TimPhillip/ac_grammar_vae).

This implementation builds a pipeline to perform Gaussian Hamiltonian Monte Carlo (GHMC) sampling in the learned latent space of a Grammar VAE, designed for generating valid syntactic structures in a defined language domain.

## Requirements

Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt
```

## Create Dataset

### Expressions Dataset
To create the expressions dataset, navigate to the `experiments` folder and run:

```bash
python Create_dataset.py
```

### Generate Observed Data Points
To create the observed data points, navigate to the `experiments` folder and run:

```bash
python generate_datapoints.py
```

## Training
To train the Grammar VAE, navigate to the `experiments` folder and run:

```bash
python training.py
```

## Sampling
To perform GP-HMC sampling in the learned latent space, navigate to the `experiments` folder and run:

```bash
python GPHMC_GVAE.py
```

## Visualization
Besides the above steps, `visualize_latent_space.py` provides the capability to visualize the latent space representation. You can run:

```bash
python visualize_latent_space.py
```
