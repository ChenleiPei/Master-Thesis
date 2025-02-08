# Grammar Variational Autoencoder with Gaussian Hamiltonian Monte Carlo

This branch is based on the Grammar-VAE paper available [here](https://arxiv.org/abs/1703.01925) and based on the implementation work of Tim Schneider available [here](https://github.com/TimPhillip/ac_grammar_vae).

This implementation builds a pipeline to perform Gaussian Hamiltonian Monte Carlo (GHMC) sampling in the learned latent space of a Grammar VAE, designed for generating valid syntactic structures in a defined language domain.

## Requirements

Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt

## Create Datase

Expressions Dataset
To create the expressions dataset, navigate to the experiments folder and run:
```bash
python Create_dataset.py


To create the observed data points,  go to the folder -experiments, run
```bash
generate_datapoints.py

##Training
To train the Grammar VAE, go to the folder -experiments, run
```bash
training.py

##Sampling
To do the GP-HMC sampling in the learned latent space, go to the folder -experiments, run
```bash
GPHMC_GVAE.py

Beside them, 'visualize_latent_space.py' provides the possibility to visualise 2D latent space. 't-SNE.py' provides the possibility to down dimension and visual the high dimension latent space.
