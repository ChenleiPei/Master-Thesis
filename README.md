## Master Thesis

## Bayesian Reasoning in the learn structure latent space, with Character Variational Autoencoder and Gaussian Hamiltonian Monte Carlo sampling

This branch, the master branch, is the main method that we used for the thesis.
Besides this branch, we have three other branches:
Toy-Projects-of-GPHMC-and-VAE for another pipeline, which shows the first try of building the separate methods;
Character VAE-with-GPHM,C which is using the same method as this master branch. It has a better performance for this problem;
GVAE-with-GPHMC, which uses Grammar VAE instead of Character VAE.


This implementation builds a pipeline to perform Gaussian Hamiltonian Monte Carlo (GHMC) sampling in the learned latent space of a Character VAE while applying Bayesian Reasoning in the latent space.
The expressions datasets and data points datasets are in the folder "data."
Some of the pre-trained models are in the LSTMVAE_bin folder, including the model itself, vocabulary, and latent space files.

## Requirements

Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt
```

## Create Dataset

The Data file includes all expressions and data points dataset that we used for this thesis

### Expressions Dataset
To create the expressions dataset, run:

```bash
python generate_dataset.py
```

### Generate Observed Data Points
To create the observed data points, run:

```bash
python generate_datapoints.py
```

## Training
To train the Character VAE, navigate to the `experiments` folder and run:

```bash
python LSTM_VAE_Training.py
```

## Sampling
To perform GP-HMC sampling in the learned latent space, run:

```bash
python Uniform_ini_GPHMC.py
```
or
```bash
python Real_test_GPHMC.py
```
for the test part in the experiment in the thesis.

## Prior knowledge
To transfer the prior knowledge to the latent space, first run (with a trained VAE)
```bash
prior knowledge to latent space.py
```
to get the parameters for the latent space, then run
```bash
Prior_latent.py or GMM_with_prior.py
```
to visualise the prior in the latent space

## Visualization
Besides the above steps, `visualize_latent_space.py` provides the capability to visualize the latent space representation. You can run:

```bash
python visualize_latent_space.py
```
and
```bash
t-SNE.py
```
for the visualization of the high dimensional latent space.

## Other methods
The implementation of other methods are 
```bash
Genetic Program.py
```
```bash
SinDy.py
```

## Other files
During this thesis project, the other files are also generated:
To improve our method with constant estimation:
```bash
constant estimation.py
```
Autoencoder method is also be tried at the beginning:
```bash
LSTM_AE_Training.py
```
To show the generate distrition in the latent space:
```bash
Uniform_ini_GPHMC.py
```
