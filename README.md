## Master Thesis

## Bayesian Reasoning in the learn structure latent space, with Character Variational Autoencoder and Gaussian Hamiltonian Monte Carlo sampling

This branch, the master branch, is the main method that we used for the thesis.

This branch is based on the Character-VAE paper available  and based on the implementation work of available .

This implementation builds a pipeline to perform Gaussian Hamiltonian Monte Carlo (GHMC) sampling in the learned latent space of a Character VAE while applying Bayesian Reasoning in the latent space.


## Requirements

Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt
```

## Create Dataset

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
and
```bash
python Real_test_GPHMC.py
```
for the test part in the experiment in the thesis.

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
