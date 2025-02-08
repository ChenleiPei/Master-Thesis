This branch is based on the Grammar-VAE paper https://arxiv.org/abs/1703.01925 and based on the implement work of Tim Schneider https://github.com/TimPhillip/ac_grammar_vae.

This branch build a pipeline to do the Gaussian Harmilton Sampling in the learned latent space, which is trained by Grammar VAE.


##Requirements
Install using pip install -r requirements.txt

##Creating datasets
To create the expressions dataset, go to the folder -experiments, run
```bash
Create_dataset.py

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
