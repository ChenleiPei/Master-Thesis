import torch
from ac_grammar_vae.model.gvae.gvae import GrammarVariationalAutoencoder

# Load model
model = torch.load('results/gvae_pretrained_parametric_3.pth')
model.eval()

# Sample a point from the latent space
#z = torch.randn([1, model.latent_dim])
z = torch.tensor([[0,0]], dtype=torch.float32)

# Decode the sampled point to get the expression
expression = model.sample_decoded_grammar(z)
print("Decoded expression:", expression)