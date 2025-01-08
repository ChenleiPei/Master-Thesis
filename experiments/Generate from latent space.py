
from ac_grammar_vae.model.gvae.gvae import GrammarVariationalAutoencoder
import torch
import numpy as np
import torch


# 假设 GrammarVariationalAutoencoder 和其他必需的类都已经在其他文件中定义
from ac_grammar_vae.model.gvae.gvae import GrammarVariationalAutoencoder

# Load the pre-trained model
model_path = 'results/gvae_pretrained_parametric_3.pth'  # 指定模型文件的路径
model = torch.load(model_path, map_location=torch.device('cpu')) # 创建模型实例


# Load the latent space
latent_space_path = 'results/latent_space_parametric_3.npz'  # 指定潜在空间文件的路径
latent_space = np.load(latent_space_path)

# Assuming latent_dim is defined in your model, if not, you should set it manually
latent_dim = model.latent_dim  # 获取模型的潜在维度
print(f"Latent dimension: {latent_dim}")

num_points_per_dim = 4  # 每个维度的点数

# Generate a grid of points in the latent space
latent_points = np.linspace(-3, 3, num_points_per_dim)  # 创建线性空间
#print the logit of latent_points
print("latent_points:", latent_points)
grid = np.meshgrid(*[latent_points] * latent_dim)  # 创建网格
grid = np.stack(grid, axis=-1).reshape(-1, latent_dim)  # 重塑为(n, latent_dim)

# Convert numpy array to torch tensor
latent_points_tensor = torch.tensor(grid, dtype=torch.float32)
print(f"Latent points tensor shape: {latent_points_tensor.shape}")


# Generate samples from the latent space using sample_decoded_grammar
samples = []
with torch.no_grad():
    for point in latent_points_tensor:
        print("shape of point: ", point.shape)
        # remove the first dimension of point
        logits = model.decode(point.unsqueeze(0))
        print(f"Logits shape: {logits.shape}")
        #print(point.unsqueeze(0).shape)
        #print(point.unsqueeze(0))
        sample = model.sample_decoded_grammar(point.unsqueeze(0))
        samples.append(sample)

# 打印所有生成的样本
for sample in samples:
    print(sample)
