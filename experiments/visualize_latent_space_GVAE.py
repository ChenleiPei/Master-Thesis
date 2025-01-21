import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.preprocessing import LabelEncoder
from matplotlib import cm
from matplotlib import colormaps

def load_and_plot_latent_space(file_path):
    # Load the latent space data
    data = np.load(file_path)
    all_latent_vectors = data['latent_vectors']
    all_labels = data['labels'].reshape(-1, 32)  # 重塑labels以确保它是二维的

    # 标签编码
    labels_encoded = ['_'.join(map(str, label)) for label in all_labels]
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels_encoded)

    # 为每个唯一标签生成颜色
    unique_labels = np.unique(labels_encoded)
    colors = colormaps.get_cmap('viridis', len(unique_labels))

    # 绘图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_latent_vectors[:, 0], all_latent_vectors[:, 1], c=labels_encoded, cmap=colors, alpha=0.6)
    plt.colorbar(scatter, ticks=range(len(unique_labels)))
    plt.title('Latent Space Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

def load_and_plot_latent_space_with_label(file_path):
    data = np.load(file_path)
    all_latent_vectors = data['latent_vectors']
    all_labels = data['labels']

    # Convert 32-dimensional vectors into string labels
    #string_labels = ['_'.join(map(str, label)) for label in all_labels]
    string_labels = all_labels

    # Use a colormap to generate a color for each unique label
    unique_labels = np.unique(string_labels)
    cmap = plt.get_cmap('viridis')  # Get 'viridis' colormap
    colors = cmap(np.linspace(0, 1, len(unique_labels)))  # Generate a color for each unique label

    # Create a map from labels to colors
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Plotting
    plt.figure(figsize=(12, 10))  # Increase figure size
    for label in unique_labels:
        # Find data points corresponding to the current label
        indices = np.array(string_labels) == label
        plt.scatter(
            all_latent_vectors[indices, 0],
            all_latent_vectors[indices, 1],
            color=color_map[label],
            label=label,  # Use string labels directly
            alpha=0.6
        )
    plt.legend(title="Labels", bbox_to_anchor=(0.9, 1), loc='upper left', fontsize='small')
    plt.title('Latent Space Visualization by Labels')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()



import numpy as np

# Load the data to check its structure
data = np.load('results/latent_space_parametric_3.npz')

labels = data['labels']

# 检查标签的形状和内容
print("Labels shape:", labels.shape)
#print("Labels:", labels)

# Check if the data is an archive of named arrays or a single array
if isinstance(data, np.lib.npyio.NpzFile):  # Check if it's an .npz file
    print("Keys in the .npz file:", data.files)
    for key in data.files:
        print(f"Shape of data under {key}:", data[key].shape)
else:
    print("Data shape:", data.shape)  # It's a single array


# Replace 'path_to_your_file.npz' with the actual path to your .npz file
load_and_plot_latent_space_with_label('results/latent_space_parametric_2025-Jan-08-20-15-26_klloss=0.001_dense=512.npz')


def count_unique_labels(file_path):
    # 加载数据
    data = np.load(file_path)
    all_labels = data['labels']

    # 将32维向量转换为字符串标签
    string_labels = ['_'.join(map(str, label)) for label in all_labels]

    # 使用 np.unique 获取唯一标签
    unique_labels = np.unique(string_labels)
    num_unique_labels = len(unique_labels)

    print(f"Number of unique labels: {num_unique_labels}")
    return unique_labels


# 替换 'results/latent_space_parametric_3.npz' 为你的实际文件路径
unique_labels = count_unique_labels('results/latent_space_parametric_2025-Jan-08-20-15-26_klloss=0.001_dense=512.npz')