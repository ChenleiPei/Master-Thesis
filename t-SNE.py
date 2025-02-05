import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt
from adjustText import adjust_text  # Import the adjustText library
from matplotlib.font_manager import FontProperties


def load_data(npz_path):
    with np.load(npz_path) as data:
        X = data['latent_vectors']  # Assuming the data key is 'latent_vectors'
        y = data['original_expressions']  # Assuming the label key is 'original_expressions'
    return X, y


def visualize_with_tsne(X, y, perplexity=30, n_iter=1000):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Encode labels to integers

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(30, 15))
    cmap = plt.cm.get_cmap('nipy_spectral', len(np.unique(y_encoded)))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_encoded, cmap=cmap, edgecolor='k', s=50, alpha=0.6)

    # Create a colorbar with proper labels
    cbar = plt.colorbar(scatter)
    cbar.set_ticks(range(len(label_encoder.classes_)))
    cbar.set_ticklabels(label_encoder.classes_)
    font_properties = FontProperties(size=20)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Expression Types',fontproperties=font_properties)

    plt.title('t-SNE visualization of latent space with latent dimension = 16, LSTM layers = 3')
    plt.xlabel('t-SNE-1')
    plt.ylabel('t-SNE-2')

    # Improve the visibility of the labels by adjusting their positions
    adjust_text_labels(X_tsne, y_encoded, label_encoder.classes_)

    plt.show()


def adjust_text_labels(X_tsne, labels, label_classes):

    texts = []
    for i, label in enumerate(np.unique(labels)):
        x_mean = np.mean(X_tsne[labels == label, 0])
        y_mean = np.mean(X_tsne[labels == label, 1])
        texts.append(plt.text(x_mean, y_mean, label_classes[label], ha='center', va='center',
                              fontsize=15,
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='black', alpha=0.5)))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

if __name__ == "__main__":
    # Replace with your file path
    #npz_path = 'LSTMVAE_bin/best-hyper-16-3/latent_vectors_and_expressions.npz'
    npz_path = 'LSTMVAE_bin_real_test/2025-Feb-04-16-55-52/4/3/latent_vectors_and_expressions.npz'
    X, y = load_data(npz_path)
    visualize_with_tsne(X, y, perplexity=40, n_iter=300)

#npz_path = 'LSTMVAE_bin/test for t-SNE/latent_vectors_and_expressions.npz'