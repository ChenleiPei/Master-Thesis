import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import LabelEncoder

def visualize_latent_space_2d(npz_file):

    data = np.load(npz_file)
    latent_vectors = data["latent_vectors"]
    original_expressions = data["original_expressions"]

    print(f"Loaded latent vectors of shape: {latent_vectors.shape}")
    print(f"Loaded original expressions of shape: {original_expressions.shape}")

    # check if latent_vectors is two-dimensional
    if latent_vectors.shape[1] != 2:
        raise ValueError("Latent space data must be two-dimensional for direct visualization.")


    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(original_expressions)
    unique_labels = label_encoder.classes_
    num_labels = len(unique_labels)

    print(f"Found {num_labels} unique original expressions.")
    print(f"Labels: {unique_labels}")

    # use a colormap to assign colors to each label
    colormap = plt.cm.get_cmap("tab10", num_labels)


    plt.figure(figsize=(12, 8))
    for label in range(num_labels):
        mask = labels == label
        plt.scatter(
            latent_vectors[mask, 0],
            latent_vectors[mask, 1],
            alpha=0.6,
            label=unique_labels[label],
            color=colormap(label)
        )


    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title('2D Latent Space Visualization by Original Expression with beta=')
    plt.legend(loc='best', fontsize='small', title='Original Expressions')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize 2D latent space from an .npz file')
    #parser.add_argument('--npz_file', type=str, help='Path to the .npz file containing latent vectors and original expressions', default='LSTMVAE_bin/2024-Dec-04-20-23-51/latent_vectors_and_expressions.npz')
    parser.add_argument('--npz_file', type=str,
                        help='Path to the .npz file containing latent vectors and original expressions',
                        default='LSTMVAE_bin_real_test/2025-Feb-04-14-06-25/2/3/latent_vectors_and_expressions.npz')
    #parser.add_argument('--npz_file', type=str, help='Path to the .npz file containing latent vectors and original expressions', default='from_ac_grammar_vae/latent_space_parametric_3.npz')
    args = parser.parse_args()
    print("NPZ File:", args.npz_file)

    visualize_latent_space_2d(args.npz_file)


