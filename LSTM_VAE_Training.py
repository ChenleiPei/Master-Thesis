import torch
from torch.utils.data import DataLoader, TensorDataset
from LSTM_VAE_Model import LSTMVAE
import argparse
import wandb
from readeq import process_equations
import torch.nn.functional as F
from utils import idx2word
import os
import numpy as np
import time
import json

def create_dataset(padded_inputs, padded_targets, sequence_lengths):
    inputs_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.long)
    dataset = TensorDataset(inputs_tensor, inputs_tensor, lengths_tensor)
    return dataset


def loss_function(recon_x, x, mu, log_var):
    # 重建损失，使用均方误差
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL 散度损失
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld_loss, kld_loss, recon_loss


def train_vae(model, data_loader, epochs, lr, vocab_size, save_model_path, save_latent_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global_step = 0  # define a global step counter

    for epoch in range(epochs):
        model.train()  # switch to training mode
        latent_vectors = []
        epoch_loss = 0

        table = wandb.Table(columns=["Original Expression", "Reconstructed Expression"])

        for batch_idx, (inputs, targets, lengths) in enumerate(data_loader):
            optimizer.zero_grad()

            # Forward pass through the VAE
            outputs, mu, log_var, z = model(inputs, lengths)

            # Compute the VAE loss (reconstruction + KL divergence)
            targets = F.one_hot(targets, num_classes=vocab_size).float()
            loss = loss_function(outputs, targets, mu, log_var)
            # Backpropagation
            loss[0].backward()
            optimizer.step()

            epoch_loss += loss[0].item()

            # save the latent vectors
            latent_vectors.append(z.detach().cpu().numpy())

            wandb.log({"batch_loss": loss[0].item()}, step=global_step)
            global_step += 1

            if batch_idx % 300 == 0:
                model.eval()
                with torch.no_grad():
                    # Reconstruct the expressions
                    reconstructed_expressions = model.reconstruct_expression(outputs, from_indices=False)
                    #print("outputs", outputs)
                    input_expressions = model.reconstruct_expression(inputs, from_indices=True)

                    for i in range(len(reconstructed_expressions)):
                        original_expression = input_expressions[i]
                        reconstructed_expression = reconstructed_expressions[i]

                        print(f"Original Expression {i + 1}: {original_expression}")
                        print(f"Reconstructed Expression {i + 1}: {reconstructed_expression}")

                        table.add_data(original_expression, reconstructed_expression)
                model.train()  # Switch back to training mode

        wandb.log({f"Epoch {epoch} Expressions": table}, step=global_step)

        avg_epoch_loss = epoch_loss / len(data_loader.dataset)
        wandb.log({"avg_epoch_loss": avg_epoch_loss}, step=global_step)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}')
        #wandb save each part from loss
        wandb.log({"reconstruction_loss": loss[2].item()}, step=global_step)
        wandb.log({"kl_divergence_loss": loss[1].item()}, step=global_step)

        model_save_path = os.path.join(save_model_path, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"Model weights saved to {model_save_path}")

        # save the latent vectors
    latent_vectors = np.concatenate(latent_vectors, axis=0)  # 将所有 batch 的潜在向量拼接起来
    latent_vectors_path = os.path.join(save_model_path, 'latent_vectors.npy')
    np.save(latent_vectors_path, latent_vectors)
    print(f"Latent vectors saved to {latent_vectors_path}")



    # use wandb.log_artifact to save the latent vectors
    latent_artifact = wandb.Artifact('latent_vectors', type='latent_space')
    latent_artifact.add_file(latent_vectors_path)
    wandb.log_artifact(latent_artifact)

    # save the encoder weights
    decoder_save_path = os.path.join(save_model_path, 'decoder.pth')
    torch.save(model.decoder.state_dict(), decoder_save_path)
    print(f"Decoder weights saved to {decoder_save_path}")

    # use wandb.log_artifact to save the decoder weights
    decoder_artifact = wandb.Artifact('decoder', type='model')
    decoder_artifact.add_file(decoder_save_path)
    wandb.log_artifact(decoder_artifact)

    # save the final model weights
    final_model_path = os.path.join(save_model_path, 'model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model weights saved to {final_model_path}")

    # use wandb.log_artifact to save the final model weights
    model_artifact = wandb.Artifact('model', type='model')
    model_artifact.add_file(final_model_path)
    wandb.log_artifact(model_artifact)

    print(f"Training finished after {epochs} epochs.")


def main():
    ts = time.strftime('%Y-%b-%d-%H-%M-%S', time.gmtime())

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='equations_10000.txt',
                        help='File containing mathematical expressions')
    parser.add_argument('--embedding_size', type=int, default=256, help="Size of embedding layer")
    parser.add_argument('--hidden_size', type=int, default=256, help="Size of hidden LSTM layer")
    parser.add_argument('--latent_size', type=int, default=2, help="Size of latent space")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument('--save_model_path', type=str, default='LSTMVAE_bin')
    args = parser.parse_args()

    # Initialize WandB
    wandb.init(project="LSTM_VAE_eq10000", name="LSTM_VAE", config=args)
    wandb.config.update(args)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path, exist_ok=True)

    # Use the process_equations function to load and process the data
    alphabet_size, input_sequences, target_sequences, sequence_lengths, vocab = process_equations(args.data_file)

    # Create a PyTorch dataset and dataloader
    dataset = create_dataset(input_sequences, target_sequences, sequence_lengths)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Instantiate the model
    vocab_size = len(vocab)
    model = LSTMVAE(vocab_size=vocab_size, embedding_size=args.embedding_size, hidden_size=args.hidden_size,
                    latent_size=args.latent_size, vocab=vocab)

    #save the vocab
    # 保存词汇表
    vocab_path = os.path.join(save_model_path, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    print(f"Vocabulary saved to {vocab_path}")
    # Train the model
    train_vae(model, data_loader, epochs=args.epochs, lr=args.learning_rate, vocab_size=vocab_size, save_model_path=save_model_path, save_latent_path=save_model_path)

    # Finish WandB
    wandb.finish()


if __name__ == "__main__":
    main()
