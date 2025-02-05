import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
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
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchinfo import summary
from datetime import datetime



def create_dataset(padded_inputs, padded_targets, sequence_lengths):
    #print("padded_inputs:",padded_inputs)
    inputs_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    #print("inputs_tensor:",inputs_tensor)
    lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.long)
    dataset = TensorDataset(inputs_tensor, inputs_tensor, lengths_tensor)
    return dataset


'''def loss_function(recon_x, x, mu, log_var, vocab_size):
    recon_loss = F.cross_entropy(recon_x.view(-1, vocab_size), x.view(-1), reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld_loss, kld_loss, recon_loss'''

def loss_function(recon_x, x, mu, log_var, vocab_size, beta):
    """
    Compute the loss function for the VAE
    """

    #batch_size = x.size(0)

    # Reconstruction loss using CrossEntropy
    print("beta:",beta)
    recon_loss = F.cross_entropy(recon_x.view(-1, vocab_size), x.view(-1), reduction='sum')
    #recon_loss = recon_loss / batch_size

    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    #recon_loss = recon_loss / batch_size

    # Total loss with beta annealing
    total_loss = recon_loss + 0.3*beta * kld_loss

    return total_loss, kld_loss, recon_loss


def init_lstm_weights(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name: # input weight
                nn.init.kaiming_uniform_(param)
            elif 'weight_hh' in name:  # hidden weight
                nn.init.kaiming_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

                param.data[param.size(0) // 4 : param.size(0) // 2].fill_(1.0)



def train_vae(model, train_loader, validation_loader, test_loader, epochs, lr, vocab_size, save_model_path, save_latent_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
                              #centered=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=5, verbose=True)
    early_stop_step = 0  # define a global step counter

    for epoch in range(epochs):
        model.train()  # switch to training mode
        latent_vectors = []
        epoch_loss = 0
        epoch_kl_loss = 0
        epoch_recon_loss = 0

        table = wandb.Table(columns=["Original Expression", "Reconstructed Expression"])

        for batch_idx, (inputs, targets, lengths) in enumerate(train_loader):

            #print the number of samples in train_loader
            print("len(train_loader):",len(train_loader.dataset))
            #print the number of samples in validation_loader
            print("len(validation_loader):",len(validation_loader.dataset))

            optimizer.zero_grad()

            # Forward pass through the VAE
            logits, predicted_tokens, mu, log_var, z = model(inputs, lengths)
            #print("outputs.shape:",outputs.shape)
            #print("z.shape:",z.shape)

            # Compute the VAE loss (reconstruction + KL divergence)
            #targets = F.one_hot(targets, num_classes=vocab_size).float()
            #print("target.shape:",targets.shape)
            #print("outputs.shape:",outputs.shape)
            #define the total steps
            print("epochs:",epochs)
            print("len(train_loader):",len(train_loader))
            total_steps = len(train_loader) * epochs
            print("total_steps:",total_steps)
            #define current step
            current_steps = batch_idx + epoch * len(train_loader)
            print("current_steps:",current_steps)
            beta = min(1.0, current_steps / total_steps)
            loss = loss_function(logits, targets, mu, log_var, vocab_size, beta)
            # Backpropagation
            loss[0].backward()
            optimizer.step()

            """for name, param in model.named_parameters():
                if param.grad is not None:
                    wandb.log({
                        f"grad_mean/{name}": param.grad.mean().item(),
                        f"grad_norm/{name}": param.grad.norm().item()
                    })"""
            print("loss[0].item():",loss[0].item())
            epoch_loss += loss[0].item()
            epoch_kl_loss += loss[1].item()
            epoch_recon_loss += loss[2].item()

            # save the result every 10000 batchs
            if batch_idx % 10000 == 0:
                #print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss[0].item():.4f}")

                # Reconstruct the expressions
                reconstructed_expressions = model.reconstruct_expression(logits, from_indices=False)
                input_expressions = model.reconstruct_expression(inputs, from_indices=True)

                for i in range(len(reconstructed_expressions)):
                    original_expression = input_expressions[i]
                    reconstructed_expression = reconstructed_expressions[i]

                    print(f"Original Expression {i + 1}: {original_expression}")
                    print(f"Reconstructed Expression {i + 1}: {reconstructed_expression}")

                    table.add_data(original_expression, reconstructed_expression)

            # save the latent vectors
            latent_vectors.append(z.detach().cpu().numpy())

            #add early stop here, stop if 5 continous epoch have loss less than 2
            if loss[0].item() < 2:
                early_stop_step += 1
                if early_stop_step > 5:
                    break

            #wandb.log({"batch_loss": loss[0].item()}, step=global_step)
            #global_step += 1

        #wandb.log({f"Epoch {epoch} Expressions": table}, step=global_step)
        gradients = {name: param.grad.norm().item() for name, param in model.named_parameters() if
                     param.grad is not None}
        #wandb.log({"epoch": epoch, **gradients})

        # Calculate the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        print("length of train_loader.dataset:",len(train_loader.dataset))
        avg_epoch_kl_loss = epoch_kl_loss / len(train_loader.dataset)
        avg_epoch_recon_loss = epoch_recon_loss / len(train_loader.dataset)

        #evluate on the validation set after each epoch
        model.eval()
        val_loss = 0
        val_kl_loss = 0
        val_recon_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, lengths) in enumerate(validation_loader):
                logits,predicted_tokens, mu, log_var, z = model(inputs, lengths)
                loss_val = loss_function(logits, targets, mu, log_var, vocab_size, beta)
                val_loss += loss_val[0].item()
                val_kl_loss += loss_val[1].item()
                val_recon_loss += loss_val[2].item()

        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_kl_loss = val_kl_loss / len(test_loader.dataset)
        avg_val_recon_loss = val_recon_loss / len(test_loader.dataset)


        #save it to wandb together with the training loss
        #wandb.log({"avg_epoch_kl_loss": avg_epoch_kl_loss, "avg_val_kl_loss": avg_val_kl_loss}, step=epoch)
        #wandb.log({"avg_epoch_recon_loss": avg_epoch_recon_loss, "avg_val_recon_loss": avg_val_recon_loss}, step=epoch)
        #wandb.log({"avg_epoch_loss": avg_epoch_loss, "avg_val_loss": avg_val_loss}, step=epoch)
        #wandb.log({"avg_epoch_recon_loss": avg_epoch_recon_loss}, step=epoch)
        #wandb.log({"avg_epoch_loss": avg_epoch_loss}, step=epoch)
        wandb.log({
            "Loss/Train": avg_epoch_loss,
            "Loss/Validation": avg_val_loss,
            "KL Loss/Train": avg_epoch_kl_loss,
            "KL Loss/Validation": avg_val_kl_loss,
            "Reconstruction Loss/Train": avg_epoch_recon_loss,
            "Reconstruction Loss/Validation": avg_val_recon_loss
        }, step=epoch)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}')
        # wandb save each part from loss
        #wandb.log({"reconstruction_loss": loss[2].item()}, step=global_step)
        #wandb.log({"kl_divergence_loss": loss[1].item()}, step=global_step)

        model_save_path = os.path.join(save_model_path, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_save_path)
        #print(f"Model weights saved to {model_save_path}")

        #update the learning rate
        scheduler.step(avg_epoch_loss)

    # save the latent vectors
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    latent_vectors_path = os.path.join(save_model_path, 'latent_vectors.npy')
    np.save(latent_vectors_path, latent_vectors)
    #print(f"Latent vectors saved to {latent_vectors_path}")

    # use wandb.log_artifact to save the latent vectors
    latent_artifact = wandb.Artifact('latent_vectors', type='latent_space')
    latent_artifact.add_file(latent_vectors_path)
    wandb.log_artifact(latent_artifact)

    # save the final model weights
    final_model_path = os.path.join(save_model_path, 'model_final.pth')
    torch.save(model, final_model_path)
    #print(f"Final model weights saved to {final_model_path}")

    # use wandb.log_artifact to save the final model weights
    model_artifact = wandb.Artifact('model', type='model')
    model_artifact.add_file(final_model_path)
    wandb.log_artifact(model_artifact)

    #print(f"Training finished after {epochs} epochs.")

    all_latent_vectors = []
    all_original_expressions = []

    # Evaluate the model on the test set
    model.eval()
    test_loss = 0
    kl_loss = 0
    recon_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, lengths) in enumerate(test_loader):

            logits, predicted_tokens, mu, log_var, z = model(inputs, lengths)

            # 重建表达式
            reconstructed_expressions = model.reconstruct_expression(logits, from_indices=False)
            input_expressions = model.reconstruct_expression(inputs, from_indices=True)

            for i in range(len(reconstructed_expressions)):
                original_expression = input_expressions[i]
                latent_representation = z[i].cpu().numpy()

                # save latent vectors 和 original expressions
                all_latent_vectors.append(latent_representation)
                all_original_expressions.append(original_expression)


            loss_test = loss_function(logits, targets, mu, log_var, vocab_size, beta)
            test_loss += loss_test[0].item()
            kl_loss += loss_test[1].item()
            recon_loss += loss_test[2].item()

    # calculate the average test loss
    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_kl_loss = kl_loss / len(test_loader.dataset)
    avg_recon_loss = recon_loss / len(test_loader.dataset)

    print(f"Average Test Loss: {avg_test_loss}")
    print(f"Average KL Loss: {avg_kl_loss}")
    print(f"Average Reconstruction Loss: {avg_recon_loss}")


    latent_vectors_array = np.vstack(all_latent_vectors)
    original_expressions_array = np.array(all_original_expressions)

    # save the latent vectors and original expressions
    npz_file_path = os.path.join(save_model_path, 'latent_vectors_and_expressions.npz')

    np.savez(
        npz_file_path,
        latent_vectors=latent_vectors_array,
        original_expressions=original_expressions_array
    )

def main():
    ts = time.strftime('%Y-%b-%d-%H-%M-%S', time.gmtime())

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='Real_test_dataset.txt',
                        help='File containing mathematical expressions')
    parser.add_argument('--embedding_size', type=int, default=32, help="Size of embedding layer")
    parser.add_argument('--hidden_size', type=int, default=128, help="Size of hidden LSTM layer")
    parser.add_argument('--latent_size', type=int, default=4, help="Size of latent space")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=150, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=0.0005, help="Learning rate for optimizer")
    parser.add_argument('--save_model_path', type=str, default='LSTMVAE_bin_real_test')
    #addd number of layers
    parser.add_argument('--num_layers', type=int, default=3, help="Number of LSTM layers")
    args = parser.parse_args()



    ts = datetime.now().strftime("%Y-%b-%d-%H-%M-%S")

    # folder name for WandB
    folder_name = f"{ts}-LSTM_VAE_Kaiminguni-{args.latent_size}-{args.num_layers}"

    # 实际保存的文件夹路径
    save_model_path = os.path.join(args.save_model_path, ts, str(args.latent_size), str(args.num_layers))

    #save_model_path = os.path.join(args.save_model_path, ts, args.latent_size, args.num_layers)
    os.makedirs(save_model_path, exist_ok=True)

    # Initialize WandB
    wandb.init(project="LSTM_VAE_real_test",
               name=folder_name,
               config=args)

    wandb.config.update(args)

    # print the folder name and save path
    print(f"Folder name for WandB: {folder_name}")
    print(f"Model will be saved in: {save_model_path}")

    # Use the process_equations function to load and process the data
    alphabet_size, input_sequences, target_sequences, sequence_lengths, vocab = process_equations(args.data_file)
    print("vocab:",vocab)
    #print("max sequence length:",max(sequence_lengths))
    max_sequence_length = max(sequence_lengths)

    # Create a PyTorch dataset and dataloader
    #here can choose the target: move one place or not
    dataset = create_dataset(input_sequences, target_sequences, sequence_lengths)

    # Split the dataset into training , validation and test sets
    train_size = int(0.7 * len(dataset))
    validation_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - validation_size
    #train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate the model
    vocab_size = len(vocab)
    model = LSTMVAE(vocab_size=vocab_size, embedding_size=args.embedding_size, hidden_size=args.hidden_size,
                    latent_size=args.latent_size, vocab=vocab, max_length=max_sequence_length, num_layers=args.num_layers)

    # initialize the weights
    model.apply(init_lstm_weights)

    summary(model)

    for name, param in model.named_parameters():
        if 'weight' in name:
            plt.hist(param.data.cpu().numpy().flatten(), bins=50)
            plt.title(f"Weight Distribution: {name}")
            plt.show()

    # Save the vocab
    vocab_path = os.path.join(save_model_path, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    print(f"Vocabulary saved to {vocab_path}")

    #wandb.watch(model, log="gradients", log_freq=100) # log the gradients every 10 steps

    # Train the model
    train_vae(model, train_loader, validation_loader, test_loader, epochs=args.epochs, lr=args.learning_rate, vocab_size=vocab_size, save_model_path=save_model_path, save_latent_path=save_model_path)

    # Finish WandB
    wandb.finish()


if __name__ == "__main__":
    main()

