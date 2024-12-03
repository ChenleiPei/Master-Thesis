import torch
from torch.utils.data import DataLoader, TensorDataset
from LSTM_AE_Model import LSTMAutoencoder
import argparse
import wandb
from readeq import process_equations
import torch.nn.functional as F
from utils import idx2word


def create_dataset(padded_inputs, padded_targets, sequence_lengths):
    inputs_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    #targets_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.long)
    dataset = TensorDataset(inputs_tensor, inputs_tensor, lengths_tensor)
    return dataset



def train_autoencoder(model, data_loader, epochs, lr, vocab_size):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global_step = 0  # define a global step counter

    for epoch in range(epochs):
        model.train()  # switch to training mode
        epoch_loss = 0

        # Create a WandB table to store original and reconstructed expressions
        table = wandb.Table(columns=["Original Expression", "Reconstructed Expression"])

        for batch_idx, (inputs, targets, lengths) in enumerate(data_loader):
            optimizer.zero_grad()

            # Forward pass through the model
            outputs = model(inputs, lengths)

            # Convert targets to the appropriate format for MSELoss
            targets = F.one_hot(targets, num_classes=vocab_size).float()

            # Ensure targets and outputs have the same shape for MSE loss
            max_length = outputs.size(1)
            targets = targets[:, :max_length, :]  # Trim or pad targets to match output length if necessary

            # Compute loss
            loss = criterion(outputs, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Accumulate epoch loss
            epoch_loss += loss.item()

            # Log batch loss to WandB
            wandb.log({"batch_loss": loss.item()}, step=global_step)
            global_step += 1

            # Periodically reconstruct expressions for evaluation
            if batch_idx % 300 == 0:
                model.eval()  # switch to evaluation mode
                with torch.no_grad():
                    # Reconstruct the expressions
                    reconstructed_expressions = model.reconstruct_expression(outputs, from_indices=False)
                    input_expressions = model.reconstruct_expression(inputs, from_indices=True)

                    # Add original and reconstructed expressions to the table
                    for i in range(len(reconstructed_expressions)):
                        original_expression = input_expressions[i]
                        reconstructed_expression = reconstructed_expressions[i]

                        print(f"Original Expression {i + 1}: {original_expression}")
                        print(f"Reconstructed Expression {i + 1}: {reconstructed_expression}")

                        table.add_data(original_expression, reconstructed_expression)
                model.train()  # switch back to training mode

        # Log reconstructed expressions at the end of the epoch
        wandb.log({f"Epoch {epoch} Expressions": table}, step=global_step)

        # Compute and log average epoch loss
        avg_epoch_loss = epoch_loss / len(data_loader)
        wandb.log({"avg_epoch_loss": avg_epoch_loss}, step=global_step)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

    print(f"Training finished after {epochs} epochs.")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='equations_10000.txt',
                        help='File containing mathematical expressions')
    parser.add_argument('--embedding_size', type=int, default=256, help="Size of embedding layer")
    parser.add_argument('--hidden_size', type=int, default=256, help="Size of hidden LSTM layer")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for optimizer")

    args = parser.parse_args()

    # Initialize WandB
    wandb.init(project="LSTM_AE_eq100000", name="LSTM_AE", config=args)
    wandb.config.update(args)


    # use the process_equations function to load and process the data
    alphabet_size, input_sequences, target_sequences, sequence_lengths, vocab = process_equations(args.data_file)

    # create a PyTorch dataset and dataloader
    dataset = create_dataset(input_sequences, target_sequences, sequence_lengths)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # instantiate the model
    vocab_size = len(vocab)
    model = LSTMAutoencoder(vocab_size=vocab_size, embedding_size=args.embedding_size, hidden_size=args.hidden_size, vocab=vocab, max_length=max(sequence_lengths))

    # train the model
    train_autoencoder(model, data_loader, epochs=args.epochs, lr=args.learning_rate, vocab_size=vocab_size)


    #finish wandb
    wandb.finish()


if __name__ == "__main__":
    main()
