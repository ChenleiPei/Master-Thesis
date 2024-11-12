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
    global_step = 0  # 定义一个全局步数变量，用于管理 wandb 日志

    for epoch in range(epochs):
        model.train()  # 将模型设置为训练模式
        epoch_loss = 0

        # 创建一个 wandb 表格来记录当前 epoch 的原始和重建表达式
        table = wandb.Table(columns=["Original Expression", "Reconstructed Expression"])

        for batch_idx, (inputs, targets, lengths) in enumerate(data_loader):
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs, lengths)

            # 计算损失
            targets = F.one_hot(targets, num_classes=vocab_size).float()
            loss = criterion(outputs, targets)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            epoch_loss += loss.item()

            # 记录每个 batch 的损失到 wandb，使用全局步数
            wandb.log({"batch_loss": loss.item()}, step=global_step)
            global_step += 1

            # 每隔 30 个 batch 查看一次重建的表达式并添加到表格
            if batch_idx % 30 == 0:
                model.eval()  # 切换到评估模式
                with torch.no_grad():
                    # 使用模型重建表达式
                    reconstructed_expressions = model.reconstruct_expression(outputs, from_indices=False)
                    input_expressions = model.reconstruct_expression(inputs, from_indices=True)

                    # 将原始和重建的表达式添加到 wandb 表格中
                    for i in range(len(reconstructed_expressions)):
                        original_expression = input_expressions[i]
                        reconstructed_expression = reconstructed_expressions[i]

                        # 打印输出
                        print(f"Original Expression {i + 1}: {original_expression}")
                        print(f"Reconstructed Expression {i + 1}: {reconstructed_expression}")

                        # 添加数据到表格中
                        table.add_data(original_expression, reconstructed_expression)
                model.train()  # 切换回训练模式

        # 在每个 epoch 结束时记录一次表格到 wandb
        wandb.log({f"Epoch {epoch} Expressions": table}, step=global_step)

        # 记录 epoch 的平均损失，使用当前的全局步数
        avg_epoch_loss = epoch_loss / len(data_loader)
        wandb.log({"epoch_loss": avg_epoch_loss}, step=global_step)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

    print(f"Training finished after {epochs} epochs.")











def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='equations_10000.txt',
                        help='File containing mathematical expressions')
    parser.add_argument('--embedding_size', type=int, default=256, help="Size of embedding layer")
    parser.add_argument('--hidden_size', type=int, default=256, help="Size of hidden LSTM layer")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for optimizer")
    args = parser.parse_args()

    # Initialize WandB
    wandb.init(project="LSTM_AE", name="LSTM_AE", config=args)
    wandb.config.update(args)


    # use the process_equations function to load and process the data
    alphabet_size, input_sequences, target_sequences, sequence_lengths, vocab = process_equations(args.data_file)

    # create a PyTorch dataset and dataloader
    dataset = create_dataset(input_sequences, target_sequences, sequence_lengths)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # instantiate the model
    vocab_size = len(vocab)
    model = LSTMAutoencoder(vocab_size=vocab_size, embedding_size=args.embedding_size, hidden_size=args.hidden_size, vocab=vocab)

    # train the model
    train_autoencoder(model, data_loader, epochs=args.epochs, lr=args.learning_rate, vocab_size=vocab_size)


    #finish wandb
    wandb.finish()


if __name__ == "__main__":
    main()
