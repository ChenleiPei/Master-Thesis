import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import logging
from torch.utils.data import DataLoader
import random
import argparse
import tqdm
from datetime import datetime
import wandb


from ac_grammar_vae.data import CFGEquationDataset
from ac_grammar_vae.data.transforms import MathTokenEmbedding, ToTensor, Compose, PadSequencesToSameLength, GrammarParseTreeEmbedding
from ac_grammar_vae.model.gvae import GrammarVariationalAutoencoder


def setup_dataset(n_samples, n_validation_samples=0, expressions_with_parameters=False):

    training = CFGEquationDataset(n_samples=n_samples, use_original_grammar=not expressions_with_parameters)
    #save the training dataset

    validation = CFGEquationDataset(n_samples=n_validation_samples, use_original_grammar=not expressions_with_parameters) if n_validation_samples > 0 else None

    embedding = GrammarParseTreeEmbedding(training.pcfg, pad_to_length=training.max_grammar_productions)
    print("pcfg:", training.pcfg)
    training.transform = Compose([
            embedding,
            ToTensor(dtype=torch.int64)
        ])  #combine them as one action

    if validation:
        validation.transform = Compose([
            embedding,
            ToTensor(dtype=torch.int64)
        ])

    if not validation:
        return training, embedding
    else:
        return training, validation, embedding

def setup_dataset_from_file(file_path, n_validation_samples=0):
    """
    设置数据集，从文件加载表达式。
    :param file_path: 包含表达式的文件路径，每一行是一个表达式。
    :param n_validation_samples: 验证集样本数。
    :return: 如果没有验证集，返回 (training, embedding)，否则返回 (training, validation, embedding)。
    """
    import torch
    from torchvision.transforms import Compose

    # 从文件加载表达式
    with open(file_path, 'r') as file:
        expressions = [line.strip() for line in file if line.strip()]  # 去除空行

    random.shuffle(expressions)

    # 划分训练集和验证集
    if n_validation_samples > 0:
        training_expressions = expressions[:-n_validation_samples]
        validation_expressions = expressions[-n_validation_samples:]
    else:
        training_expressions = expressions
        validation_expressions = None

    # 创建临时 CFGEquationDataset 实例以获取 pcfg 和 max_grammar_productions
    temp_dataset = CFGEquationDataset(n_samples=1)  # 用于初始化 pcfg 和 max_grammar_productions
    #pcfg = temp_dataset.pcfg
    pcfg = CFGEquationDataset.get_pcfg(use_original_grammar=True)
    print("pcfg:", pcfg)
    max_grammar_productions = temp_dataset.max_grammar_productions
    print("max_grammar_productions:", max_grammar_productions)

    # 初始化 GrammarParseTreeEmbedding
    embedding = GrammarParseTreeEmbedding(pcfg, pad_to_length=max_grammar_productions)

    # 定义转换
    transform = Compose([
        embedding,
        ToTensor(dtype=torch.int64)
    ])

    # 包装数据集
    class DatasetWrapper:
        def __init__(self, expressions, transform=None, pcfg=None, max_grammar_productions=None):
            self.expressions = expressions
            self.transform = transform
            self.pcfg = pcfg  # 添加 pcfg 属性
            self.max_grammar_productions = max_grammar_productions  # 添加 max_grammar_productions 属性

        def __len__(self):
            return len(self.expressions)

        def __getitem__(self, idx):
            expression = self.expressions[idx]
            if self.transform:
                expression = self.transform(expression)
            return expression

    # 创建训练集和验证集
    training = DatasetWrapper(training_expressions, transform=transform, pcfg=pcfg, max_grammar_productions=max_grammar_productions)
    validation = DatasetWrapper(validation_expressions, transform=transform, pcfg=pcfg, max_grammar_productions=max_grammar_productions) if validation_expressions else None

    # 返回结果
    if not validation:
        return training, embedding
    else:
        return training, validation, embedding





'''def save_latent_space(model, data_loader, output_file):
    model.eval()
    all_latent_vectors = []

    with torch.no_grad():
        for X in data_loader:
            # Assuming model.encode(X) correctly extracts the latent vectors
            latent_mean, latent_log_std = model.encode(X)
            #reparameterization trick
            z = latent_mean + torch.randn_like(latent_log_std) * torch.exp(0.5 * latent_log_std)
            latent_vector = z
            # Concatenate mean and std along the last dimension
            #latent_vector = torch.stack((latent_mean, latent_log_std), dim=-1)

            all_latent_vectors.append(latent_vector.cpu().numpy())

    # Convert list to a numpy array and save
    if all_latent_vectors:
        all_latent_vectors = np.concatenate(all_latent_vectors, axis=0)
        np.savez(output_file, all_latent_vectors)
        logging.info(f"Latent space saved to {output_file}.npz")
    else:
        logging.info("No latent vectors extracted.")'''

import torch
import numpy as np
import logging

import torch
import numpy as np
import logging

def save_latent_space(model, data_loader, output_file, cfg_embedding):
    model.eval()
    all_latent_vectors = []
    all_labels = []  # 存储X作为标签

    with torch.no_grad():
        for X in data_loader:
            # 直接将X的值用作标签，这里我们假设X可以直接转换为numpy数组
            label = X.cpu().numpy()
            latent_mean, latent_log_var = model.encode(X)
            # 使用重参数化技巧
            z = latent_mean + torch.randn_like(latent_log_var) * torch.exp(0.5 * latent_log_var)
            latent_vector = z.cpu().numpy()

            #label = cfg_embedding.labels_to_expressions(X.cpu().numpy())
            for x in X.cpu().numpy():
                print("x:", x)
                expression = cfg_embedding.from_index_to_expressions(x)  # 逐个解码

                all_labels.append(str(expression))  # 添加到标签列表
                print("expression:", expression)
                print("all_labels:", all_labels)

            all_latent_vectors.append(latent_vector)
            #all_labels.append(label)  # 将X作为numpy数组存储
            print("Checking all_labels content:")
            for i, label in enumerate(all_labels):
                print(f"Label {i}: {label}, Type: {type(label)}")

            all_labels = [str(label) for label in all_labels]

    # 转换列表为 numpy 数组并保存
    if all_latent_vectors:
        all_latent_vectors = np.concatenate(all_latent_vectors, axis=0)
        #all_labels = np.array(all_labels)  # 确保labels是numpy数组
        #all_labels are strings, so we don't need to convert them to numpy array

        np.savez(output_file, latent_vectors=all_latent_vectors, labels=all_labels)
        logging.info(f"Latent space and labels saved to {output_file}.npz")
    else:
        logging.info("No latent vectors extracted.")



# 请确保您的 data_loader 能够返回正确的数据和标签格式



def main():
    ts = datetime.now().strftime("%Y-%b-%d-%H-%M-%S")

    #initialize wandb
    wandb.init(project="gvae_pretrained", name=f"gvae_pretrained_{ts}")
    wandb.config.update({vars(args)})

    # Hyperparameters
    #get number of epochs from args
    num_epochs = args.num_epochs
    early_stopping_patience = args.early_stopping_patience
    expression_with_parameters = True

    export_file = f"results/gvae_pretrained_{ts}.pth" if not expression_with_parameters else f"results/gvae_pretrained_parametric_{ts}.pth"
    #add a latent space file
    latent_space_file = f"results/latent_space_{ts}.npy" if not expression_with_parameters else f"results/latent_space_parametric_{ts}.npz"
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size

    # create the dataset
    #training, validation, embedding = setup_dataset(n_samples=10**4, n_validation_samples=10**3, expressions_with_parameters=expression_with_parameters)
    training, validation, embedding = setup_dataset_from_file(file_path='equations_5.txt', n_validation_samples=args.validation_samples)
    print("embedding:", embedding)
    index_to_rule_mapping = {i: rule for i, rule in enumerate(training.pcfg.productions())}

    # print the index to rule mapping
    print("Index to Rule Mapping:")
    for index, rule in index_to_rule_mapping.items():
        print(f"Index {index}: {rule}")

    training_loader = DataLoader(dataset=training,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 collate_fn=PadSequencesToSameLength())

    validation_loader = DataLoader(dataset=validation,
                                   batch_size=val_batch_size,
                                   shuffle=False,
                                   collate_fn=PadSequencesToSameLength())

    # build the model
    model = GrammarVariationalAutoencoder(
        len(training.pcfg.productions()) + 1,
        training.max_grammar_productions,
        #max_of_production_steps = 128,
        latent_dim=args.latent_dim,
        rule_embedding=embedding,
        expressions_with_parameters=expression_with_parameters
    )

    print(model)


    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters()) #parameters from the model
    learning_rate_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs//2, eta_min=1e-8)


    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Training Grammar Variational Autoencoder (GVAE) for { num_epochs } epochs.")


    no_improvement_for_epochs = 0
    best_validation_loss = torch.inf #infinitely large

    for epoch in range(num_epochs):

        # Start of a new epoch
        epoch_steps = tqdm.tqdm(training_loader, desc=f"Epoch {epoch + 1}: ", unit=" Batches")
        #print the data in the epoch
        print("epoch_steps:", epoch_steps)


        model.train()
        for X in epoch_steps:
            print("X:",X)
            # compute the loss
            loss = model.negative_elbo(X)

            # optimize
            loss.backward()
            optimizer.step()

            epoch_steps.set_postfix({   #Updates the display information of the progress bar
                'loss': loss.detach().item(),
                'lr': learning_rate_schedule.get_last_lr()[0]
            })

        # update the learning rate according to the scheduler
        learning_rate_schedule.step()

        # run the validation
        with torch.no_grad():
            model.eval()

            validation_loss = 0
            sample_count = 0
            for X in validation_loader:
                val_loss = model.negative_elbo(X)
                validation_loss += val_loss

                """if sample_count < 5:  # Print only the first 5 samples
                    logits = model(X)  # Assuming there's a method to generate outputs
                    probabilities = torch.softmax(logits, dim=1)
                    print("Logits:", logits)
                    predicted_indices = torch.argmax(probabilities, dim=1)

                    # Decode both inputs and outputs
                    # record the input index and the output index
                    input_indices = [x for x in X]
                    output_indices = [idx for idx in predicted_indices]
                    decoded_input_expressions = [embedding.from_index_to_expressions(idx) for idx in X]
                    decoded_output_expressions = [
                        embedding.from_index_to_expressions(idx) for idx in predicted_indices
                    ]

                    print("Sample Inputs and Outputs:")
                    for input_indices, output_indices, inp_expr, out_expr in zip(input_indices, output_indices, decoded_input_expressions, decoded_output_expressions):
                        print(f"Input Index: {input_indices}")
                        print(f"Output Index: {output_indices}")
                        print(f"Input Expression: {inp_expr}")
                        print(f"Output Expression: {out_expr}")

                    sample_count += 1"""

            validation_loss = validation_loss / len(validation_loader)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                no_improvement_for_epochs = 0

                logging.info(f"Validation {epoch + 1}: loss={validation_loss}")
            else:
                no_improvement_for_epochs += 1
                logging.info(f"Validation {epoch + 1}: loss={validation_loss} (no improvement for { no_improvement_for_epochs } epochs)")
            #print the input equation and the output equation in validation set, not the index but the euqation
            #record the train loss and validation loss in wandb
            wandb.log({"train_loss": loss.item(), "validation_loss": validation_loss
                          }, step=epoch)




        # early stopping
        if no_improvement_for_epochs > early_stopping_patience:
            return

        # save the weights to file
        torch.save(model, export_file)

        #save the latent space for validation set
        cfg_embedding = GrammarParseTreeEmbedding(training.pcfg, pad_to_length=training.max_grammar_productions)
        save_latent_space(model,validation_loader, latent_space_file, cfg_embedding=cfg_embedding)
        print("Latent space saved")

        #save the model to wandb
        wandb.save(export_file)
        wandb.save(latent_space_file)

        #finish the training
        wandb.finish()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #set the number of epochs
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of training epochs")
    #set the early stopping patience
    parser.add_argument('--early_stopping_patience', type=int, default=100, help="Number of epochs to wait for improvement before stopping")
    #set batch size and validation batch size
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--val_batch_size', type=int, default=128, help="Batch size for validation")
    #set validation samples
    parser.add_argument('--validation_samples', type=int, default=1000, help="Number of validation")
    #set the latent dimension
    parser.add_argument('--latent_dim', type=int, default=2, help="Latent dimension of the model")

    args = parser.parse_args()

    #use wandb to record all args



    main()
