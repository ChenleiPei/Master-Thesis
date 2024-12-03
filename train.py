import os
import json
import time
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from collections import OrderedDict, defaultdict
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
import numpy as np
import wandb
from readeq import process_equations
from model import SentenceVAE
from utils import to_var, idx2word, expierment_name


def main(args):
    ts = time.strftime('%Y-%b-%d-%H-%M-%S', time.gmtime())

    wandb.init(project="Bayesian_reasoning_in_latent_space", name="Training_with_eq10000_input")
    wandb.config.update(vars(args))

    # read equations from file
    vocab_size, input_sequences, target, lengths, vocab = process_equations(args.data_file)
    max_index = max(max(seq) for seq in input_sequences if seq)
    #vocab_size = max_index + 1  #make sure vocab size is larger than max index
    vocab_size = len(vocab)
    print("input sequences size:", len(input_sequences))
    print("vocab_size:", vocab_size)
    sos_idx = vocab['<start>']
    eos_idx = vocab['<end>']
    pad_idx = vocab['<pad>']
    #add unk_idx as the last index of vocab
    if '<unk>' not in vocab:
        vocab['<unk>'] = len(vocab)
    print("vocab:", vocab)
    unk_idx = vocab['<unk>']
    #unk_idx = vocab.get('<unk>', vocab['<pad>'])  # Use pad index if unk is not specified

    params = {
        'vocab_size': vocab_size,
        'sos_idx': sos_idx,
        'eos_idx': eos_idx,
        'pad_idx': pad_idx,
        'unk_idx': unk_idx,
        'max_sequence_length': args.max_sequence_length,
        'embedding_size': args.embedding_size,
        'rnn_type': args.rnn_type,
        'hidden_size': args.hidden_size,
        'word_dropout': args.word_dropout,
        'embedding_dropout': args.embedding_dropout,
        'latent_size': args.latent_size,
        'num_layers': args.num_layers,
        'bidirectional': args.bidirectional
    }
    model = SentenceVAE(**params)

    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU.")

    print(model)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path, exist_ok=True)

    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-k * (step - x0))))
        elif anneal_function == 'linear':
            return min(1, step / x0)

    NLLLoss = torch.nn.NLLLoss(ignore_index=pad_idx, reduction='sum')

    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        print("target_size:", target.size())
        logp = logp.view(-1, logp.size(2))
        print("logp_size:", logp.size())
        NLL_loss = NLLLoss(logp, target)
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)
        return NLL_loss, KL_loss, KL_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    input_tensor = torch.tensor(input_sequences, dtype=torch.long)
    target_tensor = torch.tensor(target, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    dataset = TensorDataset(input_tensor, target_tensor, lengths_tensor)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    step = 0
    latent_vectors = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for iteration, (batch_input, batch_target, batch_lengths) in enumerate(data_loader):
            batch_input = to_var(batch_input)
            batch_lengths = to_var(batch_lengths)
            batch_target = to_var(batch_target)
            logp, mean, logv, z = model(batch_input, batch_lengths)
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch_input, batch_lengths, mean, logv,
                                                   args.anneal_function, step, args.k, args.x0)
            loss = (NLL_loss + KL_weight * KL_loss) / args.batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            epoch_loss += loss.item()

            # save the latent vectors
            latent_vectors.append(z.detach().cpu().numpy())

            if iteration % args.print_every == 0:
                print(f"Epoch {epoch}, Iteration {iteration}, Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}")

        wandb.log({"epoch": epoch, "avg_loss": avg_epoch_loss, "NLL_loss": NLL_loss.item(), "KL_loss": KL_loss.item(),
                   "KL_weight": KL_weight})

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
    torch.save(model.decoder_rnn.state_dict(), decoder_save_path)
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='equations_10000.txt',
                        help='File containing mathematical expressions')
    parser.add_argument('--max_sequence_length', type=int, default=60)

    parser.add_argument('--embedding_size', type=int, default=3)
    parser.add_argument('--rnn_type', type=str, default='gru')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--word_dropout', type=float, default=0)
    parser.add_argument('--embedding_dropout', type=float, default=0.5)
    parser.add_argument('--latent_size', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--bidirectional', action='store_true')

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--tensorboard_logging', action='store_true')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--save_model_path', type=str, default='bin')

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)
    args = parser.parse_args()

    main(args)
