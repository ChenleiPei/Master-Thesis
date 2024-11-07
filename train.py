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
from readeq import process_equations
from model import SentenceVAE
from utils import to_var, idx2word, expierment_name


def main(args):
    ts = time.strftime('%Y-%b-%d-%H-%M-%S', time.gmtime())

    # read equations from file
    vocab_size, input_sequences, target, lengths, vocab = process_equations(args.data_file)
    max_index = max(max(seq) for seq in input_sequences if seq)
    #vocab_size = max_index + 1  #make sure vocab size is larger than max index
    vocab_size = len(vocab)
    print("input sequences size:", len(input_sequences))
    print("vocab:", vocab)
    print("vocab_size:", vocab_size)
    sos_idx = vocab['<start>']
    eos_idx = vocab['<end>']
    pad_idx = vocab['<pad>']
    unk_idx = vocab.get('<unk>', vocab['<pad>'])  # Use pad index if unk is not specified

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

    if args.tensorboard_logging:
        logdir_path = os.path.join(args.logdir, expierment_name(args, ts))
        print("Log directory path:", logdir_path)
        writer = SummaryWriter(logdir_path)
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

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

    for epoch in range(args.epochs):
        model.train()
        for iteration, (batch_input, batch_target, batch_lengths) in enumerate(data_loader):
            batch_input = to_var(batch_input)
            batch_lengths = to_var(batch_lengths)
            batch_target = to_var(batch_target)
            logp, mean, logv, z = model(batch_input, batch_lengths)
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch_target, batch_lengths, mean, logv,
                                                   args.anneal_function, step, args.k, args.x0)
            loss = (NLL_loss + KL_weight * KL_loss) / args.batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if iteration % args.print_every == 0:
                print(f"Epoch {epoch}, Iteration {iteration}, Loss: {loss.item():.4f}")

        model_save_path = os.path.join(save_model_path, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"Model weights saved to {model_save_path}")

    final_model_path = os.path.join(save_model_path, 'model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model weights saved to {final_model_path}")

    if args.tensorboard_logging:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='equations.txt',
                        help='File containing mathematical expressions')
    parser.add_argument('--max_sequence_length', type=int, default=60)

    parser.add_argument('--embedding_size', type=int, default=3)
    parser.add_argument('--rnn_type', type=str, default='gru')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--word_dropout', type=float, default=0)
    parser.add_argument('--embedding_dropout', type=float, default=0.5)
    parser.add_argument('--latent_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--bidirectional', action='store_true')

    parser.add_argument('--epochs', type=int, default=10)
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
