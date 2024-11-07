import os
import json
import torch
import argparse

from model import SentenceVAE
from utils import to_var, idx2word, interpolate
from readeq import process_equations

def main(args):
    # read equations from file
    alphabet_size, padded_inputs, padded_targets, sequence_lengths, vocab = process_equations(args.data_dir + '/equations.txt')
    w2i, i2w = vocab, {idx: token for token, idx in vocab.items()}
    #print the vocab and w2i, i2w
    print("w2i:", w2i)
    print("i2w:", i2w)
    print("vocab:", vocab)


    # create model
    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<start>'],
        eos_idx=w2i['<end>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i.get('<unk>', 0),  # Use pad index if unk is not specified
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
    )

    # load model
    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(f"Checkpoint file {args.load_checkpoint} not found.")
    model.load_state_dict(torch.load(args.load_checkpoint, map_location='cpu'))
    print(f"Model loaded from {args.load_checkpoint}")

    # transfer model to GPU
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # process inference output
    def process_inference_output(output):
        samples, z = output
        samples = samples.squeeze(-1)  # 调整 samples 维度
        return samples, z

    # generate samples
    raw_output = model.inference(n=args.num_samples)
    samples, z = process_inference_output(raw_output)
    print('----------SAMPLES----------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    # interpolate between two random samples
    z1 = torch.randn([args.latent_size]).numpy()
    z2 = torch.randn([args.latent_size]).numpy()
    z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    raw_output = model.inference(z=z)
    samples, _ = process_inference_output(raw_output)
    print('-------INTERPOLATION-------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('-c', '--load_checkpoint', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('-n', '--num_samples', type=int, default=10, help="Number of samples to generate")


    parser.add_argument('-dd', '--data_dir', type=str, default='data', help="Directory containing dataset")
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50, help="Maximum sequence length")
    parser.add_argument('-eb', '--embedding_size', type=int, default=3, help="Size of embeddings")
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru', help="Type of RNN: rnn, lstm, or gru")
    parser.add_argument('-hs', '--hidden_size', type=int, default=256, help="Size of hidden layers")
    parser.add_argument('-wd', '--word_dropout', type=float, default=0, help="Word dropout probability")
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5, help="Embedding dropout probability")
    parser.add_argument('-ls', '--latent_size', type=int, default=16, help="Dimensionality of latent space")
    parser.add_argument('-nl', '--num_layers', type=int, default=1, help="Number of RNN layers")
    parser.add_argument('-bi', '--bidirectional', action='store_true', help="Use bidirectional RNN")

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    assert args.rnn_type in ['rnn', 'lstm', 'gru'], "Invalid RNN type"
    assert 0 <= args.word_dropout <= 1, "Word dropout must be between 0 and 1"

    main(args)
