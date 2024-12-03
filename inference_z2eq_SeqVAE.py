import torch
import torch.nn as nn
from model import SentenceVAE
from utils import idx2word
from readeq import process_equations
import torch.nn.functional as F
import os
import argparse


def main(args):
    # read equations from file
    _, _, _, _, vocab = process_equations(args.data_dir + '/equations_100.txt')
    w2i, i2w = vocab, {idx: token for token, idx in vocab.items()}

    if '<unk>' not in w2i:
        unk_index = len(w2i)  # add <unk> token to vocab
        w2i['<unk>'] = unk_index
        i2w[unk_index] = '<unk>'


    print("Vocabulary loaded.")
    print("w2i:", w2i)
    print("i2w:", i2w)

    # 初始化解码器模型
    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<start>'],
        eos_idx=w2i['<end>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
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

    # load decoder checkpoint
    if not os.path.exists(args.load_decoder_checkpoint):
        raise FileNotFoundError(f"Decoder checkpoint file {args.load_decoder_checkpoint} not found.")
    model.decoder_rnn.load_state_dict(torch.load(args.load_decoder_checkpoint, map_location='cpu'))
    print(f"Decoder loaded from {args.load_decoder_checkpoint}")

    # transfer model to GPU
    if torch.cuda.is_available():
        model = model.cuda()

    # create latent to hidden linear layer
    latent_to_hidden = nn.Linear(args.latent_size, args.hidden_size)

    # put latent_to_hidden on GPU if available
    if torch.cuda.is_available():
        latent_to_hidden = latent_to_hidden.cuda()

    # generate a random latent vector
    #z = torch.rand(1, args.latent_size)
    z = torch.tensor([[0.1, 0.1]])
    if torch.cuda.is_available():
        z = z.cuda()
    print("Generated latent vector:", z)

    # use latent_to_hidden to get hidden state
    hidden = latent_to_hidden(z)
    print("shape of hidden:", hidden.shape)
    hidden = hidden.unsqueeze(0)  # 调整维度以适应 RNN 的输入 (num_layers, batch_size, hidden_size)
    print("shape of hidden after unsqueeze:", hidden.shape)
    # 在推理时，由于我们是单个样本，因此需要将 hidden 调整为 2D 张量 (num_layers, hidden_size)
    #hidden = hidden.squeeze(0)

    # inference
    model.decoder_rnn.eval()
    with torch.no_grad():
        # initialize input token
        start_token = torch.tensor([w2i['<start>']]).unsqueeze(0).to(z.device)  # (1, 1)
        inputs = start_token

        # decode one token at a time
        outputs = []
        max_sequence_length = args.max_sequence_length

        for _ in range(max_sequence_length):
            # use the decoder to get the next token
            print("shape of inputs:", inputs.shape)
            embedded_input = model.embedding(inputs)  # shape: (1, 1, embedding_size)
            # reduce 1 dimension

            # forward pass
            print("shape of embedded_input:", embedded_input.shape)
            print("shape of hidden:", hidden.shape)
            output, hidden = model.decoder_rnn(embedded_input, hidden)  # 输出形状：(batch_size, seq_len, vocab_size)
            print("shape of output:", output.shape)
            token_logits = output.squeeze(1)  # 去掉 seq_len 维度，形状变为：(batch_size, vocab_size)
            #print the size of token_logits
            print("token_logits:", token_logits.size())

            # 通过 softmax 得到概率分布
            token_probs = F.softmax(token_logits, dim=-1)
            token_probs[:, unk_index] = 0
            #print the probability of each token
            print("token_probs:", token_probs)

            # 从概率分布中采样一个 token
            #token_id = torch.multinomial(token_probs, num_samples=1)  # 形状：(batch_size, 1)

            token_id = torch.argmax(token_probs, dim=-1).unsqueeze(0)

            # 如果采样到的 token 超出了词汇表范围，则重新采样（确保索引合法）
            if token_id.item() >= len(w2i):
                token_id = torch.tensor([w2i['<unk>']]).to(z.device)

            outputs.append(token_id.item())

            # 如果遇到 <end> token，结束生成
            if token_id.item() == w2i['<end>']:
                break

            # 否则将当前 token 作为下一次解码的输入
            # inputs = token_id.unsqueeze(0)
            # pay attention to the shape of inputs
            inputs = token_id.view(1, 1)



    # 包装 outputs 使其符合 idx2word 的输入格式
    equation = idx2word([outputs], i2w=i2w, pad_idx=w2i['<pad>'])

    print("Generated Equation:", ' '.join(equation))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-dc', '--load_decoder_checkpoint', type=str, required=True,
                        help="Path to the decoder checkpoint")

    parser.add_argument('-dd', '--data_dir', type=str, default='data', help="Directory containing dataset")
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50, help="Maximum sequence length")
    parser.add_argument('-eb', '--embedding_size', type=int, default=3, help="Size of embeddings")
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru', help="Type of RNN: rnn, lstm, or gru")
    parser.add_argument('-hs', '--hidden_size', type=int, default=256, help="Size of hidden layers")
    parser.add_argument('-wd', '--word_dropout', type=float, default=0, help="Word dropout probability")
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5, help="Embedding dropout probability")
    parser.add_argument('-ls', '--latent_size', type=int, default=2, help="Dimensionality of latent space")
    parser.add_argument('-nl', '--num_layers', type=int, default=1, help="Number of RNN layers")
    parser.add_argument('-bi', '--bidirectional', action='store_true', help="Use bidirectional RNN")

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    assert args.rnn_type in ['rnn', 'lstm', 'gru'], "Invalid RNN type"
    assert 0 <= args.word_dropout <= 1, "Word dropout must be between 0 and 1"

    main(args)
