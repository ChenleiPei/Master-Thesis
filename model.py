import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var

import torch
import torch.nn as nn


class SentenceVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                 sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):
        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        # set the parameters
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        # set the hyper-parameters
        self.latent_size = latent_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # choose rnn cell type
        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        else:
            raise ValueError("Unsupported RNN type selected!")

        # encoder and decoder rnn cells
        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional,
                               batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional,
                               batch_first=True)

        # to infer the latent variable z from the hidden state
        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)

        # inference to hidden state
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)

        # project the hidden state to the vocabulary space
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, input_sequence, length):
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        input_embedding = self.embedding(input_sequence)

        if self.word_dropout_rate > 0:
            prob = torch.rand(input_sequence.size(), device=input_sequence.device)
            prob[(input_sequence == self.sos_idx) | (input_sequence == self.pad_idx)] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)

        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)
        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        hidden = self.latent2hidden(z)
        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        outputs, _ = self.decoder_rnn(packed_input, hidden)
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]

        b, s, _ = padded_outputs.size()
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        print('shape of logp:', logp.shape)
        print("Original padded_outputs shape:", padded_outputs.shape)
        logp = logp.view(b, s, self.vocab_size)

        return logp, mean, logv, z

    def inference(self, n=4, z=None):
        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()
        print("running_seqs1:", running_seqs)

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t = 0
        while t < self.max_sequence_length and len(running_seqs) > 0:
            print("max_sequence_length:", self.max_sequence_length)
            if t == 0:
                input_sequence = to_var(
                    torch.Tensor(batch_size, 1).fill_(self.sos_idx).long())  # 初始化时确保 input_sequence 是 [batch_size, 1]

            input_embedding = self.embedding(input_sequence)  # [batch_size, 1, embedding_dim]
            print("input_embedding shape:", input_embedding.shape)

            output, hidden = self.decoder_rnn(input_embedding, hidden)
            logits = self.outputs2vocab(output)
            input_sequence = self._sample(logits)

            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).squeeze(-1)
            sequence_running = sequence_idx.masked_select(sequence_mask)

            running_mask = (input_sequence != self.eos_idx).squeeze(-1).data
            running_seqs = running_seqs.masked_select(running_mask)
            print("running_seqs:", running_seqs)

            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]  #choose the running sequences
                hidden = hidden[:, running_seqs]
                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, logits, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(logits, 1, dim=-1)  # max value's index
        sample = sample.squeeze(-1)
        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        running_latest = save_to[running_seqs]  # get the running sequences
        sample = sample.squeeze(-1)  # make sure sample is [batch_size]
        running_latest[:, t] = sample  # update the running sequences
        save_to[running_seqs] = running_latest  # save the running sequences
        return save_to


