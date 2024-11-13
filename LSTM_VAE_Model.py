import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size):
        super(LSTMEncoder, self).__init__()
        # embedding to convert input index to embedding vector
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # LSTM layer
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        # Linear layers to map hidden state to mean and log variance of latent space
        self.hidden_to_mu = nn.Linear(hidden_size, latent_size)
        self.hidden_to_log_var = nn.Linear(hidden_size, latent_size)

    def forward(self, x, sequence_lengths):
        # embedding to convert input index to embedding vector
        embedded = self.embedding(x)
        # pack the sequence
        packed_input = rnn_utils.pack_padded_sequence(embedded, sequence_lengths, batch_first=True,
                                                      enforce_sorted=False)
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)
        # Use the last hidden state to generate mean and log variance
        hidden = hidden.squeeze(0)
        mu = self.hidden_to_mu(hidden)
        log_var = self.hidden_to_log_var(hidden)
        return mu, log_var


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size):
        super(LSTMDecoder, self).__init__()
        # embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # Linear layer to map latent vector to hidden state
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        # LSTM
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        # linear layer to map output to vocab size
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, z, sequence_lengths):
        # Map latent vector to hidden state
        hidden = self.latent_to_hidden(z).unsqueeze(0)
        cell = torch.zeros_like(hidden)  # Initialize cell state with zeros
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded, (hidden, cell))
        outputs = self.fc(outputs)
        return outputs


class LSTMVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size, vocab):
        super(LSTMVAE, self).__init__()

        self.encoder = LSTMEncoder(vocab_size, embedding_size, hidden_size, latent_size)
        self.decoder = LSTMDecoder(vocab_size, embedding_size, hidden_size, latent_size)
        self.vocab = vocab
        self.index_to_token = {index: token for token, index in vocab.items()}  # 索引到符号的映射

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1)
        """
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)  # Random normal tensor
        return mu + eps * std

    def forward(self, x, sequence_lengths):
        # Encode input to get mu and log_var
        mu, log_var = self.encoder(x, sequence_lengths)
        # Reparameterize to get latent vector
        z = self.reparameterize(mu, log_var)
        # Decode using the latent vector
        decoded = self.decoder(x, z, sequence_lengths)
        return decoded, mu, log_var,z

    def reconstruct_expression(self, sequence, from_indices=True):
        """

        Args:
            sequence (torch.Tensor): input sequence
            from_indices (bool): if True, the input sequence is a sequence of indices, otherwise it is a sequence of

        Returns:
            list: expression list
        """
        if not from_indices:
            # we need to convert the input sequence to indices
            _, predicted_indices = torch.max(sequence, dim=-1)  # [batch_size, seq_length]
        else:
            predicted_indices = sequence  # [batch_size, seq_length]

        # to store the expressions
        expressions = []
        for sequence in predicted_indices:
            print("sequence:",sequence)
            expression = [self.index_to_token[idx.item()] for idx in sequence]
            expression = [token for token in expression if token not in ['<pad>', '<end>']]
            expressions.append(" ".join(expression))

        return expressions
