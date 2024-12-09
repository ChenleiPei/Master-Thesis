import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import argparse



class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size, num_layers):
        super(LSTMEncoder, self).__init__()
        # embedding to convert input index to embedding vector
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # LSTM layer
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, num_layers=num_layers)
        # Linear layers to map hidden state to mean and log variance of latent space
        self.hidden_to_mu = nn.Linear(hidden_size, latent_size)
        self.hidden_to_log_var = nn.Linear(hidden_size, latent_size)
        self.num_layers = num_layers

    def forward(self, x, sequence_lengths):
        # embedding to convert input index to embedding vector
        #print("x:",x.shape)
        embedded = self.embedding(x)
        #print("embedded:",embedded.shape)
        # pack the sequence
        packed_input = rnn_utils.pack_padded_sequence(embedded, sequence_lengths, batch_first=True,
                                                      enforce_sorted=False)
        #print("packed_input:",packed_input.data.shape)
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)
        #print("packed_output:",packed_output.data.shape)

        # Use the last hidden state to generate mean and log variance
        # hidden = hidden.squeeze

        hidden = hidden[-1]  # Use the last layer's hidden state

        #add activate function, weak relu
        #mu = self.hidden_to_mu(hidden)
        mu = F.leaky_relu(self.hidden_to_mu(hidden))
        log_var = F.leaky_relu(self.hidden_to_log_var(hidden))

        return mu, log_var


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size, num_layers):
        super(LSTMDecoder, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # Linear layer to map latent vector to hidden state
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        # LSTM layer
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
        # Linear layer to map hidden state to vocab size
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers

    '''def forward(self, z, max_length, start_token_idx):
        # Map latent vector to hidden state
        hidden = F.leaky_relu(self.latent_to_hidden(z)).unsqueeze(0)
        hidden = hidden.repeat(self.num_layers, 1, 1)  # Repeat for number of layers
        cell = torch.zeros_like(hidden)  # Initialize cell state with zeros

        # Start with the <start> token
        batch_size = z.size(0)
        input_token = torch.full((batch_size, 1), start_token_idx, dtype=torch.long)  # [batch_size, 1]

        # Decode step by step for the target length
        outputs = []
        for _ in range(max_length):
            embedded = self.embedding(input_token)  # [batch_size, 1, embedding_size]
            lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # Remove num_layers argument
            logits = self.fc(lstm_out)  # [batch_size, 1, vocab_size]
            outputs.append(logits)

            # Get the token with the highest probability for the next input
            _, next_token = torch.max(logits, dim=-1)
            input_token = next_token  # Use the current output as the next input

        outputs = torch.cat(outputs, dim=1)  # Concatenate all outputs to get [batch_size, max_length, vocab_size]
        return outputs'''

    def forward(self, z, max_length, start_token_idx):
        # Map latent vector to hidden state
        #print("z:",z.shape)
        hidden = F.leaky_relu(self.latent_to_hidden(z)).unsqueeze(0)  # [1, batch_size, hidden_size]
        print("hidden1:", hidden.shape)
        print("number of layers", self.num_layers)
        hidden = hidden.repeat(self.num_layers, 1, 1)  # Repeat for number of layers
        print("hidden2:",hidden.shape)
        cell = torch.zeros_like(hidden)  # Initialize cell state with zeros

        # Initialize input sequence with <start> token
        batch_size = z.shape[0]
        print("batch_size:",batch_size)
        #batch_size = 1
        input_tokens = torch.full((batch_size, max_length), start_token_idx,
                                  dtype=torch.long)  # [batch_size, max_length]

        # Embed the entire input sequence
        embedded = self.embedding(input_tokens)  # [batch_size, max_length, embedding_size]

        # Pass the entire embedded sequence to LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # [batch_size, max_length, hidden_size]

        # Compute logits for all time steps
        logits = self.fc(lstm_out)  # [batch_size, max_length, vocab_size]

        # Get predicted tokens (optional: during inference)
        _, predicted_tokens = torch.max(logits, dim=-1)  # [batch_size, max_length]

        return logits, predicted_tokens
        #return logits  #While training


class LSTMVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size, vocab, max_length, num_layers):
        super(LSTMVAE, self).__init__()

        self.encoder = LSTMEncoder(vocab_size, embedding_size, hidden_size, latent_size, num_layers)
        self.decoder = LSTMDecoder(vocab_size, embedding_size, hidden_size, latent_size, num_layers)
        self.vocab = vocab
        self.index_to_token = {index: token for token, index in vocab.items()}  # 索引到符号的映射
        self.token_to_index = vocab  # 符号到索引的映射
        self.max_length = max_length

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
        start_token_idx = self.token_to_index['<start>']
        #max_length = sequence_lengths.max().item()  # Use the maximum length of the target sequences
        max_length = self.max_length
        logits, predicted_tokens = self.decoder(z, max_length, start_token_idx)
        return logits, predicted_tokens, mu, log_var, z


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
