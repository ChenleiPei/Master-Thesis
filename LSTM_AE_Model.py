import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(LSTMEncoder, self).__init__()
        # embedding to convert input index to embedding vector
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # LSTM layer
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)

    def forward(self, x, sequence_lengths):
        # embedding to convert input index to embedding vector
        print("x", x.shape)
        embedded = self.embedding(x)
        print("embedded", embedded.shape)
        # pack the sequence
        packed_input = rnn_utils.pack_padded_sequence(embedded, sequence_lengths, batch_first=True,
                                                      enforce_sorted=False)
        # LSTM
        #initialize hidden and cell states
        packed_output, (hidden, cell) = self.lstm(packed_input)
        # return hidden and cell states
        print("SHAPE", hidden.shape, cell.shape)
        return hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, max_length):
        super(LSTMDecoder, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # LSTM layer
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        # Linear layer to map LSTM output to vocabulary size
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.max_length = max_length

    def forward(self, hidden, cell, max_length, start_token_idx):
        batch_size = hidden.size(1)  # Get batch size from hidden state

        # Start with the <start> token for each sequence in the batch
        input_token = torch.full((batch_size, 1), start_token_idx, dtype=torch.long)  # [batch_size, 1]

        outputs = []

        for _ in range(max_length):
            embedded = self.embedding(input_token)  # [batch_size, 1, embedding_size]
            lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # [batch_size, 1, hidden_size]
            logits = self.fc(lstm_out)  # [batch_size, 1, vocab_size]
            outputs.append(logits)

            # Choose the token with the highest probability
            _, next_token = torch.max(logits, dim=-1)
            input_token = next_token  # Use the current output as the next input

        # Concatenate all outputs to get the final sequence: [batch_size, max_length, vocab_size]
        outputs = torch.cat(outputs, dim=1)
        return outputs


class LSTMAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, vocab, max_length):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = LSTMEncoder(vocab_size, embedding_size, hidden_size)
        self.decoder = LSTMDecoder(vocab_size, embedding_size, hidden_size, max_length)
        self.vocab = vocab
        self.index_to_token = {index: token for token, index in vocab.items()}  # Index-to-token mapping
        self.token_to_index = vocab  # Token-to-index mapping
        self.max_length = max_length

    def forward(self, x, sequence_lengths):
        # Encode the input sequence
        hidden, cell = self.encoder(x, sequence_lengths)

        # Use <start> token to initialize the decoding process
        start_token_idx = self.token_to_index['<start>']

        # Decode using the hidden state and cell state from the encoder
        decoded = self.decoder(hidden, cell, max_length=self.max_length, start_token_idx=start_token_idx)

        return decoded


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
            expression = [self.index_to_token[idx.item()] for idx in sequence]

            expression = [token for token in expression if token not in ['<pad>', '<end>']]
            expressions.append(" ".join(expression))
        return expressions
