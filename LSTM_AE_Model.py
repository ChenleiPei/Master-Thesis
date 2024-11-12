import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(LSTMEncoder, self).__init__()
        # embedding to convert input index to embedding vector
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # LSTM 层，接收嵌入向量作为输入
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
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(LSTMDecoder, self).__init__()
        # embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # LSTM
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        # linear layer to map output to vocab size
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell, sequence_lengths):
        print(x.shape)
        embedded = self.embedding(x)
        print(embedded.shape)
        outputs, _ = self.lstm(embedded, (hidden, cell))
        print("OUTPUTS1", outputs.shape)
        outputs = self.fc(outputs)

        return outputs



class LSTMAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, vocab):
        super(LSTMAutoencoder, self).__init__()
        # 初始化编码器和解码器
        self.encoder = LSTMEncoder(vocab_size, embedding_size, hidden_size)
        self.vocab = vocab
        self.decoder = LSTMDecoder(vocab_size, embedding_size, hidden_size)
        self.embedding = nn.Embedding(vocab_size, embedding_size)  # 用于解码时映射回词汇索引
        self.vocab = vocab
        self.index_to_token = {index: token for token, index in vocab.items()}  # 索引到符号的映射

    def forward(self, x, sequence_lengths):
        # 编码阶段
        hidden, cell = self.encoder(x, sequence_lengths)
        # 解码阶段，使用编码阶段的隐藏状态和单元状态
        #print state of x
        print("X", x.shape)
        decoded = self.decoder(x, hidden, cell, sequence_lengths)
        return decoded

    def reconstruct_expression(self, sequence, from_indices=True):
        """
        从解码器生成的嵌入向量序列中重建出原始的数学表达式，
        或者从词汇表索引序列重建原始输入表达式。

        Args:
            sequence (torch.Tensor): 输入序列或解码器的输出序列，可以是嵌入向量或者索引。
            from_indices (bool): 如果为 True，直接将输入视为词汇表索引，否则将输入视为嵌入序列。

        Returns:
            list: 重建后的表达式字符串列表。
        """
        if not from_indices:
            # 如果输入不是索引序列，将嵌入向量序列转换为索引
            _, predicted_indices = torch.max(sequence, dim=-1)  # [batch_size, seq_length]
        else:
            predicted_indices = sequence  # [batch_size, seq_length]

        # 将索引转换为词汇表中的符号
        expressions = []
        for sequence in predicted_indices:
            expression = [self.index_to_token[idx.item()] for idx in sequence]
            # 去掉 <pad> 和 <end> 标记，重建原始表达式
            expression = [token for token in expression if token not in ['<pad>', '<end>']]
            expressions.append(" ".join(expression))
        return expressions
