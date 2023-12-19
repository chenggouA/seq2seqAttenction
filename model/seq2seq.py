import random

from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch_size, seq_len]
        embedded = self.dropout(self.embedding(src))

        # embedded = [batch_size, seq_len, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs [batch_size, seq_len, hid_dim]
        # hidden = [batch_size, n_layers, hid_dim]
        # cell = [batch_size, n_layers, hid_dim]

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)  # 添加一维
        # x: [batch_size, 1] 

        embedded = self.dropout(self.embedding(x))
        # embedded: [batch_size, 1, emb_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        prediction = self.fc_out(output.squeeze(1))

        # prediction = [batch_size, output]

        return prediction, hidden, cell


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'

        assert encoder.hid_dim == decoder.hid_dim, \
            "解码器隐藏层维度必须和编码器隐藏层维度相等"
        assert encoder.n_layers == decoder.n_layers, \
            "编码器和解码器必须有相同的层数!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # teacher_forcing 教师强制

        # src = [batch_size, src_len]
        # trg = [batch_size, trg_len]
        # teacher_forcing_ratio是使用教师强制的概率
        # 例如，如果teacher_forcing_ratio为0.75，我们将在75%的时间内使用真值输入
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # 用于存储解码器输出的tensor
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # 将编码器的最后隐藏状态用作解码器的初始隐藏状态
        hidden, cell = self.encoder(src)

        # 解码器的第一个输出是 <BOS> tokens

        input = trg[:, 0]

        for t in range(1, trg_len):
            # 插入输入标记的词嵌入嵌入向量，前一个隐藏和前一个单元格状态
            # 接收输出张量(预测)和新的 hidden 和 cell 状态
            output, hidden, cell = self.decoder(input, hidden, cell)

            # 将预测放置在一个张量中， 其中包含每个标记的预测
            outputs[t] = output

            # 决定是否要使用教师强迫
            teacher_force = random.random() < teacher_forcing_ratio
            # teacher_force = False

            # 从我们的预测中获得最高的预测 tokens
            top1 = output.argmax(1)

            # 如果教师强制，使用实际的下一个标记作为下一个输入
            # 如果不是，使用预测的标记

            input = trg[:, t] if teacher_force else top1

        return outputs


if __name__ == "__main__":
    vocab_len = 10000
    batch_size = 64
    emb_dim = 300
    hidden = 64
    n_layers = 3

    input_len = 140
    output_len = 252

    encoder = Encoder(vocab_len, emb_dim, hidden, n_layers, 0.5)
    decoder = Decoder(vocab_len, emb_dim, hidden, n_layers, dropout=0.5)

    s2s = seq2seq(encoder, decoder)

    src = torch.randint(0, vocab_len, (batch_size, input_len))
    trg = torch.randint(0, vocab_len, (batch_size, output_len))

    print(s2s(src, trg).shape)