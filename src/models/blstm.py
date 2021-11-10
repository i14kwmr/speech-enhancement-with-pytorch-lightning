import torch
import torch.nn as nn


class BiLSTM2SPK(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=300, num_layers=4, dropout=0.3
    ):
        # input_dim : 入力の最終次元，今回はn_bin
        # output_dim: 出力の次元，今回はn_binにしておく．

        super(BiLSTM2SPK, self).__init__()
        self.output_dim = output_dim
        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim * 2)
        self.init_rnn(self.rnn)

    def init_rnn(self, m):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                for ih in param.chunk(3, 0):

                    torch.nn.init.xavier_uniform_(ih)

            elif "weight_hh" in name:
                for hh in param.chunk(3, 0):
                    torch.nn.init.orthogonal_(hh)

            elif "bias" in name:
                torch.nn.init.zeros_(param)

    def forward(self, x):
        # (n_batch, n_channel, n_freq, n_frame)

        # (n_batch, n_channel, n_frame, n_freq)
        x = x.permute(0, 1, 3, 2)  # masksと対応取った
        n_batch, _, n_frame, n_freq = x.size()

        rnn_output, _ = self.rnn(x[:, 0, :, :])

        masks = self.fc(rnn_output)
        masks = torch.sigmoid(masks)
        masks = masks.reshape(n_batch, n_frame, self.output_dim, 2)

        # (n_batch, n_channel, n_freq, n_frame)
        masks = masks.permute(0, 2, 1, 3)

        return masks[..., 0], masks[..., 1]
