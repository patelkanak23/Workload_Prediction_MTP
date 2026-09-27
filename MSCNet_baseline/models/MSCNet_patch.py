import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer, Attention_Block
from layers.Embed import PatchEmbedding
from layers.Autoformer_EncDec import series_decomp
from layers.MSGBlock import GraphBlock, simpleVIT, Attention_Block, Predict
class MSPBlock(nn.Module):
    def __init__(self, configs, patch_sizes=[16,24,32,48,96], kernel_sizes=[6,12,24]):
        # 6,12,24   19,24,32,
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = 1
        self.patch_sizes = patch_sizes

        self.norm = torch.nn.LayerNorm(self.k)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

        self.Linear_Scales = nn.ModuleList()
        self.flatten = nn.Flatten(start_dim=-2)
        self.codeEncoders = nn.ModuleList()
        self.codeEncoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=configs.output_attention), self.seq_len, configs.n_heads),
                        self.seq_len,
                        self.seq_len,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(1)
                ],
                norm_layer=torch.nn.LayerNorm(self.seq_len)
            )
        for patch_size in self.patch_sizes:
            linear_layer = nn.Linear(patch_size, patch_size)
            self.Linear_Scales.append(linear_layer)

        #粗粒度卷积
        self.conv_layers1 = nn.ModuleList([
            nn.Conv1d(in_channels=self.k, out_channels=self.k, kernel_size=kernel_size, padding=kernel_size // 2, stride=kernel_size // 2)
            for kernel_size in kernel_sizes
        ])
        #等距卷积
        self.conv_layers2 = nn.ModuleList([
            nn.Conv1d(in_channels=self.k, out_channels=self.k, kernel_size=kernel_size, padding=0, stride=1)
            for kernel_size in kernel_sizes
        ])
        #细粒度卷积
        self.conv_layers3 = nn.ModuleList([
            nn.Conv1d(in_channels=self.k, out_channels=self.k, kernel_size=kernel_size // 2, padding=0, stride=kernel_size // 2)
            for kernel_size in kernel_sizes
        ])

        total_out_length = 0
        for kernel_size in kernel_sizes:
            length1 = (configs.seq_len + 2 * 0 - kernel_size) // 1 + 1
            length2 = (length1 + 2 * (kernel_size // 2) - kernel_size) // (kernel_size // 2) + 1
            length3 = (length1 + length2 + 2 * 0 - kernel_size // 2) // (kernel_size // 2) + 1
            total_out_length += length1 + length2 + length3

        self.predict = nn.Linear(total_out_length * self.k, self.seq_len)
        self.projection = nn.Linear(self.k * self.seq_len, self.seq_len)

    def forward(self, x):
        dec_out, attn = self.codeEncoder(x.reshape(x.shape[0] * x.shape[1], 1, -1))

        # 对每个 kernel_size 使用不同的卷积层
        conv_outs = []
        for conv_layer1, conv_layer2, conv_layer3 in zip (self.conv_layers1, self.conv_layers2, self.conv_layers3):
            x1 = conv_layer2(dec_out)
            x2 = conv_layer1(x1)
            x3 = torch.cat((x1, x2), dim=-1)
            x4 = conv_layer3(x3)
            x5 = torch.cat((x3, x4), dim=-1)
            conv_outs.append(x5)

        conv_out = torch.cat(conv_outs, dim=-1)
        conv_out = conv_out.reshape(x.shape[0], x.shape[1], -1)
        dec_out = self.predict(conv_out)

        res = x + dec_out
        return res


class Model(nn.Module):

    def __init__(self, configs, individual=False, patch_sizes=[16,24,32,48,96], strides=[12,16,24]):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.seq_len)
        self.channels = configs.enc_in
        self.model = nn.ModuleList([MSPBlock(configs) for _ in range(configs.e_layers)])

        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def encoder(self, x):
        means = x.mean(1, keepdim=True).detach()
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        seasonal_init, trend_init = self.decompsition(x_enc)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)

        seasonal_init = x_enc.permute(0, 2, 1)
        for i in range(self.layer):
            seasonal_init = self.layer_norm(self.model[i](seasonal_init))

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output

        dec_out = x.permute(0, 2, 1) * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None