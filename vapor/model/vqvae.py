import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # print (encoding_indices.shape, quantized.shape, encodings.shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


## Credit: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/srgan/models.py


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


# class GeneratorResNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
#         super(GeneratorResNet, self).__init__()

#         # First layer
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
#             nn.PReLU(),
#         )

#         # Residual blocks
#         res_blocks = []
#         for _ in range(n_residual_blocks):
#             res_blocks.append(ResidualBlock(64))
#         self.res_blocks = nn.Sequential(*res_blocks)

#         # Second conv layer post residual blocks
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=9, stride=1, padding=4),
#             nn.BatchNorm2d(64, 0.8),
#             nn.PReLU(),
#         )

#         # Upsampling layers
#         upsampling = []
#         for out_features in range(2):
#             upsampling += [
#                 # nn.Upsample(scale_factor=2),
#                 nn.Conv2d(64, 256, 3, 1, 1),
#                 nn.BatchNorm2d(256),
#                 nn.PixelShuffle(upscale_factor=2),
#                 nn.PReLU(),
#             ]
#         self.upsampling = nn.Sequential(*upsampling)

#         # Final output layer
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4),
#             # nn.Tanh(),
#             nn.PReLU(),
#         )

#     def forward(self, x):
#         out1 = self.conv1(x)
#         out = self.res_blocks(out1)
#         out2 = self.conv2(out)
#         out = torch.add(out1, out2)
#         # print (x, out1, out2, out)
#         out = self.upsampling(out)
#         out = self.conv3(out)
#         return out


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self._conv_3 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # self._residual_stack = ResidualStack(in_channels=num_hiddens,
        #                                      num_hiddens=num_hiddens,
        #                                      num_residual_layers=num_residual_layers,
        #                                      num_residual_hiddens=num_residual_hiddens)

        # Residual blocks
        res_blocks = []
        for _ in range(num_residual_layers):
            res_blocks.append(ResidualBlock(num_hiddens))
        self._residual_stack = nn.Sequential(*res_blocks)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        x = F.relu(x)

        x = self._residual_stack(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        out_channels,
    ):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # self._residual_stack_x = ResidualStack(
        #     in_channels=num_hiddens,
        #     num_hiddens=num_hiddens,
        #     num_residual_layers=num_residual_layers,
        #     num_residual_hiddens=num_residual_hiddens,
        # )

        # Residual blocks
        res_blocks = []
        for _ in range(num_residual_layers):
            res_blocks.append(ResidualBlock(num_hiddens))
        self._residual_stack = nn.Sequential(*res_blocks)

        # self._upsampling_x = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         in_channels=num_hiddens,
        #         out_channels=num_hiddens // 2,
        #         kernel_size=4,
        #         stride=2,
        #         padding=1,
        #         output_padding=0,
        #         dilation=1,
        #     ),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(
        #         in_channels=num_hiddens // 2,
        #         out_channels=out_channels,
        #         kernel_size=4,
        #         stride=2,
        #         padding=1,
        #         output_padding=0,
        #         dilation=1,
        #     ),
        #     nn.ReLU(),
        # )

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(num_hiddens, num_hiddens * 4, 3, 1, 1),
                nn.BatchNorm2d(num_hiddens * 4),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]

        upsampling += [
            nn.Conv2d(
                num_hiddens, num_hiddens // 2, kernel_size=3, stride=1, padding=1
            ),
            nn.PReLU(),
            nn.Conv2d(
                num_hiddens // 2, out_channels, kernel_size=3, stride=1, padding=1
            ),
            # nn.Tanh(),
        ]
        self._upsampling = nn.Sequential(*upsampling)

    def forward(self, inputs):
        x1 = self._conv_1(inputs)

        x2 = self._residual_stack(x1)

        #         x = self._conv_trans_1(x)
        #         x = F.relu(x)
        #         x = self._conv_trans_2(x)
        #         x = F.relu(x)

        x = torch.add(x1, x2)
        x = self._upsampling(x)

        return x


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay=0,
    ):
        super(VQVAE, self).__init__()

        self._encoder = Encoder(
            in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
        )
        self._pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1
        )
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, decay
            )
        else:
            self._vq_vae = VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )
        self._decoder = Decoder(
            embedding_dim,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            out_channels,
        )
        # self._decoder_x = GeneratorResNet(embedding_dim)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity
