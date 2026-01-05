import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class arDCA(nn.Module):
    def __init__(self, seq_len: int, l_alphabet: int):
        """
        seq_len = L
        l_alphabet = q
        """
        super().__init__()
        self.seq_len = seq_len
        self.l_alphabet = l_alphabet
        dim = self.seq_len * self.l_alphabet

        self.DCA_bias = nn.Parameter(torch.zeros(dim))

        # Couplings W_{i,a; j,b}
        # Initialized small so training is stable
        self.DCA_weights = nn.Parameter(0.01 * torch.randn(dim, dim))

        # -----------------------------
        # Autoregressive mask (non-trainable)
        # mask[i*q:(i+1)*q, j*q:(j+1)*q] = 1 if j < i else 0
        # -----------------------------
        mask = torch.zeros(dim, dim)
        for i in range(self.l_alphabet):
            for j in range(i):
                mask[i*self.l_alphabet:(i+1)*self.l_alphabet, j*self.l_alphabet:(j+1)*self.l_alphabet] = 1.0

        self.register_buffer("mask", mask)  # not trainable

    def forward(self, x):
        """
        x: (B, L) integer-encoded sequence
        Returns: (B, L, q) logits
        """
        B = x.shape[0]
        x_flat = x.reshape(B, self.seq_len * self.l_alphabet)

        # Masked couplings
        W_masked = self.DCA_weights * self.mask  # (L*q, L*q)

        logits_flat = self.DCA_bias + x_flat @ W_masked  # (B, L*q)
        logits = logits_flat.reshape(B, self.seq_len, self.l_alphabet)

        return logits

    def probs(self, x):
        return F.softmax(self.forward(x), dim=-1)



class arDCA_slow(nn.Module):
    def __init__(self, seq_len: int, l_alphabet: int):
        super().__init__()
        self.seq_len = seq_len
        self.l_alphabet = l_alphabet

        # One linear layer per position i
        # Each takes the flattened prefix (positions < i) and outputs logits for q states
        self.nn_bias0 = nn.Linear(1, self.q, bias=True)
        self.layers = nn.ModuleList([
            nn.Linear(i * self.q, self.q)
            for i in range(1, seq_len)
        ])

    def forward(self, x):
        B = x.size(0)

        outs = []
        for i in range(self.L):
            if i == 0:
                # First position: no prefix, just bias
                out = self.nn_bias0(torch.ones(B, 1).to(x.device))
            else:
                prefix = x[:, :i, :].reshape(B, -1)  # flatten prefix
                out = self.layers[i-1](prefix)
            outs.append(out)
        
        logits = torch.stack(outs, dim=1)
        return logits # (B, L, q)


    def probs(self, x):
        """Return probabilities for each position/state."""
        return F.softmax(self.forward(x), dim=-1)



class FFNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, n_hidden_nodes):
        super().__init__()
        layers = []

        # First layer: input → hidden
        layers.append(nn.Linear(input_dim, n_hidden_nodes))
        layers.append(nn.LeakyReLU(negative_slope=0.1))

        # Middle hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden_nodes, n_hidden_nodes))
            layers.append(nn.LeakyReLU(negative_slope=0.1)) # leaky ReLU for better convergence

        # Final layer: hidden → output
        layers.append(nn.Linear(n_hidden_nodes, output_dim))

        # Wrap in Sequential
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class arNN(nn.Module):
    def __init__(self, seq_len: int, l_alphabet: int, n_layers: int, h_dim: int):
        """
        L = sequence length
        q = alphabet size (e.g. 20 amino acids)
        """
        super().__init__()
        self.L = seq_len
        self.q = l_alphabet

        # One linear layer per position i
        # Each takes the flattened prefix (positions < i) and outputs logits for q states
        self.nn_bias0 = nn.Linear(1, self.q, bias=True)
        self.FFNets = nn.ModuleList([
            FFNet(i * self.q, self.q, n_layers, h_dim)
            for i in range(1, seq_len)
        ])

    def forward(self, x):
        B = x.size(0)

        outs = []
        for i in range(self.L):
            if i == 0:
                # First position: no prefix, just bias
                out = self.nn_bias0(torch.ones(B, 1).to(x.device))
            else:
                prefix = x[:, :i, :].reshape(B, -1)  # flatten prefix
                out = self.FFNets[i-1](prefix)
            outs.append(out)

        logits = torch.stack(outs, dim=1)
        return logits # (B, L, q)

    def probs(self, x):
        """Return probabilities for each position/state."""
        return F.softmax(self.forward(x), dim=-1)



class arHOCA(nn.Module):
    def __init__(self, seq_len: int, l_alphabet: int, n_layers: int, h_dim: int):
        super().__init__()
        self.arDCA = arDCA(seq_len, l_alphabet)
        self.arNN = arNN(seq_len, l_alphabet, n_layers, h_dim)

    def forward(self, x):        
        logits_DCA = self.arDCA(x)
        logits_NN = self.arNN(x)

        return logits_DCA + logits_NN # (B, L, q)

    def probs(self, x, ardca_only: bool = False):
        if ardca_only:
            return F.softmax(self.arDCA.forward(x), dim=-1)

        """Return probabilities for each position/state."""
        return F.softmax(self.forward(x), dim=-1)

