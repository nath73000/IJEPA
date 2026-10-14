import torch
import torch.nn as nn 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import math
import matplotlib.pyplot as plt
from typing import Tuple



class PatchEmbedding(nn.Module):
    def __init__(self, patch_shape, embed_dim):
        super().__init__()
        
        self.patch_shape        = patch_shape
        self.patch_size         = self.patch_shape[0]
        self.num_pixels_patches = 3 * (self.patch_size**2)
        self.embed_dim          = embed_dim
        
        self.linear = nn.Linear(self.num_pixels_patches, embed_dim)

    def forward(self, x):

        B, C, H, W = x.shape

        nb_raw_division    = W // self.patch_size
        nb_column_division = H // self.patch_size

        x = x.reshape(
            B, 
            C, 
            nb_raw_division, 
            self.patch_size, 
            nb_column_division, 
            self.patch_size
        )

        x = x.permute(0, 2, 4, 1, 3, 5)

        x = x.reshape(
            B,
            nb_raw_division * nb_column_division,
            C * self.patch_size ** 2
        )

        return self.linear(x)



def get_2d_sincos_pos_embed(embed_dim: int, grid_shape: tuple[int, int]) -> torch.Tensor:
    """
    embed_dim : dimension totale D (ex: 768)
    grid_shape : tuple (H, W) du nombre de patchs (ex: (14, 14))
    """
    assert embed_dim % 2 == 0, "embed_dim doit être divisible par 2"
    h, w = grid_shape

    # 1. Grille de coordonnées (H, W)
    # y va de 0 à h-1, x va de 0 à w-1
    grid_y = torch.arange(h, dtype=torch.float32)
    grid_x = torch.arange(w, dtype=torch.float32)
    
    # meshgrid génère deux matrices (H, W) de coordonnées
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')

    # Aplatir en vecteurs 1D de longueur H * W
    grid_y = grid_y.flatten()  # Forme: (H*W,)
    grid_x = grid_x.flatten()  # Forme: (H*W,)

    # 2. Calcul des encodages 1D pour chaque axe avec embed_dim // 2
    dim_per_axis = embed_dim // 2
    emb_y = get_1d_sincos_pos_embed(dim_per_axis, grid_y)  # Forme: (H*W, D/2)
    emb_x = get_1d_sincos_pos_embed(dim_per_axis, grid_x)  # Forme: (H*W, D/2)

    # 3. Concaténation le long de la dernière dimension
    pos_embed = torch.cat([emb_y, emb_x], dim=1)  # Forme: (H*W, D)

    return pos_embed



def get_1d_sincos_pos_embed(dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    dim : dimension de l'encodage pour cet axe (doit être paire)
    pos : tenseur 1D de coordonnées de taille (N,)
    """
    assert dim % 2 == 0
    # Calcul des fréquences : omega = 1 / (10000 ** (2i / dim))
    omega = torch.arange(dim // 2, dtype=torch.float32)
    omega /= (dim / 2)
    omega = 1.0 / (10000 ** omega)  # Forme: (dim / 2,)

    # Multiplication matricielle externe : (N, 1) * (1, dim/2) -> (N, dim/2)
    out = torch.einsum('m,d->md', pos, omega)
    
    # Sinus et Cosinus entrelacés ou concaténés
    emb_sin = torch.sin(out)  # (N, dim/2)
    emb_cos = torch.cos(out)  # (N, dim/2)

    return torch.cat([emb_sin, emb_cos], dim=1)  # (N, dim)



class PositionalEncoding2D(torch.nn.Module):
    def __init__(
            self,
            image_shape: Tuple[int, int, int],
            patch_shape: Tuple[int, int],
            embedding_size: int
    ):
        super().__init__()

        _, image_h, image_w = image_shape[0], image_shape[1], image_shape[2]
        patch_h, patch_w    = patch_shape[0], patch_shape[1]
        grid_h = image_h // patch_h
        grid_w = image_w // patch_w
        self.num_patches = grid_h * grid_w

        # Générer l'encodage fixe avec token CLS
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=embedding_size, 
            grid_shape=(grid_h, grid_w)
        )
        
        # Enregistrer en buffer figé (non entraînable)
        # Forme finale : (1, 1 + num_patches, embed_dim)
        self.register_buffer("pos_encoding", pos_embed.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, idx=None) -> torch.Tensor:
        pos = self.pos_encoding.expand(x.shape[0], -1, -1)
        if idx is not None:
            pos = pos[:, idx, :]
        return x + pos



class AttentionHead(nn.Module):
    def __init__(self, embedding_size, head_size):
        super().__init__()

        self.head_size = head_size

        self.quary = nn.Linear(embedding_size, head_size)
        self.key   = nn.Linear(embedding_size, head_size)
        self.value = nn.Linear(embedding_size, head_size)


    def forward(self, x):
        quary_values = self.quary(x)
        key_values   = self.key(x)
        value_values = self.value(x)

        attention = quary_values @ key_values.transpose(-2, -1)      # We tranpose only the last two dimensions of the matrice to not touch the batch dim
        attention = attention / (self.head_size ** 0.5)
        attention = torch.softmax(attention, dim=-1)                 # We apply the softmax on the last dimension e.i. the column
        attention = attention @ value_values

        return attention



class MultipleHeadAttention(nn.Module):
    def __init__(self, num_heads, embedding_size):
        super().__init__()

        self.head_size = embedding_size // num_heads

        self.out_linear = nn.Linear(embedding_size, embedding_size)

        self.heads = nn.ModuleList([AttentionHead(embedding_size, self.head_size) for _ in range(num_heads)])


    def forward(self, x):
        out = [head(x) for head in self.heads]
        out = torch.cat(out, dim=-1)             # We add the colums together
        out = self.out_linear(out)

        return out



class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size):
        super().__init__()
        self.num_heads     = num_heads
        self.embedding_size = embedding_size

        self.layer_norm_1 = nn.LayerNorm(embedding_size)
        self.layer_norm_2 = nn.LayerNorm(embedding_size)

        self.multi_head_attention = MultipleHeadAttention(num_heads, embedding_size)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_size), 
            nn.GELU(),
            nn.Linear(hidden_size, embedding_size),
        )


    def forward(self, x):
        out1 = x + self.multi_head_attention(self.layer_norm_1(x))
        out2 = out1 + self.mlp(self.layer_norm_2(out1))
        return out2