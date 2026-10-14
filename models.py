from dataclasses import dataclass, fields
import yaml
import os
from typing import Tuple, List
import random
import torch
import torch.nn as nn

from vision_transformer import PositionalEncoding2D, TransformerEncoder, PatchEmbedding
from import_config import ModelConfig, GlobalConfig, MaskingConfig, EncoderConfig, TrainingConfig, PredictorConfig



class Mask():
    def __init__(
        self,
        config,
    ):
        self.context_scale       = config.context_scale
        self.target_scale        = config.target_scale
        self.target_aspect_ratio = config.target_aspect_ratio
        self.num_target_blocks   = config.num_target_blocks


    def _get_blocks_indices(self, grid_shape: Tuple[int, int]):
        # ----- image shape -----
        grid_h, grid_w    = grid_shape[0], grid_shape[1]
        total_num_patches = grid_h * grid_w

        # ----- Context block -----
        context_scale        = random.uniform(*self.context_scale)
        context_aspect_ratio = 1
        num_patches_context_block = round(total_num_patches * context_scale)
        height_context_block      = round((num_patches_context_block / context_aspect_ratio) ** (1/2))
        width_context_block       = round((num_patches_context_block * context_aspect_ratio) ** (1/2))

        topleft_patch_context = (
            random.randint(0, grid_h - height_context_block), 
            random.randint(0, grid_w  - width_context_block)
        )

        context_patch_indices_2d = [
            (r, c) 
            for r in range(topleft_patch_context[0], topleft_patch_context[0] + height_context_block) 
            for c in range(topleft_patch_context[1], topleft_patch_context[1] + width_context_block)
        ]
        context_patches_indices = [r * grid_w + c for r,c in context_patch_indices_2d]

        # ----- Target blocks -----
        target_patches_indices = []
        all_target_patches     = set()
        for _ in range(self.num_target_blocks):
            target_scale, target_aspect_ratio = random.uniform(*self.target_scale), random.uniform(*self.target_aspect_ratio)
            num_patches_block_target = round(total_num_patches * target_scale)
            height_target_block      = round((num_patches_block_target / target_aspect_ratio) ** (1/2))
            width_target_block       = round((num_patches_block_target * target_aspect_ratio) ** (1/2))

            topleft_patch_target = (
                random.randint(0, grid_h - height_target_block), 
                random.randint(0, grid_w  - width_target_block)
            )

            target_patch_indices_2d = [
                (r, c) 
                for r in range(topleft_patch_target[0], topleft_patch_target[0] + height_target_block) 
                for c in range(topleft_patch_target[1], topleft_patch_target[1] + width_target_block)
            ]
            target_patch_indices_1d = [r * grid_w + c for r,c in target_patch_indices_2d]

            target_patches_indices.append(target_patch_indices_1d)
            all_target_patches.update(target_patch_indices_1d)

        # ----- Filter the conext block -----
        filtered_context = [p for p in context_patches_indices if p not in all_target_patches]

        return filtered_context, target_patches_indices


class ContextEncoder(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        patch_shape: Tuple[int, int],
        config: EncoderConfig
    ):
        super().__init__()
        self.embedding_size = config.embedding_size
        self.num_heads      = config.num_heads
        self.num_layers     = config.num_layers
        self.hidden_size    = config.hidden_size

        self.patch_embedding     = PatchEmbedding(patch_shape, self.embedding_size)
        self.positional_encoding = PositionalEncoding2D(image_shape, patch_shape, self.embedding_size)

        self.tranformer_encoder = nn.Sequential(*[TransformerEncoder(self.embedding_size, self.num_heads, self.hidden_size) for _ in range(self.num_layers)])
        self.norm               = nn.LayerNorm(self.embedding_size)

    def forward(self, x, indices = None):
        # x -> Batch of context blocks patches (batch_size, num_patches_context)
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)

        if indices is not None:
            x = self.tranformer_encoder(x[:, indices, :])
        x = self.tranformer_encoder(x)      # For the target encoder, in case we donc have specific indices and we just want to put the full image
        
        x = self.norm(x)
        # x -> Batch of context blocks patches (batch_size, num_patches_context)
        return x


class Predictor(nn.Module):
    def __init__(
        self,
        encoder_embedding_size: int,
        num_patches: int,
        image_shape: Tuple[int, int, int],
        patch_shape: Tuple[int, int],
        config: PredictorConfig,
    ): 
        super().__init__()
        self.embedding_size = config.embedding_size
        self.num_heads      = config.num_heads
        self.num_layers     = config.num_layers
        self.hidden_size    = config.hidden_size

        self.input_projection    = nn.Linear(encoder_embedding_size, self.embedding_size)
        self.outpout_projection  = nn.Linear(self.embedding_size, encoder_embedding_size)
        self.positional_encoding = PositionalEncoding2D(image_shape, patch_shape, self.embedding_size)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(self.embedding_size, self.num_heads, self.hidden_size) for _ in range(self.num_layers)])
        self.norm                = nn.LayerNorm(self.embedding_size)
        self.mask_token          = nn.Parameter(torch.rand(1, 1, self.embedding_size))

    def forward(self, x, num_target_patches, context_idices, target_indices):

        x = self.input_projection(x)
        x = self.positional_encoding(x, context_idices)
        masks_token = self.mask_token.expand(x.shape[0], num_target_patches, self.embedding_size)
        masks_token = self.positional_encoding(masks_token, target_indices)

        x = torch.cat([x, masks_token], 1)
        x = self.transformer_encoder(x)
        x = self.norm(x)

        x = x[:, -num_target_patches:, :]    # we cut off the context block

        return self.outpout_projection(x)
        

