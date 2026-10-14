from dataclasses import dataclass, fields
from typing import Tuple, List
import torch
import torch.nn as nn
import copy

from utils import l1_loss
from vision_transformer import PatchEmbedding, PositionalEncoding2D, AttentionHead, MultipleHeadAttention, TransformerEncoder
from models import ContextEncoder, Predictor, Mask
from import_config import ModelConfig, GlobalConfig, EncoderConfig, MaskingConfig, TrainingConfig, PredictorConfig


class IJEPA():
    def __init__(
            self,
            device,
            config: ModelConfig
    ):
        super().__init__()
        self.device = device
        self.config = config

        self.image_shape = config.image_shape
        self.patch_shape = config.patch_shape
        self.grid_shape  = (self.image_shape[1] // self.patch_shape[0], self.image_shape[2] // self.patch_shape[1])
        self.num_patches = config.num_patches
        self.momentum    = config.momentum

        # ----- NN creation -----
        self.context_encoder = ContextEncoder(image_shape=self.image_shape, patch_shape=self.patch_shape, config=config.encoder).to(self.device)
        self.target_encoder  = copy.deepcopy(self.context_encoder).to(self.device)
        self.predictor       = Predictor(encoder_embedding_size=config.encoder.embedding_size, num_patches=self.num_patches, image_shape=self.image_shape, patch_shape=self.patch_shape, config=config.predictor).to(self.device)
        self.masking         = Mask(config=config.masking)

        # ----- No gradient update for target encoder -----
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # ----- Optimizer creation -----

        self._context_params = list(self.context_encoder.parameters())
        self._target_params  = list(self.target_encoder.parameters())

        self.context_encoder_optimizer = torch.optim.AdamW(self.context_encoder.parameters(), lr=self.config.context_learning_rate)
        self.predictor_optimizer       = torch.optim.AdamW(self.predictor.parameters()      , lr=self.config.predictor_learning_rate)

    @torch.no_grad()
    def _EMA_update(self):
        weight = 1.0 - self.momentum
        torch._foreach_lerp_(self._target_params, self._context_params, weight)


    def train(self, x):
        # x -> Batch of images : (batch_size, C, H, W)

        # x = self.patch_embedding(x)
        # # x -> Batch of embedded patches : (batch_size, num_patches, embedding_size)

        # x = self.positional_encoding(x)
        # # x -> Batch of embedded patches : (batch_size, num_patches, embedding_size) we just add the positional encoding for the corresponding patches

        context_indices, targets_indices = self.masking._get_blocks_indices(grid_shape=self.grid_shape)
        #context_block = x[:, context_indices, :]    # same context block for all images in the batch, TRY WITH DIFFERENT FOR ALL IMAGES LIKE ORIGINAL PAPER
        # context_block -> Batch of context block embedded patches : (batch_size, num_patches_context, embedding_size)
        with torch.no_grad():
            full_target_block = self.target_encoder(x)

        context_block = self.context_encoder(x, context_indices)  # same context block for all images in the batch, TRY WITH DIFFERENT FOR ALL IMAGES LIKE ORIGINAL PAPER
        # context_block -> Batch of context block embedded patches : (batch_size, num_patches_context, embedding_size)

        tot_loss = 0.0
        for target_index in targets_indices:
            prediction = self.predictor(context_block, len(target_index), context_indices, target_index)
            target_block  = full_target_block[:, target_index, :]
            loss = l1_loss(prediction=prediction, target=target_block)
            tot_loss = tot_loss + loss

        tot_loss = tot_loss / len(targets_indices)

        self.context_encoder_optimizer.zero_grad()
        self.predictor_optimizer.zero_grad()
        tot_loss.backward()
        self.context_encoder_optimizer.step()
        self.predictor_optimizer.step()

        # EMA update
        self._EMA_update()

        return {"loss": tot_loss.detach(),
                "target_std": full_target_block.std(dim=0).mean(),}





        
        