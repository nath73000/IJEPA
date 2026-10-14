from dataclasses import dataclass, fields
import yaml
import os
from typing import Tuple, List
import random
import torch
import torch.nn as nn

from vision_transformer import PatchEmbedding, PositionalEncoding2D, AttentionHead, MultipleHeadAttention, TransformerEncoder



def get_mask_indices(
    image_shape: Tuple[int, int, int], 
    context_scale_range: Tuple[float, float], 
    target_scale_range: Tuple[float, float], 
    target_aspect_ratio_range: Tuple[float, float], 
    num_target_blocks: int
):
    # ----- image shape -----
    grid_h, grid_w = image_shape[1], image_shape[2]
    total_num_patches = grid_h * grid_w

    # ----- Context block -----
    context_scale, context_aspect_ratio = random.uniform(*context_scale_range), 1
    num_patches_context_block = round(total_num_patches * context_scale)
    height_context_block      = round((num_patches_context_block / context_aspect_ratio) ** (1/2))
    width_context_block       = round((num_patches_context_block * context_aspect_ratio) ** (1/2))

    topleft_patch_context = (random.randint(0, grid_h - height_context_block), 
                             random.randint(0, grid_w  - width_context_block)
                             )

    context_patch_indices_2d = [(r, c) for r in range(topleft_patch_context[0], topleft_patch_context[0] + height_context_block) 
                                       for c in range(topleft_patch_context[1], topleft_patch_context[1] + width_context_block)
                                ]
    context_patches_indices = [r * grid_w + c for r,c in context_patch_indices_2d]

    # ----- Target blocks -----
    target_patches_indices = []
    all_target_patches     = set()
    for _ in range(num_target_blocks):
        target_scale, target_aspect_ratio = random.uniform(*target_scale_range), random.uniform(*target_aspect_ratio_range)
        num_patches_block_target = round(total_num_patches * target_scale)
        height_target_block      = round((num_patches_block_target / target_aspect_ratio) ** (1/2))
        width_target_block       = round((num_patches_block_target * target_aspect_ratio) ** (1/2))

        topleft_patch_target = (random.randint(0, grid_h - height_target_block), 
                                random.randint(0, grid_w  - width_target_block)
                                )

        target_patch_indices_2d = [(r, c) for r in range(topleft_patch_target[0], topleft_patch_target[0] + height_target_block) 
                                          for c in range(topleft_patch_target[1], topleft_patch_target[1] + width_target_block)
                                    ]
        target_patch_indices_1d = [r * grid_w + c for r,c in target_patch_indices_2d]

        target_patches_indices.append(target_patch_indices_1d)
        all_target_patches.update(target_patch_indices_1d)

    # ----- Filter the conext block -----
    filtered_context = [p for p in context_patches_indices if p not in all_target_patches]

    return filtered_context, target_patches_indices


def l1_loss(prediction, target):
    return abs(target - prediction).sum() / prediction.numel()


def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)