# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Sequence

import torch
from mmengine.model import BaseModel
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizerFast, CLIPTextConfig  # NEW: Fast tokenizer

from mmdet.registry import MODELS


def generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_tokens_list):
    """Generate attention mask between each pair of special tokens."""
    input_ids = tokenized['input_ids']
    bs, num_token = input_ids.shape
    special_tokens_mask = torch.zeros((bs, num_token),
                                      device=input_ids.device).bool()

    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    idxs = torch.nonzero(special_tokens_mask)
    attention_mask = (
        torch.eye(num_token,
                  device=input_ids.device).bool().unsqueeze(0).repeat(
                      bs, 1, 1))
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1:col + 1,
                           previous_col + 1:col + 1] = True
            position_ids[row, previous_col + 1:col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device)
        previous_col = col

    return attention_mask, position_ids.to(torch.long)


@MODELS.register_module()
class CLIPModel(BaseModel):
    """CLIP text encoder with enhanced features."""

    def __init__(self,
                 name: str = 'openai/clip-vit-base-patch32',
                 max_tokens: int = 77,
                 pad_to_max: bool = True,
                 use_sub_sentence_represent: bool = False,
                 special_tokens_list: list = None,
                 num_layers_of_embedded: int = 1,
                 use_checkpoint: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max

        # NEW: Use faster tokenizer
        self.tokenizer = CLIPTokenizerFast.from_pretrained(name)
        self.language_backbone = nn.Sequential(
            OrderedDict([('body',
                          CLIPTextEncoder(
                              name,
                              num_layers_of_embedded=num_layers_of_embedded,
                              use_checkpoint=use_checkpoint))]))

        self.use_sub_sentence_represent = use_sub_sentence_represent
        if self.use_sub_sentence_represent:
            assert special_tokens_list is not None, \
                'special_tokens required for sub-sentence representation'
            self.special_tokens = self.tokenizer.convert_tokens_to_ids(
                special_tokens_list)

    def forward(self, captions: Sequence[str], **kwargs) -> dict:
        device = next(self.language_backbone.parameters()).device
        
        # NEW: Add offset mapping
        tokenized = self.tokenizer(
            captions,
            max_length=self.max_tokens,
            padding='max_length' if self.pad_to_max else 'longest',
            return_tensors='pt',
            truncation=True,
            return_offsets_mapping=True  # NEW: Get offset mappings
        ).to(device)

        # NEW: Extract and remove offset mapping from model inputs
        offset_mapping = tokenized.pop('offset_mapping')

        if self.use_sub_sentence_represent:
            attention_mask, position_ids = \
                generate_masks_with_special_tokens_and_transfer_map(
                    tokenized, self.special_tokens)
        else:
            attention_mask = tokenized.attention_mask
            position_ids = None

        tokenizer_input = {
            'input_ids': tokenized.input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }
        
        language_features = self.language_backbone(tokenizer_input)
        output = {
            'embedded': language_features['embedded'],
            'masks': language_features['masks'],
            'offset_mapping': offset_mapping,  # NEW: Add offset mapping
            'input_ids': tokenized.input_ids,  # NEW: Optional but useful
        }
        if self.use_sub_sentence_represent:
            output['position_ids'] = position_ids
            output['text_token_mask'] = tokenized.attention_mask.bool()
        return output


class CLIPTextEncoder(nn.Module):
    """Enhanced CLIP encoder with layer aggregation."""

    def __init__(self,
                 name: str,
                 num_layers_of_embedded: int = 1,
                 use_checkpoint: bool = False):
        super().__init__()
        config = CLIPTextConfig.from_pretrained(name)
        config.output_hidden_states = True
        self.model = CLIPTextModel.from_pretrained(name, config=config)
        self.model.gradient_checkpointing = use_checkpoint
        
        self.language_dim = config.hidden_size
        self.num_layers_of_embedded = num_layers_of_embedded

    def forward(self, x) -> dict:
        if x['attention_mask'].dim() == 3:
            # attention mask is 3D
            outputs = self.model(
                input_ids=x['input_ids'],
                attention_mask=x['attention_mask'][:,0,:],
                position_ids=x.get('position_ids', None),
                output_hidden_states=True,
            )
        else:
            outputs = self.model(
                input_ids=x['input_ids'],
                attention_mask=x['attention_mask'],
                position_ids=x.get('position_ids', None),
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states
        encoded_layers = hidden_states[-self.num_layers_of_embedded:]
        
        features = torch.stack(encoded_layers, dim=1).mean(dim=1)
        features = features / self.num_layers_of_embedded
        
        mask = x['attention_mask']
        if mask.dim() == 2:
            embedded = features * mask.unsqueeze(-1).float()
        else:
            embedded = features

        return {
            'embedded': embedded,
            'masks': mask,
            'hidden': encoded_layers[-1]
        }

