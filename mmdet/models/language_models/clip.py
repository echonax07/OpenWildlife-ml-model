from typing import Sequence
import torch
from mmengine.model import BaseModel
from torch import nn
from mmdet.registry import MODELS
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTokenizerFast

@MODELS.register_module()
class ClipTextEncoder(BaseModel):
    def __init__(self,
                 name: str = 'openai/clip-vit-base-patch32',
                 max_tokens: int = 77,  # CLIP's fixed sequence length
                 pad_to_max: bool = True,
                 use_sub_sentence_represent = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max
        self.use_sub_sentence_represent = True
        
        # CLIP components
        self.tokenizer = CLIPTokenizerFast.from_pretrained(name)
        self.model = CLIPTextModel.from_pretrained(name)
        self.text_feature_dim = self.model.config.hidden_size

    def forward(self, captions: Sequence[str]) -> dict:
        device = next(self.model.parameters()).device
        
        # CLIP's tokenization (automatically handles special tokens)
        tokenized = self.tokenizer(
            captions,
            max_length=self.max_tokens,
            padding='max_length' if self.pad_to_max else 'longest',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        ).to(device)
        # Extract offset mapping and remove from model inputs
        offset_mapping = tokenized.pop('offset_mapping')
        attention_mask_2d = tokenized['attention_mask'].bool()
    
        # Convert to 3D mask [bs, seq_len, seq_len]
        seq_len = attention_mask_2d.shape[1]
        attention_mask_3d = attention_mask_2d.unsqueeze(2) & attention_mask_2d.unsqueeze(1)
    
        # Forward through CLIP
        outputs = self.model(**tokenized)
        
        # Build output dict to match BERT's structure
        language_dict_features = {
            'embedded': outputs.last_hidden_state,
            'masks': attention_mask_3d,  # 3D mask for transformer
            'hidden': outputs.last_hidden_state,  # Same as embedded for CLIP
            'input_ids': tokenized['input_ids'],
            'offset_mapping': offset_mapping  # For grounding alignment
        }
        
             # Add dummy position_ids to match BERT's interface
        if self.use_sub_sentence_represent:
            # CLIP doesn't need special position handling, but create sequential IDs
            seq_length = tokenized['input_ids'].shape[1]
            position_ids = torch.arange(seq_length, device=device).expand_as(tokenized['input_ids'])
            language_dict_features['position_ids'] = position_ids
            language_dict_features['text_token_mask'] = attention_mask_2d
        else:
            language_dict_features['position_ids'] = None
            language_dict_features['text_token_mask'] =attention_mask_2d
            # Get CLIP's 2D attention mask [bs, seq_len]

        return language_dict_features