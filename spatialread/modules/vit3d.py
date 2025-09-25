import torch
import torch.nn as nn
from torch.nn import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from einops import rearrange
import einops

from typing import List, Dict
from torch import Tensor



class ViT3D(nn.Module):
    def __init__(self, img_size: int, patch_size: int, nlayers: int, hid_dim: int, nheads: int, ff_dim: int, dropout, batch_first: bool =True, cross_emb_len: int = 300, mask_ratio: float=None, mask_patch_size: int = None, enable_basis: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size ** 3, hid_dim)

        assert img_size % patch_size == 0
        num_patches = (img_size // patch_size)**3
        self.num_patch = img_size // patch_size
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches+1, hid_dim))
    
        decoder_layer = TransformerDecoderLayer(
            hid_dim,
            nheads,
            ff_dim,
            dropout,
            batch_first=batch_first,
        )
        self.encoder = TransformerDecoder(
            decoder_layer, nlayers
        )

        self.cls_emb = nn.Parameter(torch.zeros(hid_dim, dtype=torch.float))
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        if mask_ratio is not None:
            self.mask_emb = nn.Parameter(torch.zeros(hid_dim, dtype=torch.float))

        self.cross_emb = nn.Parameter(
            torch.randn(
                1,
                cross_emb_len,
                hid_dim,
            )
        )

        self.enable_basis = enable_basis
        if enable_basis:
            self.pes_mapper = nn.Linear(18, 1, bias=False)

    def forward(self, img: Tensor, merge_feat: Tensor = None, memory: Tensor = None, memory_key_padding_mask: Tensor = None, **kwargs):
        # img: [bs, num_grid, num_grid, num_grid, 20]
        if self.enable_basis:
            add_term = torch.cat(
                [img[..., :12], img[..., 13:19]], dim=-1
            )
            add_term = self.pes_mapper(add_term).squeeze(-1)
            img = img[..., 12] + add_term
        else:
            if len(img.shape) == 5:
                img = img[..., 12]
            else:
                img = img

        patch_size = self.patch_size

        # step 1: 划分成patches
        img = rearrange(
            img,
            'bs (a p1) (b p2) (c p3) -> bs (a b c) (p1 p2 p3)',
            p1=patch_size, p2=patch_size, p3=patch_size
        )  # => [bs, num_patches, patch_dim]
        emb = self.proj(img)  # [bs, num_patches, hid_dim]

        if self.mask_ratio is not None:
            mask_patch_size = self.mask_patch_size
            N = mask_patch_size // patch_size
            num_patch = self.num_patch
            assert num_patch % N == 0, "grid size must be divisible by N"

            mask_grid = torch.rand(emb.shape[0], num_patch//N, num_patch//N, num_patch//N, device=emb.device) < self.mask_ratio

            mask_grid = mask_grid.repeat_interleave(N, 1).repeat_interleave(N, 2).repeat_interleave(N, 3)
            mask = mask_grid.reshape(emb.shape[0], -1)

            emb[mask] = self.mask_emb.to(emb.dtype)
            label = img[mask]

        # patch_size = self.patch_size
        # img = rearrange(img, 'bs (a p1) (b p2) (c p3) -> bs (a b c) (p1 p2 p3)', p1=patch_size, p2=patch_size, p3=patch_size)
        # emb = self.proj(img) # b img_len hid_dim

        # # random mask tokens
        # if self.mask_ratio is not None:
        #     mask = torch.rand(emb.shape[0], emb.shape[1], device=emb.device) < self.mask_ratio
        #     # emb[mask] = self.mask_emb
        #     emb[mask] = self.mask_emb.to(emb.dtype)
        #     label = img[mask]
        
        if merge_feat is not None:
            # emb += merge_feat
            emb = merge_feat

        # print("DEBUG", emb.shape, self.cls_emb.shape)
        emb = torch.cat([self.cls_emb.reshape(1, 1, -1).repeat(emb.shape[0], 1, 1), emb], dim=1) # bs img_len+1 hid_dim
        emb = emb + self.pos_emb.to(emb.dtype)
        # feat = self.encoder(img) # [bs, img_len+1, hid_dim]

        if memory is None:
            feat = self.encoder(
                emb,
                self.cross_emb.repeat(emb.shape[0], 1, 1),
            )
        else:
            feat = self.encoder(
                emb,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
                **kwargs
            ) # [bs, img_len+1, hid_dim]
        
        if self.mask_ratio is not None:
            return feat[:, 0, :], feat[:, 1:, :], mask, label
        return feat[:, 0, :], feat[:, 1:, :]
