"""
Adapted from: https://github.com/openai/CLIP/blob/main/clip/clip.py
"""
from collections import OrderedDict
from typing import Tuple, Union, List

import hashlib
import os
import urllib
import warnings
from tqdm import tqdm
from .module_transformer import Transformer as TransformerClip
from .module_transformer import Attention as TAttention
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
import math
import logging

logger = logging.getLogger(__name__)

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}
_PT_NAME = {
    "RN50": "RN50.pt",
    "RN101": "RN101.pt",
    "RN50x4": "RN50x4.pt",
    "RN50x16": "RN50x16.pt",
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-B/16": "ViT-B-16.pt",
    "ViT-L/14": "ViT-L-14.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def available_models():
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


# =============================

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super(AttentionPool2d, self).__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super(ModifiedResNet, self).__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, lora_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = float(self.head_dim) ** -0.5
        self.lora_dim = lora_dim

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        self.TVPt_LoRA_a = nn.Parameter(torch.zeros(lora_dim, embed_dim))
        nn.init.kaiming_uniform_(self.TVPt_LoRA_a, a=math.sqrt(5))
        self.TVPt_LoRA_b = nn.Parameter(torch.zeros(3 * embed_dim, lora_dim))
    
        # self._reset_parameters() ### lora init
    
    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)
    
    def forward(self, x):
        bsz, tgt_len, embed_dim = x.size()
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q * self.scaling

        q = q.contiguous().view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.contiguous().view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.contiguous().view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        x = F.linear(x, self.out_proj.weight, self.out_proj.bias)
        
        return x, None

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_dim: int, frame_num: int, attn_mask=None):
        super(ResidualAttentionBlock, self).__init__()

        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attn = Attention(d_model, n_head, lora_dim)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

        if frame_num > 0:
            self.TVPt_Video_Positional_embedding = nn.Parameter(torch.zeros(1, frame_num, 1, d_model))

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))  # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]
    
    def forward(self, x, attn_mask=None):
        # x = x + self.attention(self.ln_1(x))
        x = x + self.attn(self.ln_1(x))[0]
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, lora_dim: int, merge_layer: List[int], frame_pos: int, frame_num_list: List[int], attn_mask=None):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers

        if frame_pos == 0:
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, lora_dim, 0, attn_mask) for _ in range(layers)])
        else:
            resblocks_list = []
            idx_pos = 0
            for _l in range(layers):
                if _l in merge_layer:
                    resblocks_list.append(ResidualAttentionBlock(width, heads, lora_dim, frame_num_list[idx_pos], attn_mask))
                    idx_pos += 1
                else:
                    resblocks_list.append(ResidualAttentionBlock(width, heads, lora_dim, 0, attn_mask))
            self.resblocks = nn.Sequential(*resblocks_list)
        
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, lora_dim: int, merge_layer: List[int], frame_pos: int, frame_num_list: List[int]):
        super(VisualTransformer, self).__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, lora_dim, merge_layer, frame_pos, frame_num_list)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        for param in self.conv1.parameters():
            param.requires_grad = False  # not update by gradient

    def forward(self, x: torch.Tensor, mask=None):

        x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 lora_dim: int,
                 merge_layer: List[int],
                 frame_pos: int,
                 frame_num_list: List[int]
                 ):
        super(CLIP, self).__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                lora_dim=lora_dim,
                merge_layer=merge_layer,
                frame_pos=frame_pos,
                frame_num_list=frame_num_list
            )

        self.transformer = TransformerClip(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            lora_dim=lora_dim
            # attn_mask=self.build_attention_mask
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))

        self.token_embedding.requires_grad = False  # not update by gradient

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @staticmethod
    def get_config(pretrained_clip_name="ViT-B/32"):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViT-B-32.pt")
        if pretrained_clip_name in _MODELS and pretrained_clip_name in _PT_NAME:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[pretrained_clip_name])

        if pretrained_clip_name in ["ViT-B/32", "ViT-B/16"] and os.path.exists(model_path):
            pass
        else:
            if pretrained_clip_name in _MODELS:
                model_path = _download(_MODELS[pretrained_clip_name])
            elif os.path.isfile(pretrained_clip_name):
                model_path = pretrained_clip_name
            else:
                raise RuntimeError(f"Model {pretrained_clip_name} not found; available models = {available_models()}")

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        return state_dict

    def build_attention_mask(self, context_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_hidden=False, mask=None):
        hidden = self.visual(image.type(self.dtype))
        hidden = self.visual.ln_post(hidden) @ self.visual.proj

        x = hidden[:, 0, :]

        if return_hidden:
            return x, hidden

        return x

    def encode_text(self, text, return_hidden=False, mask=None):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)
        
        attn_mask = self.build_attention_mask(x.size(1)).repeat(x.size(0), 1, 1).to(mask.device)
        inf = torch.zeros((x.size(1), x.size(1))).fill_(float("-inf")).repeat(x.size(0), 1, 1).to(mask.device)
        mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)
        attn_mask = torch.where(mask>0, attn_mask, inf)
        
        if self.ctx_layers > 0:
            text_end= text.argmax(dim=-1)
            for b_i in range(x.size(0)):
                attn_mask[b_i,text_end[b_i]][-self.ctx_text:]=0
                for t_i in range(1,self.ctx_text):
                    attn_mask[b_i,-(t_i+1)][-t_i:] = 0

        x = x + pos_emd
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection

        x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]

        if return_hidden:
            return x, hidden

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
            
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        if isinstance(l, (Attention, TAttention)):
            for attr in ["in_proj_weight", "in_proj_bias", "TVPt_LoRA_a", "TVPt_LoRA_b"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
            for attr in ["TVPt_LoRA_st_a", "TVPt_LoRA_st_b"]:
                if hasattr(l, attr):
                    tensor = getattr(l, attr)
                    if tensor is not None:
                        tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)