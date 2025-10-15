#!/usr/bin/env python3

import os
import math
import random
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import timm
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
import joblib
import torch.nn.functional as F


def gaussian_nll_log_targets(y_log, mu, logvar, logvar_min=-1.0, logvar_max=5.0):
    """
    Stable Gaussian negative log-likelihood for log-targets.

    - Expects inputs on the log scale (e.g. log1p(price)).
    - Clamps log-variance so the loss doesn't blow up or go strongly negative.
    - Returns the mean NLL over the batch.
    """
    logvar_clamped = torch.clamp(logvar, min=logvar_min, max=logvar_max)
    inv_var = torch.exp(-logvar_clamped)
    nll = 0.5 * (inv_var * (y_log - mu) ** 2 + logvar_clamped)
    return nll.mean()

nll_loss_gaussian = gaussian_nll_log_targets

# -------------------- Utilities --------------------
def load_config(path: Path):
    # load YAML config from disk
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    # set random seeds for reproducibility across python, numpy and torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic cuDNN (can slow training but helps reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def smape_np(y_true, y_pred, eps=1e-6):
    # SMAPE implemented with numpy (percentage)
    num = np.abs(y_pred - y_true)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return np.mean(num / denom) * 100.0


def smape_torch(y_true, y_pred, eps=1e-6):
    # SMAPE implemented with torch tensors (percentage)
    num = torch.abs(y_pred - y_true)
    denom = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0 + eps
    return torch.mean(num / denom) * 100.0

# -------------------- Dataset --------------------
class MultimodalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str, tokenizer, image_transform, text_max_len=128,
                 is_train=True, numeric_cols=None, img_size=224):
        # keep a local copy of dataframe and transforms/tokenizer
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.text_max_len = text_max_len
        self.is_train = is_train
        # default numeric features used by the model
        self.numeric_cols = numeric_cols or ['amount_per_unit_log1p','has_quantity','is_weight','is_volume','is_count','pack_count']
        # ensure numeric columns exist in dataframe (fill with zeros if missing)
        for c in self.numeric_cols:
            if c not in self.df.columns:
                self.df[c] = 0.0

        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def _get_image_path(self, image_link: str):
        # extract filename from URL-like image_link and join with images_dir
        if not isinstance(image_link, str) or image_link.strip() == '':
            return None
        name = image_link[image_link.rfind('/')+1:]
        return os.path.join(self.images_dir, name)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # tokenize text field (catalog_content)
        text = str(row.get('catalog_content', ''))
        tokenized = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.text_max_len, return_tensors='pt')
        # remove batch dimension returned by tokenizer
        for k,v in tokenized.items():
            tokenized[k] = v.squeeze(0)

        # load image if present; otherwise create a black image as a placeholder
        img_path = self._get_image_path(row.get('image_link', ''))
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                # if image is corrupted, use a blank image
                img = Image.new('RGB', (self.img_size, self.img_size), (0,0,0))
        else:
            # missing image -> blank placeholder
            img = Image.new('RGB', (self.img_size, self.img_size), (0,0,0))

        img = self.image_transform(img)
        # numeric features as float tensor
        nums = torch.tensor(row[self.numeric_cols].fillna(0).values.astype(np.float32), dtype=torch.float)
        if self.is_train:
            # target is log1p(price) for stability
            price = float(row['price'])
            y = torch.tensor(np.log1p(price), dtype=torch.float)
        else:
            y = torch.tensor(0.0, dtype=torch.float)
        return tokenized, img, nums, y


def collate_fn(batch):
    # combine samples into a batch; tokenized fields are stacked
    batch_token = {}
    keys = batch[0][0].keys()
    for k in keys:
        batch_token[k] = torch.stack([b[0][k] for b in batch], dim=0)
    images = torch.stack([b[1] for b in batch], dim=0)
    nums = torch.stack([b[2] for b in batch], dim=0)
    ys = torch.stack([b[3] for b in batch], dim=0)
    return batch_token, images, nums, ys

# -------------------- Model components --------------------
class ImageEncoder(nn.Module):
    def __init__(self, backbone_name='vit_base_patch16_224', pretrained=True, out_dim=512, return_patch_tokens=False):
        super().__init__()
        self.return_patch_tokens = return_patch_tokens
        # use timm to construct a vision backbone without a classification head
        gp = '' if return_patch_tokens else 'avg'
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool=gp)
        feat_dim = self.backbone.num_features
        self.proj = nn.Linear(feat_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # try to use the backbone's forward_features if available (more robust)
        try:
            f = self.backbone.forward_features(x)
        except Exception:
            f = self.backbone(x)

        # backbone may return pooled vectors [B, E] or patch tokens [B, N, E]
        if f.dim() == 2:
            out = self.proj(f)
            out = self.norm(out)
            return out
        elif f.dim() == 3:
            B, N, D = f.shape
            f_proj = self.proj(f.reshape(B * N, D)).reshape(B, N, -1)
            out = self.norm(f_proj)
            return out
        else:
            # fall back to a simple pooling if output shape is unexpected
            f_pool = f.mean(dim=tuple(range(2, f.dim()))) if f.dim() > 2 else f
            out = self.proj(f_pool)
            out = self.norm(out)
            return out

class TextEncoder(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', out_dim=512):
        super().__init__()
        # load transformer text model and add a projection to desired embedding size
        self.text_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.text_model.config.hidden_size
        self.proj = nn.Linear(hidden_size, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def mean_pooling(self, last_hidden_state, attention_mask):
        # pool token embeddings using the attention mask (avoid counting padded tokens)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        pooled = self.mean_pooling(last_hidden, attention_mask)
        out = self.proj(pooled)
        out = self.norm(out)
        return out

# Fusion module producing mu & logvar
class PatchAwareModalityTransformerFusion(nn.Module):
    def __init__(self, embed_dim=512, proj_dim=384, num_numeric=6,
                 n_layers=4, n_heads=8, dim_feedforward=None, dropout=0.1,
                 use_fuse_token=True):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = proj_dim * 4

        # small projection networks to map each modality into the common space
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )
        self.image_proj = nn.Linear(embed_dim, proj_dim)
        self.image_norm = nn.LayerNorm(proj_dim)
        self.num_proj = nn.Sequential(
            nn.Linear(num_numeric, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )

        # modality embeddings help the transformer tell modalities apart
        self.modality_embed = nn.Embedding(3, proj_dim)

        self.use_fuse_token = use_fuse_token
        if self.use_fuse_token:
            # optional learnable token that collects fused information
            self.fuse_token = nn.Parameter(torch.randn(1, 1, proj_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # two small heads: one for predicted mean (mu) and one for log-variance
        self.readout_mu = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        self.readout_logvar = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, text_emb, img_emb, numeric):
        B = text_emb.size(0)
        device = text_emb.device

        t = self.text_proj(text_emb)  # text -> projected token
        # handle image embeddings that may be pooled [B, E] or patch tokens [B, N, E]
        if img_emb.dim() == 2:
            v = self.image_proj(img_emb)
            v = self.image_norm(v)
            v = v.unsqueeze(1)  # make it a length-1 token sequence
        elif img_emb.dim() == 3:
            Bimg, N, E = img_emb.size()
            v_reshaped = img_emb.reshape(Bimg * N, E)
            v_proj = self.image_proj(v_reshaped).reshape(Bimg, N, -1)
            v_proj = self.image_norm(v_proj)
            v = v_proj
        else:
            raise ValueError('img_emb has unexpected dim: %d' % img_emb.dim())

        n = self.num_proj(numeric)
        n = n.unsqueeze(1)  # numeric features become a single token

        # modality-specific embeddings (one per modality)
        text_mod = self.modality_embed(torch.tensor(0, device=device)).unsqueeze(0)
        image_mod = self.modality_embed(torch.tensor(1, device=device)).unsqueeze(0)
        num_mod = self.modality_embed(torch.tensor(2, device=device)).unsqueeze(0)

        # add modality identifiers
        t = t.unsqueeze(1) + text_mod
        v = v + image_mod
        n = n + num_mod

        tokens_list = []
        if self.use_fuse_token:
            # place fuse token at the beginning so the transformer can write to it
            fuse = self.fuse_token.expand(B, -1, -1)
            tokens_list.append(fuse)

        tokens_list.append(t)
        tokens_list.append(v)
        tokens_list.append(n)

        tokens = torch.cat(tokens_list, dim=1)  # sequence of tokens for transformer
        out = self.transformer(tokens)  # run cross-modal attention

        # readout: either take the fuse token or average all tokens
        if self.use_fuse_token:
            read_token = out[:, 0, :]
        else:
            read_token = out.mean(dim=1)

        # heads for mean and log-variance
        mu = self.readout_mu(read_token).squeeze(-1)
        raw_logvar = self.readout_logvar(read_token).squeeze(-1)

        # ensure variance is positive and numerically stable
        min_var = 1e-2
        var = F.softplus(raw_logvar) + min_var
        logvar = torch.log(var)

        return mu, logvar


class FusionRegressor(nn.Module):
    def __init__(self, image_encoder: Optional[ImageEncoder] = None, text_encoder: Optional[TextEncoder] = None,
                 image_backbone_cfg='vit_base_patch16_224', text_model_cfg='sentence-transformers/all-mpnet-base-v2',
                 num_numeric=6, embed_dim=512, proj_dim=384, use_patch_tokens=False, fusion_kwargs=None):
        super().__init__()
        # allow passing shared encoders; otherwise construct fresh ones
        if image_encoder is None:
            self.image_encoder = ImageEncoder(backbone_name=image_backbone_cfg, out_dim=embed_dim, return_patch_tokens=use_patch_tokens)
        else:
            self.image_encoder = image_encoder

        if text_encoder is None:
            self.text_encoder = TextEncoder(model_name=text_model_cfg, out_dim=embed_dim)
        else:
            self.text_encoder = text_encoder

        fusion_kwargs = fusion_kwargs or {}
        self.fusion = PatchAwareModalityTransformerFusion(embed_dim=embed_dim, proj_dim=proj_dim, num_numeric=num_numeric, **fusion_kwargs)

    def forward(self, tokenized_batch, images, numeric):
        input_ids = tokenized_batch['input_ids']
        attention_mask = tokenized_batch['attention_mask']
        text_emb = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)  # [B, E]
        img_emb = self.image_encoder(images)  # [B, E] or [B, N, E]
        mu, logvar = self.fusion(text_emb, img_emb, numeric)
        return mu, logvar

# Image-only regressor uses shared image encoder if provided
class ImageOnlyRegressor(nn.Module):
    """
    Image-only model that supports pooled image vectors or patch token sequences.

    If patch tokens are present it applies a small attention pooling to get a single
    image embedding before the prediction head.
    """
    def __init__(self, image_encoder: Optional[ImageEncoder] = None,
                 image_backbone_cfg='vit_base_patch16_224',
                 embed_dim=512, proj_dim=256, num_out=1):
        super().__init__()
        if image_encoder is None:
            # default uses pooled embedding unless encoder returns patches
            self.image_encoder = ImageEncoder(backbone_name=image_backbone_cfg,
                                              out_dim=embed_dim, return_patch_tokens=False)
        else:
            self.image_encoder = image_encoder

        # small network to score each patch and pool by attention
        self.pool_attn = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)   # score per patch
        )

        # final regression head from pooled embedding -> scalar
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, num_out)
        )

    def forward(self, images):
        emb = self.image_encoder(images)  # -> [B, E] OR [B, N, E]
        if emb.dim() == 3:
            # attention pooling over patches -> single vector per image
            scores = self.pool_attn(emb)            # [B, N, 1]
            scores = scores.squeeze(-1)             # [B, N]
            weights = torch.softmax(scores, dim=1)  # [B, N]
            pooled = (emb * weights.unsqueeze(-1)).sum(dim=1)
        else:
            pooled = emb  # already a pooled embedding

        out = self.head(pooled).squeeze(-1)  # -> [B]
        return out

# Text+numeric regressor uses shared text encoder if provided
class TextNumRegressor(nn.Module):
    def __init__(self, text_encoder: Optional[TextEncoder] = None, text_model_cfg='sentence-transformers/all-mpnet-base-v2',
                 embed_dim=512, proj_dim=256, num_numeric=6):
        super().__init__()
        if text_encoder is None:
            self.text_encoder = TextEncoder(model_name=text_model_cfg, out_dim=embed_dim)
        else:
            self.text_encoder = text_encoder
        # simple head that concatenates text embedding + numeric features
        self.head = nn.Sequential(
            nn.Linear(embed_dim + num_numeric, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, 1)
        )

    def forward(self, tokenized_batch, numeric):
        input_ids = tokenized_batch['input_ids']
        attention_mask = tokenized_batch['attention_mask']
        t_emb = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = torch.cat([t_emb, numeric], dim=1)
        out = self.head(x).squeeze(-1)
        return out

# -------------------- Training helpers --------------------
def gather_unique_param_ids(modules: List[nn.Module]):
    """
    Return a mapping from parameter id to parameter for the provided modules.

    This helps ensure parameters shared between models (shared encoders) are not
    included multiple times when building an optimizer.
    """
    id_to_param = {}
    for m in modules:
        for p in m.parameters():
            id_to_param[id(p)] = p
    return id_to_param


def build_optimizer_shared(modules: List[nn.Module], cfg: dict):
    """
    Build AdamW optimizer over unique parameters and create named groups:
      - 'head' : non-encoder params
      - 'image': image encoder params
      - 'text' : text encoder params

    Supports optionally freezing encoders for some initial epochs (lr=0 + requires_grad=False).
    """
    base_lr = float(cfg.get('lr', 3e-4))
    image_lr_scale = float(cfg.get('image_lr_scale', 0.3))
    text_lr_scale = float(cfg.get('text_lr_scale', 0.3))
    weight_decay = float(cfg.get('weight_decay', 1e-2))

    freeze_epochs = int(cfg.get('encoder_freeze_initial_epochs', 0))

    # locate the first image/text encoder objects among the modules
    image_enc = None
    text_enc = None
    for m in modules:
        if hasattr(m, 'image_encoder') and image_enc is None:
            image_enc = getattr(m, 'image_encoder')
        if hasattr(m, 'text_encoder') and text_enc is None:
            text_enc = getattr(m, 'text_encoder')

    # collect unique params across modules
    id_to_param = gather_unique_param_ids(modules)
    # collect ids belonging to image/text encoders (if found)
    image_param_ids = set()
    text_param_ids = set()
    if image_enc is not None:
        for p in image_enc.parameters():
            image_param_ids.add(id(p))
    if text_enc is not None:
        for p in text_enc.parameters():
            text_param_ids.add(id(p))

    head_params = []
    image_params = []
    text_params = []
    for pid, p in id_to_param.items():
        if pid in image_param_ids:
            image_params.append(p)
        elif pid in text_param_ids:
            text_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if len(head_params) > 0:
        param_groups.append({'params': head_params, 'lr': base_lr, 'weight_decay': weight_decay, 'group_name': 'head'})
    if len(image_params) > 0:
        # optionally start frozen by setting lr=0
        lr_img = 0.0 if freeze_epochs > 0 else base_lr * image_lr_scale
        param_groups.append({'params': image_params, 'lr': lr_img, 'weight_decay': weight_decay, 'group_name': 'image'})
    if len(text_params) > 0:
        lr_txt = 0.0 if freeze_epochs > 0 else base_lr * text_lr_scale
        param_groups.append({'params': text_params, 'lr': lr_txt, 'weight_decay': weight_decay, 'group_name': 'text'})

    optimizer = torch.optim.AdamW(param_groups)

    # if encoders should be frozen initially, also disable gradients
    if freeze_epochs > 0:
        if image_enc is not None:
            for p in image_enc.parameters():
                p.requires_grad = False
        if text_enc is not None:
            for p in text_enc.parameters():
                p.requires_grad = False

    return optimizer


def smape_surrogate_torch(true_price, pred_price, eps=1e-6):
    # surrogate SMAPE working on raw price tensors (not log-scale)
    denom = (torch.abs(true_price) + torch.abs(pred_price)) / 2.0 + eps
    return torch.mean(torch.abs(pred_price - true_price) / denom) * 100.0


def rel_mae_torch(true_price, pred_price, eps=1e-6):
    # relative MAE (division by ground truth)
    return torch.mean(torch.abs(pred_price - true_price) / (true_price + eps))


# ----------------- Gradual unfreeze helpers -----------------
def _get_sequential_layers(module: nn.Module):
    """
    Try to get a list-like container of the module's transformer/blocks so we can
    unfreeze them gradually. If common attributes (blocks, layers, stages) are not
    present, fall back to the module's children.
    """
    # common patterns for ViT and transformer stacks
    if hasattr(module, 'blocks'):
        try:
            return list(module.blocks)
        except Exception:
            pass
    if hasattr(module, 'layers'):
        try:
            return list(module.layers)
        except Exception:
            pass
    if hasattr(module, 'stages'):
        try:
            return list(module.stages)
        except Exception:
            pass
    # fallback: coarse-grained children
    return list(module.children())


def gradual_unfreeze_module(module: nn.Module, proportion: float, extra_params_to_unfreeze: Optional[List[nn.Parameter]] = None):
    """
    Unfreeze the last `proportion` fraction of the module's blocks (unfreeze newest
    layers first). Also attempt to unfreeze obvious projection/norm layers.

    If the module has no block-like structure, unfreeze the whole module when proportion>0.
    """
    if proportion <= 0.0:
        return
    proportion = min(max(proportion, 0.0), 1.0)

    blocks = _get_sequential_layers(module)
    L = len(blocks)
    if L == 0:
        # nothing to split — unfreeze everything
        for p in module.parameters():
            p.requires_grad = True
        return

    num_to_unfreeze = int(math.ceil(L * proportion))
    num_to_unfreeze = min(max(num_to_unfreeze, 1), L)

    # unfreeze the last `num_to_unfreeze` blocks
    for blk in blocks[-num_to_unfreeze:]:
        for p in blk.parameters():
            p.requires_grad = True

    # also try to unfreeze likely useful modules like proj/norm
    for name in ['proj', 'proj_layer', 'norm', 'layernorm', 'ln']:
        if hasattr(module, name):
            attr = getattr(module, name)
            try:
                for p in attr.parameters():
                    p.requires_grad = True
            except Exception:
                pass

    if extra_params_to_unfreeze is not None:
        for p in extra_params_to_unfreeze:
            if p is not None:
                p.requires_grad = True


def _identify_shared_encoders(modules: List[nn.Module]):
    """
    Find the first image_encoder and text_encoder objects among the provided modules.
    Useful when multiple models share the same encoder instances.
    """
    image_enc = None
    text_enc = None
    for m in modules:
        if hasattr(m, 'image_encoder') and image_enc is None:
            image_enc = getattr(m, 'image_encoder')
        if hasattr(m, 'text_encoder') and text_enc is None:
            text_enc = getattr(m, 'text_encoder')
    return image_enc, text_enc

# ----------------- Combined training: main + optional aux (shared encoders) -----------------
def train_one_epoch(model_main, img_aux, txt_aux, loader, optimizer, scaler, device, epoch, cfg,
                    shared_img_enc=None, shared_txt_enc=None, freeze_cfg=None):
    # run one epoch of training for main model, optionally training aux models too
    model_main.train()
    if img_aux is not None:
        img_aux.train()
    if txt_aux is not None:
        txt_aux.train()

    running_loss = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Train E{epoch}')
    lambda_nll = cfg.get('loss_lambda_nll', 0.7)
    lambda_smape = cfg.get('loss_lambda_smape', 0.25)
    lambda_rel = cfg.get('loss_lambda_rel', 0.05)
    aux_loss_weight = cfg.get('aux_loss_weight', 0.5)  # relative weight of aux tasks
    logvar_min = cfg.get('logvar_min', -1.0)
    grad_accum = max(1, int(cfg.get('grad_accum_steps', 1)))
    l1_criterion = nn.L1Loss()

    for step, (tokenized, images, numeric, ys_log) in pbar:
        for k in tokenized:
            tokenized[k] = tokenized[k].to(device)
        images = images.to(device)
        numeric = numeric.to(device)
        ys_log = ys_log.to(device)

        with torch.cuda.amp.autocast(enabled=cfg.get('use_amp', True)):
            # main model forward and losses
            mu, logvar = model_main(tokenized, images, numeric)
            loss_main_nll = nll_loss_gaussian(ys_log, mu, logvar, logvar_min=logvar_min)
            pred_price = torch.expm1(mu)
            true_price = torch.expm1(ys_log)
            loss_smape = smape_surrogate_torch(true_price, pred_price)
            loss_rel = rel_mae_torch(true_price, pred_price)
            # weighted combination of NLL + surrogate metrics
            loss_main = lambda_nll * loss_main_nll + lambda_smape * (loss_smape / 100.0) + lambda_rel * loss_rel

            # small penalty to avoid extremely small predicted variance
            var_penalty_weight = float(cfg.get('var_penalty', 0.5))
            var_penalty_thresh = float(cfg.get('var_penalty_thresh', -1.0))
            if var_penalty_weight > 0:
                var_pen = var_penalty_weight * torch.mean(torch.square(torch.relu(var_penalty_thresh - logvar)))
            else:
                var_pen = 0.0
            loss_main = loss_main + var_pen

            # compute auxiliary losses if aux models exist
            loss_aux = 0.0
            if img_aux is not None:
                out_img_log = img_aux(images)  # predicts log1p price
                loss_img = l1_criterion(out_img_log, ys_log)  # L1 on log scale
                loss_aux += loss_img
            if txt_aux is not None:
                out_txt_log = txt_aux(tokenized, numeric)
                loss_txt = l1_criterion(out_txt_log, ys_log)
                loss_aux += loss_txt

            if (img_aux is not None) or (txt_aux is not None):
                loss_total = loss_main + aux_loss_weight * loss_aux
            else:
                loss_total = loss_main

            loss_step = loss_total / grad_accum
        scaler.scale(loss_step).backward()

        if (step + 1) % grad_accum == 0:
            # unscale and clip gradients before optimizer step (AMP-safe)
            try:
                scaler.unscale_(optimizer)
            except Exception:
                # not all setups expose unscale_; continue if unavailable
                pass

            max_norm = float(cfg.get('grad_clip', 5.0))
            params = [p for g in optimizer.param_groups for p in g['params'] if p.grad is not None]
            if len(params) > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        running_loss += float(loss_step.item()) * grad_accum
        if step % cfg.get('print_freq', 100) == 0:
            # small diagnostics printed on progress bar
            logvar_clamped = torch.clamp(logvar, min=logvar_min)
            inv_var = torch.exp(-logvar_clamped)
            mse_term = 0.5 * (inv_var * (ys_log - mu) ** 2).mean().item()
            log_term = 0.5 * logvar_clamped.mean().item()
            smape_term = (loss_smape / 100.0).item()
            var_pen_val = var_pen.item() if isinstance(var_pen, torch.Tensor) else var_pen
            pbar.set_postfix({
                'loss': running_loss / (step + 1),
                'nll_mse': f'{mse_term:.4f}',
                'nll_log': f'{log_term:.4f}',
                'smape%': f'{loss_smape:.3f}',
                'var_pen': f'{var_pen_val:.4f}',
                'logvar_mean': float(logvar_clamped.mean().detach().cpu().numpy())
            })

    return running_loss / len(loader)

@torch.no_grad()
def validate_all(model_main, img_aux, txt_aux, loader, device):
    # run validation and return predictions + simple metrics for each model
    model_main.eval()
    if img_aux is not None:
        img_aux.eval()
    if txt_aux is not None:
        txt_aux.eval()

    preds_main = []
    logvars_main = []
    trues = []

    preds_img = []
    preds_txt = []

    for tokenized, images, numeric, ys_log in tqdm(loader, desc='Valid'):
        for k in tokenized:
            tokenized[k] = tokenized[k].to(device)
        images = images.to(device)
        numeric = numeric.to(device)

        # main model predictions
        mu, logvar = model_main(tokenized, images, numeric)
        preds_main.append(np.expm1(mu.detach().cpu().numpy()))
        logvars_main.append(logvar.detach().cpu().numpy())
        trues.append(np.expm1(ys_log.numpy()))
        if len(logvars_main) % 100 == 0:
            print('debug: recent logvar mean', np.mean(logvars_main[-1]))

        # aux predictions (if any)
        if img_aux is not None:
            out_img = img_aux(images)
            preds_img.append(np.expm1(out_img.detach().cpu().numpy()))
        if txt_aux is not None:
            out_txt = txt_aux(tokenized, numeric)
            preds_txt.append(np.expm1(out_txt.detach().cpu().numpy()))

    preds_main = np.concatenate(preds_main, axis=0)
    logvars_main = np.concatenate(logvars_main, axis=0)
    trues = np.concatenate(trues, axis=0)

    metrics_main = {
        'smape': smape_np(trues, preds_main),
        'mae': np.mean(np.abs(trues - preds_main)),
        'mae_log': np.mean(np.abs(np.log1p(trues) - np.log1p(preds_main)))
    }

    metrics_img = None
    metrics_txt = None

    if img_aux is not None:
        preds_img = np.concatenate(preds_img, axis=0)
        metrics_img = {
            'smape': smape_np(trues, preds_img),
            'mae': np.mean(np.abs(trues - preds_img)),
            'mae_log': np.mean(np.abs(np.log1p(trues) - np.log1p(preds_img)))
        }
    if txt_aux is not None:
        preds_txt = np.concatenate(preds_txt, axis=0)
        metrics_txt = {
            'smape': smape_np(trues, preds_txt),
            'mae': np.mean(np.abs(trues - preds_txt)),
            'mae_log': np.mean(np.abs(np.log1p(trues) - np.log1p(preds_txt)))
        }

    return metrics_main, metrics_img, metrics_txt, trues, preds_main, logvars_main, (preds_img if img_aux is not None else None), (preds_txt if txt_aux is not None else None)

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='path to config.yaml (default: repo root config.yaml)')
    args = parser.parse_args()

    if args.config:
        cfg_path = Path(args.config)
    else:
        # default to parent directory's config.yaml
        cfg_path = Path(__file__).resolve().parents[1] / 'config.yaml'
    cfg = load_config(cfg_path)

    set_seed(cfg.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    out_dir = Path(cfg.get('out_dir', 'outputs'))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg['data_csv'])
    print('Loaded', len(df), 'rows')

    # create price buckets for stratified CV (tries qcut then falls back to simple cutting)
    q = cfg.get('price_bucket_q', 10)
    try:
        df['price_rank'] = df['price'].rank(method='first')
        df['price_bucket'] = pd.qcut(df['price_rank'], q=q, labels=False, duplicates='drop')
    except Exception:
        df['price_bucket'] = pd.cut(df['price'], bins=q, labels=False, duplicates='drop')

    num_folds = cfg.get('num_folds', 5)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=cfg.get('seed', 42))

    tokenizer = AutoTokenizer.from_pretrained(cfg.get('text_model'))
    global CFG_IMG_SIZE
    CFG_IMG_SIZE = cfg.get('img_size', 224)

    # simple train/val image transforms
    image_transform_train = transforms.Compose([
        transforms.Resize((CFG_IMG_SIZE, CFG_IMG_SIZE)),
        transforms.RandomResizedCrop(CFG_IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_transform_val = transforms.Compose([
        transforms.Resize((CFG_IMG_SIZE, CFG_IMG_SIZE)),
        transforms.CenterCrop(CFG_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    oof_main = np.zeros(len(df), dtype=np.float32)
    oof_main_logvar = np.zeros(len(df), dtype=np.float32)

    train_aux = cfg.get('train_aux_models', True)
    oof_img = np.zeros(len(df), dtype=np.float32) if train_aux else None
    oof_text = np.zeros(len(df), dtype=np.float32) if train_aux else None

    fold_idx = 0
    for train_idx, val_idx in skf.split(df, df['price_bucket']):
        fold_idx += 1
        print(f'===== Fold {fold_idx}/{num_folds} =====')
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        numeric_cols = cfg.get('numeric_cols', ['amount_per_unit_log1p','has_quantity','is_weight','is_volume','is_count','pack_count'])

        train_ds = MultimodalDataset(train_df, cfg.get('images_dir'), tokenizer, image_transform_train, text_max_len=cfg.get('text_max_len', 384), is_train=True, numeric_cols=numeric_cols, img_size=CFG_IMG_SIZE)
        val_ds = MultimodalDataset(val_df, cfg.get('images_dir'), tokenizer, image_transform_val, text_max_len=cfg.get('text_max_len', 384), is_train=True, numeric_cols=numeric_cols, img_size=CFG_IMG_SIZE)

        train_loader = DataLoader(train_ds, batch_size=cfg.get('batch_size', 8), shuffle=True, num_workers=cfg.get('num_workers', 8), collate_fn=collate_fn, pin_memory=cfg.get('pin_memory', True))
        val_loader = DataLoader(val_ds, batch_size=cfg.get('batch_size', 8), shuffle=False, num_workers=cfg.get('num_workers', 8), collate_fn=collate_fn, pin_memory=cfg.get('pin_memory', True))

        # ---- create shared encoders ----
        shared_img_enc = ImageEncoder(backbone_name=cfg.get('image_backbone'), out_dim=cfg.get('embed_dim', 512), return_patch_tokens=bool(cfg.get('use_patch_tokens', True)))
        shared_txt_enc = TextEncoder(model_name=cfg.get('text_model'), out_dim=cfg.get('embed_dim', 512))

        # create models that share encoders
        model_main = FusionRegressor(image_encoder=shared_img_enc, text_encoder=shared_txt_enc,
                                     num_numeric=len(numeric_cols),
                                     embed_dim=cfg.get('embed_dim', 512),
                                     proj_dim=cfg.get('proj_dim', 384),
                                     use_patch_tokens=bool(cfg.get('use_patch_tokens', True)),
                                     fusion_kwargs={
                                         'n_layers': int(cfg.get('fusion_n_layers', 4)),
                                         'n_heads': int(cfg.get('fusion_n_heads', 8)),
                                         'dim_feedforward': int(cfg.get('fusion_dim_feedforward', 1536)),
                                         'dropout': float(cfg.get('fusion_dropout', 0.1)),
                                         'use_fuse_token': bool(cfg.get('fusion_use_fuse_token', True))
                                     })
        model_main.to(device)

        if train_aux:
            img_model = ImageOnlyRegressor(image_encoder=shared_img_enc, embed_dim=cfg.get('embed_dim', 512), proj_dim=cfg.get('aux_proj_dim', 256))
            txt_model = TextNumRegressor(text_encoder=shared_txt_enc, embed_dim=cfg.get('embed_dim', 512), proj_dim=cfg.get('aux_proj_dim', 256), num_numeric=len(numeric_cols))
            img_model.to(device)
            txt_model.to(device)
        else:
            img_model = None
            txt_model = None

        # Build single optimizer over unique params across models
        modules_for_optim = [model_main]
        if train_aux:
            modules_for_optim.append(img_model)
            modules_for_optim.append(txt_model)
        optimizer = build_optimizer_shared(modules_for_optim, cfg)

        # identify shared encoders for later unfreeze operations
        identified_img_enc, identified_txt_enc = _identify_shared_encoders(modules_for_optim)
        if identified_img_enc is not None:
            print('Identified shared image encoder:', type(identified_img_enc), 'params frozen=' , any(not p.requires_grad for p in identified_img_enc.parameters()))
        if identified_txt_enc is not None:
            print('Identified shared text encoder:', type(identified_txt_enc), 'params frozen=' , any(not p.requires_grad for p in identified_txt_enc.parameters()))

        total_steps = int(len(train_loader) * cfg.get('epochs', 12) / max(1, cfg.get('grad_accum_steps', 1)))
        warmup_steps = int(total_steps * cfg.get('warmup_ratio', 0.05))
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        scaler = torch.amp.GradScaler(enabled=cfg.get('use_amp', True), device='cuda')

        # --- config for freezing/unfreezing ---
        freeze_epochs = int(cfg.get('encoder_freeze_initial_epochs', 0))
        unfreeze_epochs = int(cfg.get('encoder_unfreeze_epochs', 3))
        encoder_unfreeze_lr_scale = float(cfg.get('encoder_unfreeze_lr_scale', 0.33))
        base_lr = float(cfg.get('lr', 3e-4))
        image_lr_scale = float(cfg.get('image_lr_scale', 0.3))
        text_lr_scale = float(cfg.get('text_lr_scale', 0.3))

        # Helper: update optimizer lr for group_name
        def _set_optimizer_group_lr(opt, group_name, lr_val):
            for g in opt.param_groups:
                if g.get('group_name', None) == group_name:
                    g['lr'] = lr_val

        best_smape = float('inf')
        best_ckpt = None
        no_improve = 0
        patience = cfg.get('early_stop_patience', None)

        for epoch in range(1, cfg.get('epochs', 12) + 1):
            # handle gradual unfreeze schedule at epoch boundaries BEFORE training
            if (freeze_epochs > 0) and (epoch <= freeze_epochs):
                # still frozen - nothing to do (they were frozen earlier in build_optimizer_shared)
                if epoch == 1:
                    print(f'Encoders frozen for initial {freeze_epochs} epochs')
            else:
                # start or continue unfreezing
                # calculate proportion of unfreeze completed in [0,1]
                if unfreeze_epochs <= 0:
                    prop = 1.0
                else:
                    # epoch - freeze_epochs ranges from 1 .. unfreeze_epochs -> proportion
                    prop = min(1.0, max(0.0, float(epoch - freeze_epochs) / float(unfreeze_epochs)))

                if prop > 0.0:
                    # unfreeze image encoder progressively
                    if identified_img_enc is not None:
                        gradual_unfreeze_module(identified_img_enc.backbone if hasattr(identified_img_enc, 'backbone') else identified_img_enc,
                                                 prop,
                                                 extra_params_to_unfreeze=[getattr(identified_img_enc, 'proj', None), getattr(identified_img_enc, 'norm', None)])
                        # set optimizer lr for image params to a low value (low-LR fine-tune)
                        low_img_lr = base_lr * image_lr_scale * encoder_unfreeze_lr_scale
                        _set_optimizer_group_lr(optimizer, 'image', low_img_lr)
                    # unfreeze text encoder progressively
                    if identified_txt_enc is not None:
                        # try to pass inner transformer module if available
                        txt_inner = getattr(identified_txt_enc, 'text_model', identified_txt_enc)
                        gradual_unfreeze_module(getattr(txt_inner, 'encoder', txt_inner),
                                                 prop,
                                                 extra_params_to_unfreeze=[getattr(identified_txt_enc, 'proj', None), getattr(identified_txt_enc, 'norm', None)])
                        low_txt_lr = base_lr * text_lr_scale * encoder_unfreeze_lr_scale
                        _set_optimizer_group_lr(optimizer, 'text', low_txt_lr)

                    print(f'Epoch {epoch}: unfreeze proportion={prop:.3f} -> image_lr={[g["lr"] for g in optimizer.param_groups if g.get("group_name")=="image"]} text_lr={[g["lr"] for g in optimizer.param_groups if g.get("group_name")=="text"]}')

            train_loss = train_one_epoch(model_main, img_model, txt_model, train_loader, optimizer, scaler, device, epoch, cfg,
                                         shared_img_enc=identified_img_enc, shared_txt_enc=identified_txt_enc,
                                         freeze_cfg={'freeze_epochs': freeze_epochs, 'unfreeze_epochs': unfreeze_epochs})
            scheduler.step()

            metrics_main, metrics_img, metrics_txt, trues, preds_main, logvars_main, preds_img_val, preds_txt_val = validate_all(model_main, img_model, txt_model, val_loader, device)
            val_smape = metrics_main['smape']
            val_mae = metrics_main['mae']
            val_mae_log = metrics_main['mae_log']
            print(f'Fold {fold_idx} Epoch {epoch} | train_loss={train_loss:.4f} | val_smape={val_smape:.4f}% | val_mae={val_mae:.4f} | val_mae_log={val_mae_log:.6f}')
            if train_aux and metrics_img is not None and metrics_txt is not None:
                print(f'  Aux Img val_smape={metrics_img["smape"]:.4f}% | Aux Txt val_smape={metrics_txt["smape"]:.4f}%')

            # save periodic
            save_every = cfg.get('save_every_epochs', 10)
            if save_every > 0 and (epoch % save_every == 0):
                ckpt_path = out_dir / f'fold{fold_idx}_epoch{epoch}_smape{val_smape:.4f}.pth'
                torch.save({'model_state_dict': model_main.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'cfg': cfg}, ckpt_path)
                print('Saved periodic checkpoint:', ckpt_path)

            # save best
            if val_smape < best_smape:
                best_smape = val_smape
                best_ckpt = out_dir / f'fold{fold_idx}_best_smape{best_smape:.4f}.pth'
                torch.save({'model_state_dict': model_main.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'cfg': cfg}, best_ckpt)
                print('Saved new best checkpoint:', best_ckpt)
                no_improve = 0
            else:
                no_improve += 1

            if patience is not None and no_improve >= patience:
                print(f'No improvement for {no_improve} epochs. Early stopping fold {fold_idx}.')
                break

        # final predictions for val set for OOF
        all_preds = []
        all_logvars = []
        all_preds_img = []
        all_preds_txt = []
        model_main.eval()
        if img_model is not None:
            img_model.eval()
        if txt_model is not None:
            txt_model.eval()

        with torch.no_grad():
            for tokenized, images, numeric, ys in tqdm(val_loader, desc='FinalPred'):
                for k in tokenized:
                    tokenized[k] = tokenized[k].to(device)
                images = images.to(device)
                numeric = numeric.to(device)
                out_mu, out_logvar = model_main(tokenized, images, numeric)
                all_preds.append(np.expm1(out_mu.detach().cpu().numpy()))
                all_logvars.append(out_logvar.detach().cpu().numpy())

                if img_model is not None:
                    out_img = img_model(images)
                    all_preds_img.append(np.expm1(out_img.detach().cpu().numpy()))
                if txt_model is not None:
                    out_txt = txt_model(tokenized, numeric)
                    all_preds_txt.append(np.expm1(out_txt.detach().cpu().numpy()))

        preds_cat = np.concatenate(all_preds, axis=0)
        logvars_cat = np.concatenate(all_logvars, axis=0)
        oof_main[val_idx] = preds_cat
        oof_main_logvar[val_idx] = logvars_cat

        fold_out = out_dir / f'fold{fold_idx}_val_results_main.csv'
        tmp = val_df.copy()
        tmp['pred_price'] = preds_cat
        tmp['pred_logvar'] = logvars_cat
        tmp.to_csv(fold_out, index=False)
        print('Saved fold main results to', fold_out)

        if train_aux:
            preds_img_cat = np.concatenate(all_preds_img, axis=0)
            preds_txt_cat = np.concatenate(all_preds_txt, axis=0)
            oof_img[val_idx] = preds_img_cat
            oof_text[val_idx] = preds_txt_cat

            fold_out_img = out_dir / f'fold{fold_idx}_val_results_img.csv'
            tmp_img = val_df.copy()
            tmp_img['pred_price_img'] = preds_img_cat
            tmp_img.to_csv(fold_out_img, index=False)
            print('Saved fold img results to', fold_out_img)

            fold_out_txt = out_dir / f'fold{fold_idx}_val_results_text.csv'
            tmp_txt = val_df.copy()
            tmp_txt['pred_price_text'] = preds_txt_cat
            tmp_txt.to_csv(fold_out_txt, index=False)
            print('Saved fold text results to', fold_out_txt)

    # write OOF and compute full SMAPE
    full_trues = df['price'].values
    oof_preds_clipped = np.maximum(0.01, oof_main)
    full_smape = smape_np(full_trues, oof_preds_clipped)
    print('OOF SMAPE (main): {:.4f}%'.format(full_smape))

    out_oof = out_dir / 'oof_predictions_main_all_folds.csv'
    out_df = df.copy()
    out_df['pred_price'] = oof_main
    out_df['pred_logvar'] = oof_main_logvar
    out_df.to_csv(out_oof, index=False)
    print('Saved main OOF predictions to', out_oof)

    if train_aux:
        out_oof_img = out_dir / 'oof_predictions_img_all_folds.csv'
        out_df_img = df.copy()
        out_df_img['pred_price_img'] = oof_img
        out_df_img.to_csv(out_oof_img, index=False)
        print('Saved img OOF predictions to', out_oof_img)

        out_oof_txt = out_dir / 'oof_predictions_text_all_folds.csv'
        out_df_txt = df.copy()
        out_df_txt['pred_price_text'] = oof_text
        out_df_txt.to_csv(out_oof_txt, index=False)
        print('Saved text OOF predictions to', out_oof_txt)

    # Optional: simple stacker (Ridge) trained on OOFs
    if cfg.get('train_stacker', False) and train_aux:
        print('Training simple Ridge stacker on OOFs...')
        # build features: main_pred, 1/var_main, img_pred, text_pred
        var_main = np.exp(oof_main_logvar)
        inv_var_main = 1.0 / (var_main + 1e-8)
        X = np.vstack([oof_main, inv_var_main, oof_img, oof_text]).T
        y = full_trues
        ridge_alpha = float(cfg.get('stacker_alpha', 1.0))
        ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
        ridge.fit(X, y)
        stack_pred = ridge.predict(X)
        stack_smape = smape_np(y, stack_pred)
        print(f'Stacker OOF SMAPE: {stack_smape:.4f}%')
        joblib.dump(ridge, out_dir / 'stacker_ridge.joblib')
        # save stacker OOF results
        out_stack = out_dir / 'oof_predictions_stacked_all_folds.csv'
        out_df_stack = df.copy()
        out_df_stack['pred_price_stacked'] = stack_pred
        out_df_stack.to_csv(out_stack, index=False)
        print('Saved stacked OOF predictions to', out_stack)

if __name__ == '__main__':
    main()
