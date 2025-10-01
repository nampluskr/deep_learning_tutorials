import logging
import os
from abc import ABC
from collections.abc import Sequence
from typing import Dict, Any, cast
from tqdm import tqdm
import numpy as np
import math
from time import time

import einops
from FrEIA.framework import SequenceINN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from feature_extractor import TimmFeatureExtractor
from all_in_one_block import AllInOneBlock
from tiler import Tiler
from trainer import BaseTrainer


#############################################################
# anomalib\models\images\cflow\utils.py
#############################################################

logger = logging.getLogger(__name__)

def get_logp(dim_feature_vector: int, p_u: torch.Tensor, logdet_j: torch.Tensor) -> torch.Tensor:
    ln_sqrt_2pi = -np.log(np.sqrt(2 * np.pi))  # ln(sqrt(2*pi))
    return dim_feature_vector * ln_sqrt_2pi - 0.5 * torch.sum(p_u**2, 1) + logdet_j


def positional_encoding_2d(condition_vector: int, height: int, width: int) -> torch.Tensor:
    if condition_vector % 4 != 0:
        msg = f"Cannot use sin/cos positional encoding with odd dimension (got dim={condition_vector})"
        raise ValueError(msg)
    pos_encoding = torch.zeros(condition_vector, height, width)
    # Each dimension use half of condition_vector
    condition_vector = condition_vector // 2
    div_term = torch.exp(torch.arange(0.0, condition_vector, 2) * -(math.log(1e4) / condition_vector))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pos_encoding[0:condition_vector:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pos_encoding[1:condition_vector:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pos_encoding[condition_vector::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pos_encoding[condition_vector + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    return pos_encoding


def subnet_fc(dims_in: int, dims_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dims_in, 2 * dims_in), nn.ReLU(), nn.Linear(2 * dims_in, dims_out))


def cflow_head(
    condition_vector: int,
    coupling_blocks: int,
    clamp_alpha: float,
    n_features: int,
    permute_soft: bool = False,
) -> SequenceINN:

    coder = SequenceINN(n_features)
    logger.info("CNF coder: %d", n_features)
    for _ in range(coupling_blocks):
        coder.append(
            AllInOneBlock,
            cond=0,
            cond_shape=(condition_vector,),
            subnet_constructor=subnet_fc,
            affine_clamping=clamp_alpha,
            global_affine_type="SOFTPLUS",
            permute_soft=permute_soft,
        )
    return coder


#############################################################
# anomalib\models\images\cflow\anomaly_map.py
#############################################################

class AnomalyMapGenerator(nn.Module):
    def __init__(
        self,
        pool_layers: Sequence[str],
    ) -> None:
        super().__init__()
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)
        self.pool_layers: Sequence[str] = pool_layers

    def compute_anomaly_map(
        self,
        distribution: list[torch.Tensor],
        height: list[int],
        width: list[int],
        image_size: tuple[int, int] | torch.Size | None,
    ) -> torch.Tensor:

        layer_maps: list[torch.Tensor] = []
        for layer_idx in range(len(self.pool_layers)):
            layer_distribution = distribution[layer_idx].clone().detach()
            # Normalize the likelihoods to (-Inf:0] and convert to probs in range [0:1]
            layer_probabilities = torch.exp(layer_distribution - layer_distribution.max())
            layer_map = layer_probabilities.reshape(-1, height[layer_idx], width[layer_idx])
            # upsample
            if image_size is not None:
                layer_map = F.interpolate(
                    layer_map.unsqueeze(1),
                    size=image_size,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(1)
            layer_maps.append(layer_map)
        # score aggregation
        score_map = torch.zeros_like(layer_maps[0])
        for layer_idx in range(len(self.pool_layers)):
            score_map += layer_maps[layer_idx]

        # Invert probs to anomaly scores
        return score_map.max() - score_map

    def forward(self, **kwargs: list[torch.Tensor] | list[int] | list[list]) -> torch.Tensor:
        if not ("distribution" in kwargs and "height" in kwargs and "width" in kwargs):
            msg = f"Expected keys `distribution`, `height` and `width`. Found {kwargs.keys()}"
            raise KeyError(msg)

        # placate mypy
        distribution: list[torch.Tensor] = cast("list[torch.Tensor]", kwargs["distribution"])
        height: list[int] = cast("list[int]", kwargs["height"])
        width: list[int] = cast("list[int]", kwargs["width"])
        image_size: tuple[int, int] | torch.Size | None = kwargs.get("image_size")
        return self.compute_anomaly_map(distribution, height, width, image_size)


#############################################################
# anomalib\models\images\cflow\torch_model.py
#############################################################

class CFlow(nn.Module):
    def __init__(self, backbone="resnet50", layers=["layer1", "layer2", "layer3"], 
        pre_trained=True, fiber_batch_size=64, decoder="freia-cflow",
        condition_vector=128, coupling_blocks=8, clamp_alpha=1.9, permute_soft=False):

        super().__init__()
        self.backbone = backbone
        self.fiber_batch_size = fiber_batch_size
        self.condition_vector: int = condition_vector
        self.dec_arch = decoder
        self.pool_layers = layers

        self.encoder = TimmFeatureExtractor(backbone=self.backbone, layers=self.pool_layers,
            pre_trained=pre_trained).eval()
        self.pool_dims = self.encoder.out_dims
        self.decoders = nn.ModuleList([
                cflow_head(
                    condition_vector=self.condition_vector,
                    coupling_blocks=coupling_blocks,
                    clamp_alpha=clamp_alpha,
                    n_features=pool_dim,
                    permute_soft=permute_soft,
                ) for pool_dim in self.pool_dims],)

        # encoder model is fixed
        for parameters in self.encoder.parameters():
            parameters.requires_grad = False

        self.anomaly_map_generator = AnomalyMapGenerator(pool_layers=self.pool_layers)

    def forward(self, images: torch.Tensor):
        self.encoder.eval()
        self.decoders.eval()
        with torch.no_grad():
            activation = self.encoder(images)

        distribution = [torch.empty(0, device=images.device) for _ in self.pool_layers]
        height: list[int] = []
        width: list[int] = []
        for layer_idx, layer in enumerate(self.pool_layers):
            encoder_activations = activation[layer]  # BxCxHxW
            batch_size, dim_feature_vector, im_height, im_width = encoder_activations.size()
            image_size = im_height * im_width
            embedding_length = batch_size * image_size  # number of rows in the conditional vector

            height.append(im_height)
            width.append(im_width)

            # repeats positional encoding for the entire batch 1 C H W to B C H W
            pos_encoding = einops.repeat(
                positional_encoding_2d(self.condition_vector, im_height, im_width).unsqueeze(0),
                "b c h w-> (tile b) c h w",
                tile=batch_size,
            ).to(images.device)
            c_r = einops.rearrange(pos_encoding, "b c h w -> (b h w) c")  # BHWxP
            e_r = einops.rearrange(encoder_activations, "b c h w -> (b h w) c")  # BHWxC
            decoder = self.decoders[layer_idx].to(images.device)

            # Sometimes during validation, the last batch E / N is not a whole number. Hence we need to add 1.
            # It is assumed that during training that E / N is a whole number as no errors were discovered during
            # testing. In case it is observed in the future, we can use only this line and ensure that FIB is at
            # least 1 or set `drop_last` in the dataloader to drop the last non-full batch.
            fiber_batches = embedding_length // self.fiber_batch_size + int(
                embedding_length % self.fiber_batch_size > 0,)

            for batch_num in range(fiber_batches):  # per-fiber processing
                if batch_num < (fiber_batches - 1):
                    idx = torch.arange(batch_num * self.fiber_batch_size, (batch_num + 1) * self.fiber_batch_size)
                else:  # When non-full batch is encountered batch_num+1 * N will go out of bounds
                    idx = torch.arange(batch_num * self.fiber_batch_size, embedding_length)
                c_p = c_r[idx]  # NxP
                e_p = e_r[idx]  # NxC

                # decoder returns the transformed variable z and the log Jacobian determinant
                with torch.no_grad():
                    p_u, log_jac_det = decoder(e_p, [c_p])
                decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
                log_prob = decoder_log_prob / dim_feature_vector  # likelihood per dim
                distribution[layer_idx] = torch.cat((distribution[layer_idx], log_prob))

        anomaly_map = self.anomaly_map_generator(
            distribution=distribution,
            height=height,
            width=width,
            image_size=images.shape[-2:],
        )
        self.decoders.train()
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)

    def predict(self, images: torch.Tensor):
        return self.forward(images)


#############################################################
# Trainer for CFlow Model
#############################################################

class CFlowTrainer(BaseTrainer):
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Only decoder parameters are trainable
        self.optimizer = optimizer or torch.optim.Adam(
            params=self._get_decoder_params(),
            lr=1e-4,
            weight_decay=0.0
        )
        self.loss_fn = None
        self.metrics = metrics or {}

    def _get_decoder_params(self):
        params = []
        for decoder in self.model.decoders:
            params += list(decoder.parameters())
        return params

    def run_epoch(self, loader, mode='train', desc=""):
        if mode == 'train':
            self.model.encoder.eval()
            self.model.decoders.train()
        else:
            self.model.encoder.eval()
            self.model.decoders.eval()

        stats = {"loss": 0.0}
        n_samples = 0
        pbar = tqdm(loader, desc=desc, leave=False, ascii=True)
        
        for batch in pbar:
            images = batch["image"].to(self.device)
            batch_size = images.size(0)
            n_samples += batch_size

            if mode == 'train':
                loss = self._training_step(images)
            else:
                with torch.no_grad():
                    loss = self._validation_step(images)

            stats["loss"] += loss
            pbar.set_postfix({"loss": f"{stats['loss']/n_samples:.4f}"})

        return {k: v / max(n_samples, 1) for k, v in stats.items()}

    def _training_step(self, images):
        # Extract features from encoder (frozen)
        with torch.no_grad():
            activation = self.model.encoder(images)

        avg_loss = 0.0
        
        # Process each layer
        for layer_idx, layer_name in enumerate(self.model.pool_layers):
            encoder_activations = activation[layer_name].detach()  # [B, C, H, W]
            B, C, H, W = encoder_activations.shape
            image_size = H * W
            embedding_length = B * image_size  # Total number of position embeddings

            # Generate positional encoding: [B, P, H, W]
            pos_encoding = positional_encoding_2d(self.model.condition_vector, H, W).unsqueeze(0)
            pos_encoding = einops.repeat(pos_encoding, 'b c h w -> (tile b) c h w', tile=B)
            pos_encoding = pos_encoding.to(images.device)
            
            # Rearrange to [BHW, P] and [BHW, C]
            c_r = einops.rearrange(pos_encoding, 'b c h w -> (b h w) c')
            e_r = einops.rearrange(encoder_activations, 'b c h w -> (b h w) c')
            
            # Random permutation for training
            perm = torch.randperm(embedding_length, device=images.device)
            
            decoder = self.model.decoders[layer_idx]

            # Calculate number of fiber batches
            fiber_batches = embedding_length // self.model.fiber_batch_size
            if fiber_batches <= 0:
                msg = f"fiber_batch_size ({self.model.fiber_batch_size}) is too large for " \
                      f"embedding_length ({embedding_length}). Decrease fiber_batch_size or increase image size."
                raise ValueError(msg)

            # Process each fiber batch
            for batch_num in range(fiber_batches):
                self.optimizer.zero_grad()
                
                # Get indices for this fiber batch
                if batch_num < (fiber_batches - 1):
                    start_idx = batch_num * self.model.fiber_batch_size
                    end_idx = (batch_num + 1) * self.model.fiber_batch_size
                    idx = torch.arange(start_idx, end_idx, device=images.device)
                else:
                    # Last batch: include remaining samples
                    start_idx = batch_num * self.model.fiber_batch_size
                    idx = torch.arange(start_idx, embedding_length, device=images.device)

                # Get random subset using permutation
                c_p = c_r[perm[idx]]  # [N, P]
                e_p = e_r[perm[idx]]  # [N, C]

                # Forward through decoder
                p_u, log_jac_det = decoder(e_p, [c_p])
                
                # Calculate log probability
                decoder_log_prob = get_logp(C, p_u, log_jac_det)
                log_prob = decoder_log_prob / C  # likelihood per dimension
                
                # Loss: negative log-sigmoid (following Lightning implementation)
                loss = -F.logsigmoid(log_prob)
                
                # Backward and optimize for this fiber batch
                loss.mean().backward()
                self.optimizer.step()

                # Accumulate total loss (using sum, not mean)
                avg_loss += loss.sum().item()
        return avg_loss

    def _validation_step(self, images):
        _ = self.model.predict(images)
        return 0.0

    def fit(self, train_loader, num_epochs, valid_loader=None, weight_path=None):
        print("\n > Start CFlow training...")
        history = {"loss": []}
        if valid_loader is not None:
            history["val_loss"] = []

        train_start = time()
        best_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            epoch_start = time()

            # Training
            train_metrics = self.run_epoch(train_loader, mode='train', desc=f"Train [{epoch}/{num_epochs}]")
            history["loss"].append(train_metrics["loss"])

            # Validation
            val_metrics = {"loss": 0.0}
            if valid_loader is not None:
                val_metrics = self.run_epoch(valid_loader, mode='valid', desc=f"Valid [{epoch}/{num_epochs}]")
                history["val_loss"].append(val_metrics["loss"])
                current_loss = val_metrics["loss"]
                if current_loss < best_loss:
                    best_loss = current_loss
                    if weight_path:
                        self.save_model(weight_path.replace(".pth", "_best.pth"))
            else:
                current_loss = train_metrics["loss"]

            train_info = f"loss={train_metrics['loss']:.4f}"
            print(f" [{epoch:3d}/{num_epochs}] {train_info} ({time() - epoch_start:.1f}s)")

            # Evaluate metrics every 5 epochs or at the last epoch
            if valid_loader is not None:
                if epoch % 5 == 0 or epoch == num_epochs:
                    print()
                    for method in ["f1", "roc"]:
                        eval_img = self.evaluate_image_level(valid_loader, method=method)
                        img_info1 = ", ".join([f"{k}={v:.3f}" for k, v in eval_img.items() if isinstance(v, float)])
                        img_info2 = ", ".join([f"{k}={v:2d}" for k, v in eval_img.items() if isinstance(v, int)])
                        print(f" > Image-level: {img_info1} | {img_info2} ({method})")
                    print()

        total_time = time() - train_start
        h, r = divmod(int(total_time), 3600)
        m, s = divmod(r, 60)
        print(f" > Training finished in {h:02d}:{m:02d}:{s:02d}")
        if weight_path:
            self.save_model(weight_path)
        return history

    def save_model(self, weight_path):
        """Save model state dict."""
        if weight_path is not None:
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            
            # Save entire model state (all decoders)
            decoder_states = [decoder.state_dict() for decoder in self.model.decoders]
            
            ckpt = {
                "decoder_states": decoder_states,
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            torch.save(ckpt, weight_path)
            print(f" > CFlow model saved to: {weight_path}\n")

    def load_model(self, weight_path):
        """Load model state dict."""
        if not os.path.isfile(weight_path):
            print(f" > No checkpoint found at: {weight_path}\n")
            return
            
        ckpt = torch.load(weight_path, map_location=self.device)
        
        # Load decoder states
        if "decoder_states" in ckpt:
            decoder_states = ckpt["decoder_states"]
            for decoder, state in zip(self.model.decoders, decoder_states):
                decoder.load_state_dict(state)
        
        # Load optimizer state
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
        print(f" > CFlow model loaded from: {weight_path}\n")


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cuda")
    model = CFlow(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        fiber_batch_size=64,
        decoder="freia-cflow",
        condition_vector=128,
        coupling_blocks=8,
        clamp_alpha=1.9,
        permute_soft=False
    ).to(device)

    x = torch.randn(32, 3, 256, 256).to(device)
    predictions = model.predict(x)
    print(predictions["pred_score"].shape)      # torch.Size([32])
    print(predictions["anomaly_map"].shape)     # torch.Size([32, 256, 256])
