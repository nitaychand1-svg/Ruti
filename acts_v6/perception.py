"""
Perception layer components for ACTS v6.

The module combines multi-modal feature extraction with Bayesian regime
classification trained via ELBO.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _FallbackEncoder(nn.Module):
    """Lightweight encoder used when heavy pretrained models are unavailable."""

    def __init__(self, input_dim: int, output_dim: int, seed: int = 17) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.linear.weight, generator=generator)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
        return torch.tanh(self.linear(x))


@dataclass
class PerceptionOutput:
    """Structured return type for multi-modal perception."""

    holistic_features: np.ndarray
    uncertainty: Dict[str, float]
    modality_embeddings: Dict[str, np.ndarray]


class MultiModalFusionEngine(nn.Module):
    """
    Multi-modal perception module with graceful fallbacks.

    The engine fuses text, vision, audio, and numerical market signals into a
    unified latent representation. Whenever heavyweight pretrained encoders are
    not available (e.g. during local development or tests), lightweight
    deterministic projections are used instead, keeping the API consistent.
    """

    text_dim: int = 768
    vision_dim: int = 512
    audio_dim: int = 256
    market_dim: int = 256

    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._async_lock = asyncio.Lock()

        self._text_encoder, self._text_tokenizer = self._init_text_encoder()
        self._vision_encoder, self._vision_preprocess = self._init_vision_encoder()
        self._audio_encoder = self._init_audio_encoder()

        fusion_in = self.text_dim + self.vision_dim + self.audio_dim + self.market_dim
        self.market_adapter: Optional[nn.Sequential] = None

        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_in, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
        ).to(self.device)

        logger.info("Multi-Modal Fusion Engine ready on %s", self.device)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    async def perceive_world(
        self,
        market_data: np.ndarray,
        news_articles: Sequence[str],
        social_posts: Sequence[str],
        geo_images: Optional[Sequence[Any]] = None,
        audio_speeches: Optional[Sequence[Any]] = None,
    ) -> PerceptionOutput:
        """
        Fuse multi-modal signals into a unified feature vector.

        The method is coroutine-safe: heavy encoders are accessed under a lock to
        avoid race conditions in inference servers.
        """

        market_proj = self._encode_market(market_data)
        text_emb = await self._encode_text_async(news_articles, social_posts)
        vision_emb = await self._encode_vision_async(geo_images) if geo_images else self._zero(self.vision_dim)
        audio_emb = await self._encode_audio_async(audio_speeches) if audio_speeches else self._zero(self.audio_dim)

        fused_input = torch.cat([market_proj, text_emb, vision_emb, audio_emb], dim=-1)
        holistic = self.fusion_network(fused_input)

        # Bayesian dropout-inspired uncertainty proxy
        epistemic = float(torch.mean(torch.abs(holistic)) / 100.0)
        aleatoric = 0.03  # placeholder - would be estimated via dedicated head
        total = float(np.sqrt(epistemic ** 2 + aleatoric ** 2))

        return PerceptionOutput(
            holistic_features=holistic.detach().cpu().numpy(),
            uncertainty={
                "epistemic": epistemic,
                "aleatoric": aleatoric,
                "total": total,
                "confidence": float(max(0.0, 1.0 - epistemic)),
            },
            modality_embeddings={
                "market": market_proj.detach().cpu().numpy(),
                "text": text_emb.detach().cpu().numpy(),
                "vision": vision_emb.detach().cpu().numpy(),
                "audio": audio_emb.detach().cpu().numpy(),
            },
        )

    # --------------------------------------------------------------------- #
    # Encoder helpers
    # --------------------------------------------------------------------- #
    def _encode_market(self, market_data: np.ndarray) -> torch.Tensor:
        market_tensor = torch.as_tensor(market_data, dtype=torch.float32, device=self.device)
        if market_tensor.ndim == 2:
            market_tensor = market_tensor.mean(dim=0)
        if market_tensor.ndim != 1:
            market_tensor = market_tensor.flatten()

        feature_dim = market_tensor.shape[0]
        if self.market_adapter is None:
            self.market_adapter = nn.Sequential(
                nn.Linear(feature_dim, self.market_dim, bias=True),
                nn.ReLU(inplace=True),
            ).to(self.device)
            logger.debug("Initialized market adapter with input dim %s", feature_dim)

        return self.market_adapter(market_tensor)

    async def _encode_text_async(
        self,
        news_articles: Sequence[str],
        social_posts: Sequence[str],
    ) -> torch.Tensor:
        texts = list(news_articles) + list(social_posts)
        if not texts:
            return self._zero(self.text_dim)

        async with self._async_lock:
            if self._text_encoder is None:
                embeddings = [self._simple_text_embedding(t) for t in texts]
                tensor = torch.stack(embeddings, dim=0).mean(dim=0)
            else:
                inputs = self._text_tokenizer(
                    texts[:32],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self._text_encoder(**inputs)
                tensor = outputs.last_hidden_state[:, 0, :].mean(dim=0)

        return tensor.to(self.device)

    async def _encode_vision_async(self, images: Sequence[Any]) -> torch.Tensor:
        if not images:
            return self._zero(self.vision_dim)

        async with self._async_lock:
            if self._vision_encoder is None:
                stacked = torch.stack([self._simple_image_embedding(img) for img in images], dim=0)
                return stacked.mean(dim=0).to(self.device)

            processed = torch.stack([self._vision_preprocess(img) for img in images])
            processed = processed.to(self.device)
            with torch.no_grad():
                features = self._vision_encoder.encode_image(processed)  # type: ignore[attr-defined]
            return features.mean(dim=0)

    async def _encode_audio_async(self, speeches: Sequence[Any]) -> torch.Tensor:
        if not speeches:
            return self._zero(self.audio_dim)

        async with self._async_lock:
            if self._audio_encoder is None:
                stacked = torch.stack([self._simple_audio_embedding(audio) for audio in speeches], dim=0)
                return stacked.mean(dim=0).to(self.device)

            raise NotImplementedError("Whisper audio encoder integration pending")

    # --------------------------------------------------------------------- #
    # Initialization helpers
    # --------------------------------------------------------------------- #
    def _init_text_encoder(self):
        try:
            from transformers import RobertaModel, RobertaTokenizer  # type: ignore

            model = RobertaModel.from_pretrained("roberta-base")
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            model.to(self.device)
            model.eval()
            return model, tokenizer
        except Exception as exc:  # pragma: no cover - depends on external deps
            logger.warning("Falling back to lightweight text encoder: %s", exc)
            return None, None

    def _init_vision_encoder(self):
        try:  # pragma: no cover - external dependency
            import clip  # type: ignore

            model, preprocess = clip.load("ViT-B/32", device=self.device)
            return model, preprocess
        except Exception as exc:
            logger.warning("Falling back to lightweight vision encoder: %s", exc)
            return None, None

    def _init_audio_encoder(self):
        try:  # pragma: no cover - heavy dependency
            from transformers import WhisperModel  # type: ignore

            model = WhisperModel.from_pretrained("openai/whisper-small")
            model.to(self.device)
            model.eval()
            return model
        except Exception as exc:
            logger.warning("Falling back to lightweight audio encoder: %s", exc)
            return None

    # --------------------------------------------------------------------- #
    # Simple deterministic fallbacks
    # --------------------------------------------------------------------- #
    def _simple_text_embedding(self, text: str) -> torch.Tensor:
        tokens = text.encode("utf-8", errors="ignore")
        arr = torch.tensor(list(tokens[: self.text_dim]), dtype=torch.float32, device=self.device)
        if arr.numel() < self.text_dim:
            arr = F.pad(arr, (0, self.text_dim - arr.numel()))
        return arr / (arr.abs().max() + 1e-6)

    def _simple_image_embedding(self, image: Any) -> torch.Tensor:
        data = torch.tensor(np.asarray(image).flatten()[: self.vision_dim], dtype=torch.float32)
        if data.numel() < self.vision_dim:
            data = F.pad(data, (0, self.vision_dim - data.numel()))
        return data.to(self.device) / (data.abs().max() + 1e-6)

    def _simple_audio_embedding(self, audio: Any) -> torch.Tensor:
        data = np.asarray(audio).flatten()
        tensor = torch.tensor(data[: self.audio_dim], dtype=torch.float32)
        if tensor.numel() < self.audio_dim:
            tensor = F.pad(tensor, (0, self.audio_dim - tensor.numel()))
        return tensor.to(self.device) / (tensor.abs().max() + 1e-6)

    def _zero(self, dim: int) -> torch.Tensor:
        return torch.zeros(dim, device=self.device)


# ========================================================================== #
# Bayesian regime prediction
# ========================================================================== #

class BayesianLinearLayer(nn.Module):
    """Bayesian linear layer with learnable posterior parameters."""

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 0.1) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))

        self.register_buffer("prior_mu", torch.tensor(0.0))
        self.register_buffer("prior_sigma_tensor", torch.tensor(prior_sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        weight_kl = 0.5 * torch.sum(
            torch.log(self.prior_sigma_tensor**2 / (weight_sigma**2 + 1e-8))
            + (weight_sigma**2 + (self.weight_mu - self.prior_mu) ** 2) / (self.prior_sigma_tensor**2 + 1e-8)
            - 1.0
        )
        bias_kl = 0.5 * torch.sum(
            torch.log(self.prior_sigma_tensor**2 / (bias_sigma**2 + 1e-8))
            + (bias_sigma**2 + (self.bias_mu - self.prior_mu) ** 2) / (self.prior_sigma_tensor**2 + 1e-8)
            - 1.0
        )
        return weight_kl + bias_kl


class BayesianRegimePredictor(nn.Module):
    """Stochastic regime classifier trained with an ELBO objective."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_regimes: int = 3) -> None:
        super().__init__()
        self.fc1 = BayesianLinearLayer(input_dim, hidden_dim)
        self.fc2 = BayesianLinearLayer(hidden_dim, hidden_dim)
        self.fc3 = BayesianLinearLayer(hidden_dim, n_regimes)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.temperature = nn.Parameter(torch.ones(1))
        self.elbo_optimizer: Optional[ELBOOptimizer] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.dropout(self.relu(self.fc1(x)))
        h2 = self.dropout(self.relu(self.fc2(h1)))
        logits = self.fc3(h2)
        return logits / (self.temperature + 1e-8)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        samples = torch.stack([torch.softmax(self.forward(x), dim=-1) for _ in range(max(1, n_samples))])
        mean_probs = samples.mean(dim=0)
        epistemic_unc = samples.var(dim=0).sum(dim=-1)
        pred_entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
        return mean_probs, epistemic_unc, pred_entropy


class ELBOOptimizer:
    """Variational optimizer with KL annealing and adaptive diagnostics."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        kl_weight_start: float = 0.0,
        kl_weight_end: float = 1.0,
        anneal_steps: int = 1_000,
    ) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.kl_weight_start = kl_weight_start
        self.kl_weight_end = kl_weight_end
        self.anneal_steps = anneal_steps
        self.current_step = 0
        self.elbo_history: List[float] = []
        self.convergence_achieved = False

    def get_kl_weight(self) -> float:
        if self.current_step >= self.anneal_steps:
            return self.kl_weight_end
        progress = self.current_step / max(1, self.anneal_steps)
        return self.kl_weight_start + progress * (self.kl_weight_end - self.kl_weight_start)

    def step(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        n_samples: int = 10,
    ) -> Dict[str, float]:
        self.model.train()

        logits_samples = torch.stack([self.model(features) for _ in range(max(1, n_samples))])

        if logits_samples.shape[-1] == 1 or targets.dtype.is_floating_point:
            # Regression branch
            preds = logits_samples.squeeze(-1)
            mse = F.mse_loss(preds, targets.float(), reduction="none")
            log_lik = -mse.sum(dim=-1)
        else:
            log_probs = -F.cross_entropy(
                logits_samples.view(-1, logits_samples.size(-1)),
                targets.repeat_interleave(max(1, n_samples)),
                reduction="none",
            )
            log_lik = log_probs.view(max(1, n_samples), -1).sum(dim=-1)

        expected_log_lik = torch.mean(log_lik)

        kl_div = sum(module.kl_divergence() for module in self.model.modules() if hasattr(module, "kl_divergence"))
        beta = self.get_kl_weight()
        elbo = expected_log_lik - beta * kl_div
        batch_size = targets.shape[0]
        loss = -(elbo / batch_size)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.current_step += 1
        self.elbo_history.append(float(elbo.detach() / batch_size))
        self._update_convergence_flag()

        return {
            "elbo": float(elbo.detach() / batch_size),
            "likelihood": float(expected_log_lik.detach() / batch_size),
            "kl": float(kl_div.detach()),
            "beta": beta,
            "loss": float(loss.detach()),
        }

    def _update_convergence_flag(self) -> None:
        if len(self.elbo_history) < 100:
            return
        window = self.elbo_history[-100:]
        if np.var(window) < 1e-3:
            self.convergence_achieved = True
