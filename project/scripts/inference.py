"""Inference and latent-inspection entry point.

Modes:

``reconstruct``
    image → latent text → image, saving originals, reconstructions, a side-by-side panel
    with the latent sentence printed underneath, and a JSON dump of the latents.
``encode``
    image → latent text only (fast; no reverse diffusion).
``decode``
    hand-written text → image, i.e. driving the decoder directly through latent space.
``prior_sample``
    z ~ p(z) from the frozen LM → image (unconditional generation).
``inspect``
    per-token analysis of a latent: top-k alternatives, entropy, prior log-probabilities,
    plus reconstructions across a temperature sweep.

Examples
--------
    python -m project.scripts.inference --checkpoint runs/textvae/default/checkpoints/best.pt \
        --mode reconstruct --input data/images --num-samples 8 --output out/recon

    python -m project.scripts.inference --checkpoint <ckpt> --mode decode \
        --text "a man with blond hair and a gray hat"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..datasets.builder import build_dataloader
from ..datasets.image_folder import IMAGE_EXTENSIONS
from ..datasets.transforms import build_image_transform
from ..models.vae import TextVAE
from ..training.checkpoint import load_checkpoint
from ..utils.config import Config, load_config
from ..utils.logging_utils import setup_logging
from ..utils.seed import set_seed
from ..utils.visualization import (
    denormalize,
    render_reconstruction_panel,
    save_image_grid,
    save_latent_report,
)

logger = logging.getLogger("textvae.inference")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct, sample and inspect latents."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="checkpoint to load"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="config (else taken from the checkpoint)",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "reconstruct",
            "encode",
            "decode",
            "prior_sample",
            "inspect",
            "evaluate",
        ],
        default="reconstruct",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=20,
        help="validation batches for --mode evaluate",
    )
    parser.add_argument(
        "--input", type=str, default=None, help="image file or directory"
    )
    parser.add_argument(
        "--text", type=str, nargs="*", default=None, help="text for --mode decode"
    )
    parser.add_argument("--output", type=str, default="outputs/inference")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--steps", type=int, default=None, help="reverse diffusion steps"
    )
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument(
        "--temperature", type=float, default=None, help="τ for the rollout"
    )
    parser.add_argument(
        "--stochastic", action="store_true", help="sample instead of argmax"
    )
    parser.add_argument("--top-k", type=int, default=None, help="prior sampling top-k")
    parser.add_argument(
        "--top-p", type=float, default=None, help="prior sampling nucleus"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE"
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------- loading
def load_model(args: argparse.Namespace) -> tuple[TextVAE, Config]:
    """Rebuild the model from a config (preferring the one stored in the checkpoint)."""
    config: Optional[Config] = None
    if args.config:
        config = load_config(args.config, overrides=args.overrides)
    if args.checkpoint:
        payload_meta = load_checkpoint(
            args.checkpoint, map_location="cpu", restore_rng_state=False
        )
        if config is None:
            config = payload_meta.config
            if config is None:
                raise ValueError(
                    "the checkpoint stores no config; pass --config explicitly"
                )
    if config is None:
        raise ValueError("provide --config and/or --checkpoint")

    model = TextVAE.from_config(config)
    if args.checkpoint:
        load_checkpoint(
            args.checkpoint,
            model=model,
            map_location="cpu",
            strict=False,
            restore_rng_state=False,
        )
    else:
        logger.warning("No checkpoint given: running with freshly initialised weights.")
    return model.to(args.device).eval(), config


def load_images(args: argparse.Namespace, config: Config) -> torch.Tensor:
    """Load images from a file/directory, or fall back to the configured dataset."""
    transform = build_image_transform(
        config.data.image_size, center_crop=True, random_flip=False
    )
    if args.input:
        from PIL import Image

        path = Path(args.input)
        if path.is_dir():
            files = sorted(
                p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            files = files[: args.num_samples]
        else:
            files = [path]
        if not files:
            raise FileNotFoundError(f"no images found at {path}")
        tensors = []
        for file in files:
            with Image.open(file) as image:
                tensors.append(transform(image))
        return torch.stack(tensors)

    loader = build_dataloader(config.validation_data(), train=False)
    batch = next(iter(loader))
    return batch["images"][: args.num_samples]


# ------------------------------------------------------------------------------ modes
def run_reconstruct(
    model: TextVAE, args: argparse.Namespace, config: Config, out: Path
) -> Dict[str, Any]:
    images = load_images(args, config).to(args.device)
    recon, texts, latent = model.reconstruct(
        images,
        temperature=args.temperature,
        stochastic=args.stochastic,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )
    save_image_grid(images, out / "original.png")
    save_image_grid(recon, out / "reconstruction.png")
    panel = render_reconstruction_panel(
        denormalize(images), denormalize(recon), texts, already_unit_range=True
    )
    panel.save(out / "panel.png")
    save_latent_report(
        out / "latents.txt",
        texts,
        token_strings=model.prior.token_strings(latent.token_ids),
        metrics=[
            {
                "log_q": float(latent.log_q_hard[i]),
                "entropy": float(latent.entropy[i].mean()),
            }
            for i in range(len(texts))
        ],
    )
    return {"latent_texts": texts, "token_ids": latent.token_ids.tolist()}


def run_encode(
    model: TextVAE, args: argparse.Namespace, config: Config, out: Path
) -> Dict[str, Any]:
    images = load_images(args, config).to(args.device)
    latent, texts = model.encode(
        images, temperature=args.temperature, stochastic=args.stochastic
    )
    save_image_grid(images, out / "original.png")
    save_latent_report(
        out / "latents.txt", texts, model.prior.token_strings(latent.token_ids)
    )
    for text in texts:
        logger.info("latent: %s", text)
    return {"latent_texts": texts, "token_ids": latent.token_ids.tolist()}


def run_decode(
    model: TextVAE, args: argparse.Namespace, config: Config, out: Path
) -> Dict[str, Any]:
    if not args.text:
        raise ValueError("--mode decode requires --text")
    size = config.data.image_size
    images = model.decode_from_text(
        args.text,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_size=(size, size),
    )
    save_image_grid(images, out / "decoded.png")
    panel = render_reconstruction_panel(
        denormalize(images),
        denormalize(images),
        list(args.text),
        already_unit_range=True,
    )
    panel.save(out / "panel.png")
    return {"texts": list(args.text)}


def run_prior_sample(
    model: TextVAE, args: argparse.Namespace, config: Config, out: Path
) -> Dict[str, Any]:
    size = config.data.image_size
    images, texts = model.sample_from_prior(
        args.num_samples,
        temperature=args.temperature or 1.0,
        top_k=args.top_k,
        top_p=args.top_p,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_size=(size, size),
    )
    save_image_grid(images, out / "prior_samples.png")
    panel = render_reconstruction_panel(
        denormalize(images), denormalize(images), texts, already_unit_range=True
    )
    panel.save(out / "panel.png")
    save_latent_report(out / "prior_latents.txt", texts)
    return {"latent_texts": texts}


def run_inspect(
    model: TextVAE, args: argparse.Namespace, config: Config, out: Path
) -> Dict[str, Any]:
    """Token-level latent inspection plus a temperature sweep (§7 interpretability)."""
    images = load_images(args, config).to(args.device)
    latent, texts = model.encode(
        images, temperature=args.temperature, stochastic=args.stochastic
    )
    prior_out = model.prior.log_prob_from_ids(latent.token_ids)
    log_probs = torch.log_softmax(latent.logits, dim=-1)
    top = log_probs.topk(5, dim=-1)

    # Decode all the token strings up front: one call per sample rather than per token.
    chosen_tokens = model.prior.token_strings(latent.token_ids)
    report: List[Dict[str, Any]] = []
    for i, text in enumerate(texts):
        alternatives = model.prior.token_strings(
            top.indices[i]
        )  # (T, k) -> T rows of k
        tokens = []
        for t in range(latent.token_ids.shape[1]):
            tokens.append(
                {
                    "position": t,
                    "token": chosen_tokens[i][t],
                    "log_q": float(latent.token_log_q_hard[i, t]),
                    "log_p": float(prior_out.per_token[i, t]),
                    "entropy": float(latent.entropy[i, t]),
                    "top_k": [
                        {
                            "token": alternatives[t][k],
                            "log_q": float(top.values[i, t, k]),
                        }
                        for k in range(top.indices.shape[-1])
                    ],
                }
            )
        report.append(
            {
                "text": text,
                "log_q": float(latent.log_q_hard[i]),
                "log_p": float(prior_out.total[i]),
                "kl": float(latent.log_q_hard[i] - prior_out.total[i]),
                "tokens": tokens,
            }
        )

    # Temperature sweep: how the latent text (and hence the reconstruction) sharpens.
    sweep: Dict[str, List[str]] = {}
    for tau in (0.05, 0.5, 1.0):
        _, sweep_texts = model.encode(images, temperature=tau, stochastic=True)
        sweep[f"tau={tau}"] = sweep_texts
        logger.info("τ=%.2f -> %s", tau, sweep_texts[0])

    save_latent_report(
        out / "latents.txt", texts, model.prior.token_strings(latent.token_ids)
    )
    return {"samples": report, "temperature_sweep": sweep}


def run_evaluate(
    model: TextVAE, args: argparse.Namespace, config: Config, out: Path
) -> Dict[str, Any]:
    """Quantitative evaluation: held-out ELBO terms plus a latent-usage probe.

    The ELBO terms alone cannot tell you whether the text latent is *doing* anything: a
    decoder that ignores ``Z'`` and denoises unconditionally still drives ``L_diff`` down,
    while the KL happily shrinks towards zero. So we also run a paired test — the same
    images, timesteps and noise, scored once with each image's own latent and once with
    the batch's latents rolled by one position:

        latent_usage = mean(L_diff[mismatched latent]) - mean(L_diff[matched latent])

    A value near zero means the decoder is ignoring the latent (posterior collapse); a
    clearly positive value means the latent carries image information the decoder uses.
    """
    loader = build_dataloader(config.validation_data(), train=False)
    tau = (
        args.temperature
        if args.temperature is not None
        else config.model.gumbel.eval_temperature
    )

    totals: Dict[str, float] = {}
    matched_all: List[float] = []
    mismatched_all: List[float] = []
    batches = 0

    for index, batch in enumerate(loader):
        if index >= args.num_batches:
            break
        images = batch["images"].to(args.device)
        if images.shape[0] < 2:
            continue  # the probe needs at least two samples to mismatch

        output = model(images, temperature=tau, stochastic=args.stochastic)
        metrics = {
            "loss": float(output.loss),
            "diffusion_loss": float(output.diffusion_loss),
            **output.metrics,
        }

        # Paired probe: identical t and ε for both branches, so the only difference is
        # which latent conditions the U-Net.
        text_embeddings = model.prior.embed_soft(output.latent.y)
        timesteps = torch.randint(
            0,
            model.decoder.num_train_timesteps,
            (images.shape[0],),
            device=images.device,
        )
        noise = torch.randn_like(model.decoder.encode_images(images))
        matched = model.decoder.diffusion_loss(
            images, text_embeddings, timesteps=timesteps, noise=noise
        )
        mismatched = model.decoder.diffusion_loss(
            images, text_embeddings.roll(1, dims=0), timesteps=timesteps, noise=noise
        )
        matched_all.append(float(matched.loss))
        mismatched_all.append(float(mismatched.loss))

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1

    if not batches:
        raise RuntimeError(
            "evaluation produced no batches; check the validation data config"
        )

    results: Dict[str, Any] = {key: value / batches for key, value in totals.items()}
    matched_mean = sum(matched_all) / batches
    mismatched_mean = sum(mismatched_all) / batches
    results.update(
        {
            "batches": batches,
            "latent_length": model.latent_length,
            "kl_bits": results["kl"] / math.log(2),
            "diffusion_loss_matched_latent": matched_mean,
            "diffusion_loss_mismatched_latent": mismatched_mean,
            "latent_usage": mismatched_mean - matched_mean,
            "latent_usage_relative": (mismatched_mean - matched_mean)
            / max(matched_mean, 1e-12),
        }
    )

    logger.info("Evaluated %d batch(es) at tau=%.3g", batches, tau)
    for key in (
        "loss",
        "diffusion_loss",
        "kl",
        "kl_per_token",
        "kl_bits",
        "posterior_entropy",
        "latent_unique_frac",
        "prior_perplexity",
    ):
        if key in results:
            logger.info("  %-26s %10.4f", key, results[key])
    logger.info(
        "  %-26s %10.4f  (matched %.4f vs mismatched %.4f, %+.1f%%)",
        "latent_usage",
        results["latent_usage"],
        matched_mean,
        mismatched_mean,
        100 * results["latent_usage_relative"],
    )
    if results["latent_usage_relative"] < 0.01:
        # No %-args here, so the format string is emitted verbatim: use a single '%'.
        logger.warning(
            "Latent usage is below 1%: the decoder is essentially ignoring the text latent "
            "(posterior collapse). Lower loss.beta or use loss.kl_reduction=mean_per_token."
        )
    return results


MODES = {
    "reconstruct": run_reconstruct,
    "encode": run_encode,
    "decode": run_decode,
    "prior_sample": run_prior_sample,
    "inspect": run_inspect,
    "evaluate": run_evaluate,
}


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    setup_logging(args.log_level)
    set_seed(args.seed)

    model, config = load_model(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        result = MODES[args.mode](model, args, config, out)

    result_path = out / f"{args.mode}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s outputs to %s", args.mode, out)
    return result


if __name__ == "__main__":
    main()
