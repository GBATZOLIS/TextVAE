"""Image de-normalisation, grids, and latent-text panels for qualitative inspection."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import torch
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "denormalize",
    "to_pil_images",
    "make_grid",
    "save_image_grid",
    "render_reconstruction_panel",
    "save_latent_report",
]


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Map diffusion-space images in [-1, 1] to [0, 1]."""
    return (images.detach().float() / 2 + 0.5).clamp(0, 1)


def to_pil_images(
    images: torch.Tensor, already_unit_range: bool = False
) -> List[Image.Image]:
    """Convert a (B, C, H, W) tensor to a list of PIL images."""
    batch = images if already_unit_range else denormalize(images)
    batch = batch.detach().float().clamp(0, 1).cpu()
    if batch.dim() == 3:
        batch = batch.unsqueeze(0)
    out = []
    for img in batch:
        array = (img.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
        if array.shape[-1] == 1:
            out.append(Image.fromarray(array[..., 0], mode="L").convert("RGB"))
        else:
            out.append(Image.fromarray(array, mode="RGB"))
    return out


def make_grid(
    images: Sequence[Image.Image], ncol: Optional[int] = None, pad: int = 4
) -> Image.Image:
    """Tile PIL images into a single grid image."""
    if not images:
        raise ValueError("make_grid received an empty image list")
    ncol = ncol or len(images)
    ncol = max(1, min(ncol, len(images)))
    nrow = (len(images) + ncol - 1) // ncol
    width = max(img.width for img in images)
    height = max(img.height for img in images)
    canvas = Image.new(
        "RGB",
        (ncol * width + (ncol + 1) * pad, nrow * height + (nrow + 1) * pad),
        color=(255, 255, 255),
    )
    for idx, img in enumerate(images):
        row, col = divmod(idx, ncol)
        canvas.paste(img, (pad + col * (width + pad), pad + row * (height + pad)))
    return canvas


def save_image_grid(
    images: torch.Tensor,
    path: Union[str, Path],
    ncol: Optional[int] = None,
    already_unit_range: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(
        to_pil_images(images, already_unit_range=already_unit_range), ncol=ncol
    )
    grid.save(path)
    return path


def render_reconstruction_panel(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    latent_texts: Sequence[str],
    already_unit_range: bool = False,
    min_column_width: int = 190,
    max_caption_chars: int = 400,
) -> Image.Image:
    """Original / reconstruction pairs with the latent sentence printed underneath.

    This is the core interpretability artefact of the model (§7): it shows the text the
    encoder chose next to the image the decoder reconstructed from that text alone.

    Columns are widened to ``min_column_width`` when the thumbnails are small, and the
    caption is wrapped to the column width, so latent sentences never overlap between
    adjacent samples.
    """
    orig = to_pil_images(originals, already_unit_range=already_unit_range)
    recon = to_pil_images(reconstructions, already_unit_range=already_unit_range)
    n = min(len(orig), len(recon))
    if n == 0:
        raise ValueError("render_reconstruction_panel received empty batches")

    cell_w = max(img.width for img in orig + recon)
    cell_h = max(img.height for img in orig + recon)
    column_w = max(cell_w, min_column_width)
    pad = 6

    font = ImageFont.load_default()
    char_w = max(1, _text_width(font, "n"))
    line_h = max(8, _text_height(font, "Ag") + 2)
    wrap_chars = max(10, (column_w - 2) // char_w)

    wrapped = [
        textwrap.fill(
            (latent_texts[i] if i < len(latent_texts) else "")
            .replace("\n", " ")
            .strip()[:max_caption_chars],
            width=wrap_chars,
        )
        for i in range(n)
    ]
    caption_lines = max(1, max(text.count("\n") + 1 for text in wrapped))
    caption_height = caption_lines * line_h + pad

    panel = Image.new(
        "RGB",
        (n * (column_w + pad) + pad, 2 * (cell_h + pad) + caption_height + pad),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(panel)
    for i in range(n):
        x = pad + i * (column_w + pad)
        # Centre the thumbnails inside their (possibly wider) column.
        offset = (column_w - cell_w) // 2
        panel.paste(orig[i], (x + offset, pad))
        panel.paste(recon[i], (x + offset, pad + cell_h + pad))
        draw.multiline_text(
            (x, 2 * (cell_h + pad) + 2),
            wrapped[i],
            fill=(0, 0, 0),
            spacing=2,
            font=font,
        )
    return panel


def _text_width(font: Any, text: str) -> int:
    box = font.getbbox(text)
    return int(box[2] - box[0])


def _text_height(font: Any, text: str) -> int:
    box = font.getbbox(text)
    return int(box[3] - box[1])


def save_latent_report(
    path: Union[str, Path],
    latent_texts: Sequence[str],
    token_strings: Optional[Sequence[Sequence[str]]] = None,
    metrics: Optional[Sequence[dict]] = None,
) -> Path:
    """Write a human-readable dump of the latent sequences (one block per sample)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for i, text in enumerate(latent_texts):
        lines.append(f"=== sample {i} ===")
        lines.append(f"text: {text}")
        if token_strings is not None and i < len(token_strings):
            lines.append("tokens: " + " | ".join(token_strings[i]))
        if metrics is not None and i < len(metrics):
            rendered = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics[i].items()
            )
            lines.append(f"metrics: {rendered}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
