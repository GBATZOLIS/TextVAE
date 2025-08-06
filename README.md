# Text‑VAE with Diffusion Decoder

## 🧱 Project Structure

```

.
├── README.md
├── config.py
├── datasets.py
├── main.py
├── models
│   ├── __init__.py
│   ├── diffusion_decoder.py
│   ├── vae.py
│   └── vision_encoder.py
├── requirements.txt
└── trainer.py


````

---

## 🚀 Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/GBATZOLIS/TextVAE.git
cd TextVAE
pip install -r requirements.txt
````

### 2. Run training (example)

```bash
python main.py --epochs 10 --batch 8
```

---

## 🧪 Notes

* The text decoder uses GPT-2 embeddings and can optionally freeze the language model.
* Gumbel-Softmax annealing is supported for hard/soft token sampling.
* The image reconstruction is performed via a UNet-based DDPM.

---

## ✏️ TODO

* Replace `DummyImageDataset` with real data (e.g. ImageNet, CelebA).
* Add evaluation/metrics (e.g., FID for images, BLEU for text).

---
