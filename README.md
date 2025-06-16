# Text‑VAE with Diffusion Decoder

## 🧱 Project Structure

```

TextVAE/
├── config.py               # Configuration dataclass
├── datasets.py            # Dummy dataset loader
├── main.py                # Entry-point script
├── trainer.py             # Training loop
├── requirements.txt
├── README.md
└── models/
├── init.py
├── gumbel_softmax.py
├── vision_encoder.py
├── text_decoder.py
├── vae.py
└── diffusion/
├── init.py
├── diffusion_decoder.py
└── unet.py

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


