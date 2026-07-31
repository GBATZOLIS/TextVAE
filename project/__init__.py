"""Image -> text-latent -> image Variational Autoencoder.

Reference implementation of *A Variational Autoencoder with Interpretable Textual Latents
and a Diffusion-based Decoder*: a ViT encoder emits an autoregressive distribution over
text-token sequences, Gumbel-Softmax makes the discrete latent differentiable, a frozen
language model provides the prior p(z), and a text-conditional DDPM reconstructs the
image.
"""

__version__ = "0.1.0"
