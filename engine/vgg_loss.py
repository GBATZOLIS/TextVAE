# engine/vgg_loss.py

import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """
    VGG Perceptual Loss module.
    Calculates the L1 loss between feature maps of two images from a pre-trained VGG19 network.
    """

    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        # Load pre-trained VGG19 model and select feature extraction layers
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # We'll use the output of the conv layers before the 3rd and 4th max-pooling layers
        self.features = nn.Sequential(*list(vgg.children())[:35]).eval()

        # Freeze the parameters of the VGG network
        for param in self.features.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

        # Normalize images using ImageNet mean and std before passing to VGG
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, generated_img, target_img):
        """
        Calculates the perceptual loss.

        Args:
            generated_img (torch.Tensor): The reconstructed image from the VQ-VAE.
                                          Expected to be in [-1, 1] range.
            target_img (torch.Tensor): The original input image.
                                       Expected to be in [-1, 1] range.
        """
        # First, denormalize from [-1, 1] to [0, 1]
        generated_img = (generated_img + 1) / 2
        target_img = (target_img + 1) / 2

        # Then, normalize for VGG (ImageNet stats)
        generated_img = (generated_img - self.mean) / self.std
        target_img = (target_img - self.mean) / self.std

        # Extract features
        gen_features = self.features(generated_img)
        target_features = self.features(target_img)

        # Calculate L1 loss between feature maps
        return self.criterion(gen_features, target_features)
