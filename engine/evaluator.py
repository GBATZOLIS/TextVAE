# engine/evaluator.py
import torch
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, data_loader, device, config, logger, wandb_run=None):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.config = config
        self.logger = logger
        self.wandb_run = wandb_run

    def evaluate(self):
        self.model.eval()
        total_loss, total_recon_loss, total_vq_loss, total_perplexity = 0, 0, 0, 0
        pbar = tqdm(self.data_loader, desc="Final Evaluation", leave=False)
        with torch.no_grad():
            for images, _ in pbar:
                images = images.to(self.device)

                model_output = self.model(images)

                total_loss += model_output["loss"].item()
                total_recon_loss += model_output["recon_loss"].item()
                total_vq_loss += model_output["vq_loss"].item()
                total_perplexity += model_output["perplexity"].item()

        avg_loss = total_loss / len(self.data_loader)
        avg_recon_loss = total_recon_loss / len(self.data_loader)
        avg_vq_loss = total_vq_loss / len(self.data_loader)
        avg_perplexity = total_perplexity / len(self.data_loader)

        self.logger.info(
            f"Final Validation -> Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, VQ Loss: {avg_vq_loss:.4f}, Perplexity: {avg_perplexity:.2f}"
        )

        if self.wandb_run:
            self.wandb_run.summary["final_val_loss"] = avg_loss
            self.wandb_run.summary["final_val_recon_loss"] = avg_recon_loss
            self.wandb_run.summary["final_val_vq_loss"] = avg_vq_loss
            self.wandb_run.summary["final_val_perplexity"] = avg_perplexity
            self.logger.info("Logged final metrics to W&B summary.")

        return avg_loss, avg_recon_loss, avg_vq_loss, avg_perplexity
