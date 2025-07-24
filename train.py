import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb

from diffusers.training_utils import EMAModel
from diffusers import get_scheduler
from diffusers.schedulers import DDPMScheduler

from conditional_1d_unet import ConditionalUnet1D
from dataset import DloDataset


MAIN_DIR = os.path.join(os.path.dirname(__file__))

LOG_INTERVAL = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = dict(
    batch_size=128,
    epochs=100,
    lr=5e-4,
    hidden_dim=256,
    dim_points=2,
    num_points=51,
    scale_action=True,
    scale_disp=0.05,
    scale_rot=np.pi / 8,
    dataset_path="train3",
    obs_dim=206,
    obs_ee_dim=3,
    obs_dlo_dim=102,
    action_dim=4,
    obs_h_dim=2,
    pred_h_dim=16,
)

wandb.init(config=CONFIG, project="diffusion_model", entity="riccardo_mengozzi", mode="disabled")
config = wandb.config

###################################
print(f"Using device: {DEVICE}")
print("*" * 20)
for k, v in config.items():
    print(f"\t{k}: {v}")
print("*" * 20)
###################################


class DiffusionTrainer:
    def __init__(self, config: dict):
        self.config = config

        # train_path = os.path.join(DATASETS_PATH, config["dataset_path"], "train")
        # val_path = os.path.join(DATASETS_PATH, config["dataset_path"], "val")
        train_path = os.path.join(MAIN_DIR, config["dataset_path"])
        val_path = os.path.join(MAIN_DIR, config["dataset_path"])

        # Build dataset and dataloader
        train_data = DloDataset(
            train_path,
            num_points=config["num_points"],
            scale_action=config["scale_action"],
            linear_action_range=config["scale_disp"],
            rot_action_range=config["scale_rot"],
            obs_h_dim=config["obs_h_dim"],
            pred_h_dim=config["pred_h_dim"],
        )

        val_data = DloDataset(
            val_path,
            num_points=config["num_points"],
            scale_action=config["scale_action"],
            linear_action_range=config["scale_disp"],
            rot_action_range=config["scale_rot"],
            obs_h_dim=config["obs_h_dim"],
            pred_h_dim=config["pred_h_dim"],
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=config["batch_size"], shuffle=True, num_workers=0
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=config["batch_size"], shuffle=False, num_workers=0
        )

        # Build diffusion components
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon"
        )

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=config["action_dim"], global_cond_dim=config["obs_dim"] * config["obs_h_dim"]
        ).to(DEVICE)

        # EMA wrapper
        self.ema = EMAModel(parameters=self.noise_pred_net.parameters(), power=0.75)

        # Optimizer + LR scheduler
        self.optimizer = torch.optim.AdamW(
            params=self.noise_pred_net.parameters(),
            lr=config["lr"],
            weight_decay=1e-6,
        )
        total_training_steps = len(self.train_loader) * config["epochs"]
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=total_training_steps,
        )

        self.global_step = 0
        self.start_epoch = 0

        self.model_path = os.path.join(
            os.path.join(MAIN_DIR, "checkpoints"), "diffusion_" + wandb.run.name + "_best.pth"
        )

    def train(self):

        min_val_loss = np.inf
        print("\nStarting training...")
        with tqdm(range(self.start_epoch, self.config.epochs), desc="Epoch") as epoch_bar:
            for epoch in epoch_bar:
                epoch_losses = []
                self.noise_pred_net.train()
                with tqdm(self.train_loader, desc="Batch", leave=False) as batch_bar:
                    for batch in batch_bar:
                        loss_value = self._train_on_batch(batch)
                    epoch_losses.append(loss_value)
                    batch_bar.set_postfix(loss=loss_value)
                    wandb.log({"loss": loss_value}, step=self.global_step)
                self.global_step += 1

            avg_loss = float(np.mean(epoch_losses))
            wandb.log({"epoch_avg_loss": avg_loss}, step=self.global_step)
            print("Epoch: {}, step {}, train loss: {}".format(epoch, self.global_step, avg_loss))

            # Validation
            val_loss = self.validation()
            if val_loss < min_val_loss:
                print(f"Validation loss improved: {min_val_loss} -> {val_loss}")
                min_val_loss = val_loss

                # Save the best model
                print(f"Saving best model to {self.model_path}")
                state = dict(self.config)
                state["model"] = self.noise_pred_net.state_dict()
                torch.save(state, self.model_path)

    def _train_on_batch(self, batch: dict) -> float:
        """
        Performs one gradient step on a single batch and updates EMA.
        Returns the scalar loss value.
        """
        obs, action = batch

        # Move data to device
        obs = obs.to(DEVICE).float()
        action = action.to(DEVICE).float()
        batch_size = obs.shape[0]

        # Sample random noise and timesteps
        noise = torch.randn_like(action)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=DEVICE
        ).long()

        # Add noise (forward process)
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)

        # Predict noise
        noise_pred = self.noise_pred_net(noisy_action, timesteps, global_cond=obs.flatten(start_dim=1))

        # Compute L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        # Backpropagate and step optimizer + scheduler
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        # Update EMA weights
        self.ema.step(self.noise_pred_net.parameters())

        return loss.item()

    def validation(self) -> float:
        self.noise_pred_net.eval()

        val_loss = 0.0
        for batch in self.val_loader:
            obs, action = batch

            # Move data to device
            obs = obs.to(DEVICE).float()
            action = action.to(DEVICE).float()
            batch_size = obs.shape[0]

            # Sample random noise and timesteps
            noise = torch.randn_like(action)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=DEVICE
            ).long()

            # Add noise (forward process)
            noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)

            # Predict noise
            noise_pred = self.noise_pred_net(noisy_action, timesteps, global_cond=obs.flatten(start_dim=1))

            # Compute L2 loss
            loss = nn.functional.mse_loss(noise_pred, noise)

            val_loss += loss.item()

        self.noise_pred_net.train()
        return val_loss / len(self.val_loader)


if __name__ == "__main__":

    trainer = DiffusionTrainer(config=CONFIG)
    trainer.train()