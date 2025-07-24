import torch

# Load the state dictionary
state_path = "diffusion_mild-sponge-2_best.pt"
state = torch.load(state_path)

# Change 'obs_dim' to 207
state["obs_dim"] = 206

# Save the modified state dictionary
torch.save(state, state_path)