import glob
import os
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

consolidated_dir = "/home/mengo/research/dlo_diffusion/train_episodes"
episode_files = glob.glob(os.path.join(consolidated_dir, "*.pkl"))

if not episode_files:
    raise FileNotFoundError("No episode_*.pkl files found.")

lengths = []
for episode_file in tqdm(episode_files):
    with open(episode_file, "rb") as f:
        data = pickle.load(f)
    lengths.append(data["number_of_actions"])

# Count occurrences of each length
length_counts = Counter(lengths)

# --- Bar plot ---
plt.figure(figsize=(8, 5))
plt.bar(length_counts.keys(), length_counts.values(), width=0.8)
plt.xlabel("Episode length (# actions)")
plt.ylabel("Count")
plt.title("Distribution of Episode Lengths")
plt.xticks(sorted(length_counts.keys()))
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
