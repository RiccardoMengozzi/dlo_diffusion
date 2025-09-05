import os
import pickle
import numpy as np
from tqdm import tqdm
import glob


INPUT_DIR = "/home/mengo/research/dlo_diffusion/train"
OUTPUT_DIR = "/home/mengo/research/dlo_diffusion/train_episodes"
ACTION_HORIZON = 16
PERCENTAGE_OF_DATASET = 1
MIN_EPISODE_LENGTH = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all pkl files and sort them
pkl_files = glob.glob(os.path.join(INPUT_DIR, "*.pkl"))
print(f"Found {len(pkl_files)} pkl files")

# Sort them by (file_number, action_number)
def parse_filename(path):
    filename = os.path.basename(path).replace(".pkl", "")
    file_number, action_number = filename.split("_")
    return int(file_number), int(action_number)

pkl_files = sorted(pkl_files, key=lambda f: parse_filename(f))
pkl_files = pkl_files[:int(len(pkl_files)*PERCENTAGE_OF_DATASET)]

first_episode = True
previous_action_number = 1e6
actions_data = []
output_episode = {}

episode_idx = 0
number_of_discarded_episodes = 0
for pkl_file in tqdm(pkl_files, desc="files"):
    filename = os.path.basename(pkl_file)
    number_file_and_action = filename.replace('.pkl', '')
    file_number, action_number = map(int, number_file_and_action.split('_'))

    # Load sample first
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    if data["observation"].shape[0] > ACTION_HORIZON:
        actions_data.append(data)

    # Check if we started a new episode
    if action_number < previous_action_number:
        if not first_episode:
            output_episode = {
                'episode_idx': episode_idx,
                'number_of_actions': len(actions_data),
                'actions_data': actions_data,
            }

            if output_episode["number_of_actions"] >= MIN_EPISODE_LENGTH:
                output_filename = f"{episode_idx:04d}.pkl"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                with open(output_path, 'wb') as f:
                    pickle.dump(output_episode, f)
            else:
                number_of_discarded_episodes += 1

            episode_idx += 1
            actions_data = []
        else:
            first_episode = False

    previous_action_number = action_number

# Final flush after loop ends
if actions_data:
    output_episode = {
        'episode_idx': episode_idx,
        'number_of_actions': len(actions_data),
        'actions_data': actions_data,
    }
    if output_episode["number_of_actions"] >= MIN_EPISODE_LENGTH:
        output_filename = f"{episode_idx:04d}.pkl"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, 'wb') as f:
            pickle.dump(output_episode, f)
    else:
        number_of_discarded_episodes += 1

print("DISCARDED EPISODES =", number_of_discarded_episodes)
