import glob
import os
import pickle

dataset_path = "/home/mengo/research/dlo_diffusion/DATA/train"

data_files = glob.glob(os.path.join(dataset_path, "*.pkl"))
print("Found {} files in dataset {}".format(len(data_files), dataset_path))
data = pickle.load(open(data_files[0], "rb"))
for key in data.keys():
    print(f"{key}")
print(data["observation"].shape)