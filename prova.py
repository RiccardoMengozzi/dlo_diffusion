import numpy as np
import pickle

np.set_printoptions(precision=5, suppress=True)

with open("/home/mengo/Research/LLM_DOM/pyelastica/pyel_simulation/generate_dataset/dataset_20250826_155347/00000_00.pkl", "rb") as f:
    data = pickle.load(f)

init_shape = data["init_shape"].T
final_shape = data["final_shape"].T

# force commas between values
print("init_shape =\n", np.array2string(init_shape, separator=", "))
print("final_shape =\n", np.array2string(final_shape, separator=", "))
