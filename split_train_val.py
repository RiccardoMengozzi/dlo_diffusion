import shutil, glob, os, argparse
from sklearn.model_selection import train_test_split

# ******************
RATIO = 0.1
# ******************

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", required=True)
    args = args.parse_args()
    print(args)

    dataset_path = args.dataset_path
    if "/" == dataset_path[-1]:
        dataset_path = dataset_path[:-1]

    dataset_dir_path = os.path.dirname(dataset_path)
    dataset_raw_folder = os.path.basename(dataset_path)

    dataset_train_folder = dataset_raw_folder + "_train"
    dataset_val_folder = dataset_raw_folder + "_val"

    raw_path = os.path.join(dataset_dir_path, dataset_raw_folder)
    train_path = os.path.join(dataset_dir_path, dataset_train_folder)
    val_path = os.path.join(dataset_dir_path, dataset_val_folder)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    files_list = glob.glob(os.path.join(raw_path, "*.pkl"))
    files_names = [f.split("/")[-1] for f in files_list]
    train, test = train_test_split(files_names, test_size=RATIO)
    print("train size: {}, test size: {}".format(len(train), len(test)))

    for sample in train:
        path_old_file = os.path.join(raw_path, sample)
        path_new_file = os.path.join(train_path, sample)

        print("from {} to {}...".format(path_old_file, path_new_file))
        shutil.copyfile(path_old_file, path_new_file)

    for sample in test:
        path_old_file = os.path.join(raw_path, sample)
        path_new_file = os.path.join(val_path, sample)

        print("from {} to {}...".format(path_old_file, path_new_file))
        shutil.copyfile(path_old_file, path_new_file)