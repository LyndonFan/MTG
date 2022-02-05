import pickle
from sklearn.model_selection import train_test_split
import draftsimtools as ds
import pandas as pd

dataset_path = "../../data/"
rating_path1 = dataset_path + "m19_rating.tsv"
rating_path2 = dataset_path + "m19_land_rating.tsv"
train_path = dataset_path + "train.csv"
test_path = dataset_path + "test.csv"
output_folder = "./bots_data/nnet_train/"


def serialize_data(obj, path):
    """
    Serialize an object as a python pickle file.
    """
    with open(path, "wb+") as f:
        pickle.dump(obj, f)


cur_set = ds.create_set(rating_path1, rating_path2)
train = ds.load_drafts(train_path)
cur_set, train = ds.fix_commas(cur_set, train)
le = ds.create_le(cur_set["Name"].values)
drafts_train = ds.process_drafts(train)
drafts_train = [d for d in drafts_train if len(
    d) == 45]  # Remove incomplete drafts


serialize_data(drafts_train, output_folder + "drafts_train.pkl")

print(drafts_train[0])

# Runs for ~15 minutes.
drafts_tensor_train = ds.drafts_to_tensor(drafts_train, le)

serialize_data(drafts_tensor_train, output_folder + "drafts_tensor_train.pkl")


test = ds.load_drafts(test_path)
cur_set, test = ds.fix_commas(cur_set, test)
drafts_test = ds.process_drafts(test)
drafts_test = [d for d in drafts_test if len(
    d) == 45]  # Remove incomplete drafts


serialize_data(drafts_test, output_folder + "drafts_test.pkl")

print(drafts_test[0])

# Runs for ~15 minutes.
drafts_tensor_test = ds.drafts_to_tensor(drafts_test, le)

serialize_data(drafts_tensor_test, output_folder + "drafts_tensor_test.pkl")
