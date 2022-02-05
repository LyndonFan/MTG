import draftsimtools as ds
import pickle
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# dataset_path = "../../data/"
# rating_path1 = dataset_path + "m19_rating.tsv"
# rating_path2 = dataset_path + "m19_land_rating.tsv"
# drafts_path = dataset_path + "train.csv"
# cur_set = ds.create_set(rating_path1, rating_path2)
# raw_drafts = ds.load_drafts(drafts_path)
# cur_set, raw_drafts = ds.fix_commas(cur_set, raw_drafts)
# le = ds.create_le(cur_set["Name"].values)
# drafts = ds.process_drafts(raw_drafts)
# drafts = [d for d in drafts if len(d) == 45]  # Remove incomplete drafts

# # Separates the training data into thirds
# third = int(len(drafts) / 3)
# fold1 = drafts[0:third]
# fold2 = drafts[third:(third*2)]
# fold3 = drafts[(third*2):(len(drafts) + 1)]

# # Gets CV splits and converts to tensor

# full_tensor = ds.drafts_to_tensor(drafts, le)
# split1_train = full_tensor[0:(third*2)]
# split1_val = full_tensor[(third*2):(len(drafts) + 1)]
# split2_train = full_tensor[np.r_[0:(third), (third*2):(len(drafts) + 1)]]
# split2_val = full_tensor[third:(third*2)]
# split3_train = full_tensor[third:(len(drafts) + 1)]
# split3_val = full_tensor[0: third]

# # split1_train = ds.drafts_to_tensor(fold1 + fold2, le)
# # split1_val = ds.drafts_to_tensor(fold3, le)
# # split2_train = ds.drafts_to_tensor(fold1 + fold3, le)
# # split2_val = ds.drafts_to_tensor(fold2, le)
# # split3_train = ds.drafts_to_tensor(fold2 + fold3, le)
# # split3_val = ds.drafts_to_tensor(fold1, le)

# # Helper function for pickling data

dataset_path = "bots_data/nnet_train/"


def load_data(path):
    """
    Load a pickle file from disk. 
    """
    with open(path, "rb") as f:
        return pickle.load(f)


drafts = load_data(dataset_path + "drafts_tensor_train.pkl")
print(type(drafts))
third = int(len(drafts) / 3)
split1_train = drafts[0:third*2]
split1_val = drafts[third*2:len(drafts)]
split2_train = drafts[np.r_[0:third, third*2:len(drafts)]]
split2_val = drafts[third:third*2]
split3_train = drafts[third:len(drafts)]
split3_val = drafts[0: third]


def serialize_data(obj, path):
    """
    Serialize an object as a python pickle file.
    """
    with open(path, "wb+") as f:
        pickle.dump(obj, f)


# Writes all splits to file
output_folder = "bots_data/nnet_train/"
serialize_data(split1_train, output_folder + "split1_train.pkl")
serialize_data(split1_val, output_folder + "split1_val.pkl")
serialize_data(split2_train, output_folder + "split2_train.pkl")
serialize_data(split2_val, output_folder + "split2_val.pkl")
serialize_data(split3_train, output_folder + "split3_train.pkl")
serialize_data(split3_val, output_folder + "split3_val.pkl")
