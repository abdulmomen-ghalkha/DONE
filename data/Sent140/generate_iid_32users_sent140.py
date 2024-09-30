from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import json
import os

import itertools
import re
import string
import unicodedata

import torch
from torch.utils.data import Dataset, IterableDataset, random_split
from sklearn.model_selection import train_test_split

# 1. The Sent140Dataset will store the tweets and corresponding sentiment for each user.

class Sent140Dataset(Dataset):
    def __init__(self, data_root, max_seq_len):
        self.data_root = data_root
        self.max_seq_len = max_seq_len
        self.all_letters = {c: i for i, c in enumerate(string.printable)}
        self.num_letters = len(self.all_letters)
        self.UNK = self.num_letters

        with open(data_root, "r+") as f:
            self.dataset = json.load(f)

        self.data = {}
        self.targets = {}
        self.num_classes = 2  # binary sentiment classification
        self.length = 0

        # Populate self.data and self.targets
        for user_id, user_data in self.dataset["user_data"].items():
            for (i, j) in zip(self.process_x(list(user_data["x"])), self.process_y(list(user_data["y"]))):
              self.data[self.length] = i.numpy()
              self.targets[self.length] = np.int32(j)
              self.length += 1



    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for user_id in self.data.keys():
            yield self.__getitem__(user_id)

    def __getitem__(self, user_id: str):
        if user_id not in self.data or user_id not in self.targets:
            raise IndexError(f"User {user_id} is not in dataset")
        return self.data[user_id], self.targets[user_id]

    def unicodeToAscii(self, s):
        return "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn" and c in self.all_letters
        )

    def line_to_indices(self, line: str, max_seq_len: int):
        line_list = self.split_line(line)  # split phrase in words
        line_list = line_list
        chars = self.flatten_list([list(word) for word in line_list])
        indices = [
            self.all_letters.get(letter, self.UNK)
            for i, letter in enumerate(chars)
            if i < max_seq_len
        ]
        # Add padding
        indices = indices + [self.UNK] * (max_seq_len - len(indices))
        return indices

    def process_x(self, raw_x_batch):
        x_batch = [e[4] for e in raw_x_batch]  # e[4] contains the actual tweet
        x_batch = [self.line_to_indices(e, self.max_seq_len) for e in x_batch]
        x_batch = torch.LongTensor(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [int(e) for e in raw_y_batch]
        return y_batch

    def split_line(self, line):
        """
        Split given line/phrase (str) into list of words (List[str])
        """
        return re.findall(r"[\w']+|[.,!?;]", line)

    def flatten_list(self, nested_list):
        return list(itertools.chain.from_iterable(nested_list))


random.seed(1)
np.random.seed(1)
NUM_USERS = 32  
NUM_LABELS = 2
import os
print(os.getcwd())
# Setup directory for train/test data
train_path = './data/Sent140/data/train/sent140_train.json'
test_path = './data/Sent140/data/test/sent140_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


#LOCAL_BATCH_SIZE = 32
MAX_SEQ_LEN = 50

# suppress large outputs
VERBOSE = False

TRAIN_DATA = "./data/Sent140/all_data_0_01_0_keep_200_train_9.json"
TEST_DATA = "./data/Sent140/all_data_0_01_0_keep_200_test_9.json"


mnist_data = []

# 2. Load the train and test datasets.
train_dataset = Sent140Dataset(
    data_root=TRAIN_DATA,
    max_seq_len=MAX_SEQ_LEN,
)
test_dataset = Sent140Dataset(
    data_root=TEST_DATA,
    max_seq_len=MAX_SEQ_LEN,
)

# Total dataset size

total_size = len(train_dataset)

# Define the sizes for each user (assuming equal split)
user_size = total_size // NUM_USERS
sizes = [user_size] * NUM_USERS
remainder = total_size % NUM_USERS
# Distribute the remainder across some users
for i in range(remainder):
    sizes[i] += 1

user_datasets = random_split(train_dataset, sizes)




# Convert each user's dataset into numpy arrays and store in a list
X = []
y = []

for user_dataset in user_datasets:
    user_data = []
    user_targets = []
    
    for data, target in user_dataset:
        user_data.append(data)  # Convert image tensor to numpy
        user_targets.append(target)  # Convert label tensor to numpy int32
    
    # Combine the data and targets into a tuple of numpy arrays (data, labels)
    user_data_np = np.array(user_data)
    user_targets_np = np.array(user_targets, dtype=np.int32)
    
    X.append(user_data_np)  # Append to users' list
    y.append(user_targets_np)



train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

for i in range(NUM_USERS):
    uname = 'f_{0:05d}'.format(i)
    

    X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])

    train_data["user_data"][uname] = {'x': X_train.tolist(), 'y': y_train.tolist()}
    train_data['users'].append(uname)
    train_data['num_samples'].append(len(y_train))
    
    test_data['users'].append(uname)
    test_data["user_data"][uname] = {'x': X_test.tolist(), 'y': y_test.tolist()}
    test_data['num_samples'].append(len(y_test))
    print(uname)

    
print("train", train_data['num_samples'])
print("test", test_data['num_samples'])
print("Num_samples:", train_data['num_samples']+ test_data['num_samples'])
print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")

