from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import json
import os
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data

random.seed(1)
np.random.seed(1)
NUM_USERS = 20
NUM_LABELS = 10

# Setup directory for train/test data
train_path = './data/train/fashion_train.json'
test_path = './data/test/fashion_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
mnist_data_image = []
mnist_data_lable = []
mnist_data_image.extend(train_images)
mnist_data_image.extend(test_images)
mnist_data_lable.extend(train_labels)
mnist_data_lable.extend(test_labels)

mu = np.mean(mnist_data_image)
sigma = np.std(mnist_data_image)
nom_Fashion_data = (mnist_data_image - mu)/(sigma+0.001)
nom_Fashion_lable = np.array(mnist_data_lable)
mnist_data = []
for i in trange(10):
    idx = nom_Fashion_lable == i
    mnist_data.append(nom_Fashion_data[idx])

print([len(v) for v in mnist_data])

users_lables = []

print("idx", idx)
# devide for label for each users:
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 4 labels for each users
        l = (user + j) % 10
        users_lables.append(l)
unique, counts = np.unique(users_lables, return_counts=True)
print("--------------")
print(unique, counts)


def ram_dom_gen(total, size):
    print(total)
    nums = []
    temp = []
    for i in range(size - 1):
        val = np.random.randint(total // (size + 1), total // (size - 8))
        temp.append(val)
        total -= val
    temp.append(total)
    print(temp)
    return temp


number_sample = []
for total_value, count in zip(mnist_data, counts):
    temp = ram_dom_gen(len(total_value), count)
    number_sample.append(temp)
print("--------------")
print(number_sample)

i = 0
number_samples = []
for i in range(len(number_sample[0])):
    for sample in number_sample:
        print(sample)
        number_samples.append(sample[i])

print("--------------")
print(number_samples)

###### CREATE USER DATA SPLIT #######
# Assign 100 samples to each user
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
count = 0
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 4 labels for each users
        l = (user + j) % 10
        print("value of L", l)
        print("value of count", count)
        num_samples = number_samples[count]  # num sample
        count = count + 1
        if idx[l] + num_samples < len(mnist_data[l]):
            X[user] += mnist_data[l][idx[l]:num_samples].tolist()
            y[user] += (l * np.ones(num_samples)).tolist()
            idx[l] += num_samples
            print("check len os user:", user, j, "len data", len(X[user]), num_samples)

print("IDX2:", idx)  # counting samples for each labels

# Create data structure
train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}

# Setup 5 users
# for i in trange(5, ncols=120):
for i in range(NUM_USERS):
    uname = 'f_{0:05d}'.format(i)

    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.75 * num_samples)
    test_len = num_samples - train_len

    train_data['users'].append(uname)
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)

print("Num_samples:", train_data['num_samples'])
print("Total_samples:", sum(train_data['num_samples'] + test_data['num_samples']))

with open(train_path, 'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")
