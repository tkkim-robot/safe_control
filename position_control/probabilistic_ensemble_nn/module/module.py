import pandas as pd
import numpy as np
import csv
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
# import pyautogui


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, item):
        x_ = torch.FloatTensor(self.x_data[item])
        y_ = torch.FloatTensor(self.y_data[item])
        return x_, y_
    

def normalize(state):
    state[0] = state[0] / 10.0
    state[1] = state[1] / 5.0
    state[2] = state[2] / 1.0
    # state[3] = state[3] / 0.5
    # state[4] = state[4] / 1.0

    return state


def denormalize(state):
    state[:, 0] = state[:, 0] * 10.0
    state[:, 1] = state[:, 1] * 5.0
    state[:, 2] = state[:, 2] * 1.0
    # state[:, 3] = state[:, 3] * 0.5
    # state[:, 4] = state[:, 4] * 1.0

    return state


def inject_sensor_noise(state):
    # inject sensor noise up to 10 %
    # ex) vx range from 0 ~ 10, 10% of the range is 1.0
    #     3 * sigma = 1.0
    # state[0] += np.random.normal(0.0, 0.3)
    # state[1] += np.random.normal(0.0, 0.15)
    # state[2] += np.random.normal(0.0, 0.03)
    state[0] += np.random.normal(0.0, 0.6)
    state[1] += np.random.normal(0.0, 0.3)
    state[2] += np.random.normal(0.0, 0.06)
    return state


def replace_in_file(file_path, find_name, old, new):
    fr = open(file_path, 'r')
    lines = fr.readlines()
    fr.close()

    fw = open(file_path, 'w')
    for line in lines:
        fw.write(line.replace(find_name + str(old),
                 find_name + str(new)))
