import csv
import random
import numpy as np
import torch


# Path to the MyFonts dataset
MYFONTS_PATH = 'MyFonts'

# Garbled tags included in the MyFonts dataset
INVALID_TAGS = [
    '%d0%b3%d1%80%d0%be%d1%82%d0%b5%d1%81%d0%ba', 
    '%d0%ba%d0%b8%d1%80%d0%b8%d0%bb%d0%bb%d0%b8%d1%86%d0%b0', 
    '%d9%86%d8%b3%d8%ae'
    ]


def get_fontnames(dataset):
    fontnames = load_csv(f'fontnames/{dataset}.csv', 'ndarray')[:,0]
    return fontnames
    
def get_paths(dataset):
    fontnames = get_fontnames(dataset)
    img_paths = [f'dataset/fonts/{dataset}/{fontname}.npz' for fontname in fontnames]
    tag_paths = [f'dataset/tags/{dataset}/{fontname}.csv' for fontname in fontnames]
    return img_paths, tag_paths

def get_org_tags(fontname, myfonts_path=MYFONTS_PATH):
    tag_path = f'{myfonts_path}/taglabel/{fontname}'
    with open(tag_path, encoding='utf8', newline='') as f:
        csvreader = csv.reader(f, delimiter=' ')
        tags = [row for row in csvreader]
    return tags[0][:-1]

def load_csv(path, mode='list'):
    with open(path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = list(reader)
    if mode=='list':
        return data
    elif mode=='ndarray':
        return np.asarray(data)
    
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class EarlyStopping:
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.min_val_loss = np.Inf
    def __call__(self, val_loss):
        if self.min_val_loss+self.delta <= val_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            if  val_loss < self.min_val_loss:
                print(f'Validation loss decreased ({self.min_val_loss} --> {val_loss})')
                self.min_val_loss = val_loss