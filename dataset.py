from torch.utils.data import Dataset
import numpy as np

import utils


def tokenize(tokenizer, prompt):
    tokenized_text = tokenizer(prompt, return_tensors='pt', 
                               max_length=tokenizer.model_max_length, 
                               padding='max_length', truncation=True)
    return tokenized_text

class ImageDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        font = np.load(self.img_paths[idx])['arr_0'].astype(np.float32)
        return font


def get_prompt(tags):
    if len(tags)==1:
        prompt = f'The impression is {tags[0]}.'
    elif len(tags) == 2:
        prompt = f'First and second impressions are {tags[0]} and {tags[1]}, respectively.'
    elif len(tags) >= 3:
        ordinal = ['First', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
        prompt1 = ordinal[0]
        prompt2 = tags[0]
        i = 0
        for i in range(1, min(len(tags)-1, 10-1)):
            prompt1 = prompt1 + ', ' + ordinal[i]
            prompt2 = prompt2 + ', ' + tags[i]
        prompt1 = prompt1 + ', and ' + ordinal[i+1] + ' impressions are '
        prompt2 = prompt2 + ', and ' + tags[i+1] + ', respectively.'                
        prompt = prompt1 + prompt2
    return prompt

class FontDataset(Dataset):
    def __init__(self, img_paths, tag_paths, tokenizer):
        self.img_paths = img_paths
        self.tag_paths = tag_paths
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        # font
        font = np.load(self.img_paths[idx])['arr_0'].astype(np.float32)
        # tags
        tags = utils.load_csv(self.tag_paths[idx])       
        prompt = get_prompt(tags)
        return font, prompt