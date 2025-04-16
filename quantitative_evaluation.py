import torch
import os
import numpy as np
from transformers import CLIPTokenizer, CLIPModel

import utils
import dataset
import models


IMG_AUTOENCODER_WEIGHTS_PATH = f'ImageAutoencoder_best.pt'
IMPRESSION_CLIP_WEIGHTS_PATH = 'ImpressionCLIP_best.pt' 
DATASET = 'test'


def average_retrieval_rank(similarity_matrix, mode=None):
    """
    Computes the average retrieval rank from similarity matrix.
    mode: 'img2tag' calculates image-to-tag rank, 'tag2img' calculates tag-to-image rank.
    """
    if mode=='img2tag':
        pass
    elif mode=='tag2img':
        similarity_matrix = similarity_matrix.T
    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    ranks = [int((sorted_indices[i] == i).nonzero(as_tuple=True)[0]) + 1 for i in range(sorted_indices.size(0))]
    return np.mean(ranks)


# Prepare data
img_paths, tag_paths = utils.get_paths(DATASET)
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
dataset_ = dataset.FontDataset(img_paths, tag_paths, tokenizer)
dataloader = torch.utils.data.DataLoader(dataset_, batch_size=8192, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

# Prepare models
device = torch.device('cuda:0')
img_autoencoder = models.ImageAutoencoder(models.ResidualBlock, [2, 2]).to(device)
img_autoencoder.load_state_dict(torch.load(IMG_AUTOENCODER_WEIGHTS_PATH))
img_encoder = img_autoencoder.encoder
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
emb_img = models.MLP().to(device)
emb_tag = models.MLP().to(device)
Impression_CLIP_weights = torch.load(IMPRESSION_CLIP_WEIGHTS_PATH)
emb_img.load_state_dict(Impression_CLIP_weights['emb_img'])
emb_tag.load_state_dict(Impression_CLIP_weights['emb_tag'])

# embedding font images and impression tags
embedded_img_feature = []
embedded_tag_feature = []

with torch.no_grad():
    for data in dataloader:
        img, prompt = data
        img = img.to(device)
        tokenized_text = dataset.tokenize(tokenizer, prompt)
        tag_input = {key: value.to(device) for key, value in tokenized_text.items()}
        img_feature = img_encoder(img)
        tag_feature = clip_model.get_text_features(**tag_input)
        embedded_img_feature.append(emb_img(img_feature))
        embedded_tag_feature.append(emb_tag(tag_feature))

embedded_img_feature = torch.cat(embedded_img_feature, dim=0)
embedded_tag_feature = torch.cat(embedded_tag_feature, dim=0)

# Compute similarity matrix between image and tag features
similarity_matrix = torch.matmul(embedded_img_feature, embedded_tag_feature.T)

# Compute average retrieval rank for img2tag and tag2img
arr_img2tag = average_retrieval_rank(similarity_matrix, mode='img2tag')
arr_tag2img = average_retrieval_rank(similarity_matrix, mode='tag2img')
print(f'average retrieval rank img2tag: {arr_img2tag:.1f}')
print(f'average retrieval rank tag2img: {arr_tag2img:.1f}')