import torch
import os
import copy
import numpy as np
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel

import utils
import dataset
import models


IMG_AUTOENCODER_WEIGHTS_PATH = f'ImageAutoencoder_best.pt'
SAVE_PATH = 'ImpressionCLIP_best.pt' 


def train(model_list, optimizer, dataloader, tokenizer, device):
    # Do not train the encoders
    img_encoder, clip_model, emb_img, emb_tag, temperature = model_list
    img_encoder.eval()
    clip_model.eval()
    emb_img.train()
    emb_tag.train()
    temperature.train()

    running_loss = []
    for data in dataloader:
        # Forward pass
        img, prompt = data
        img = img.to(device)
        tokenized_text = dataset.tokenize(tokenizer, prompt)
        tag_input = {key: value.to(device) for key, value in tokenized_text.items()}
        with torch.no_grad():
            img_feature = img_encoder(img)
            tag_feature = clip_model.get_text_features(**tag_input)
        embedded_img_feature = emb_img(img_feature)
        embedded_tag_feature = emb_tag(tag_feature)

        # Compute similarity matrix between image and tag features
        similarity_matrix = torch.matmul(embedded_img_feature, embedded_tag_feature.T)
        logits = temperature(similarity_matrix)
        logits_per_img = logits
        logits_per_tag = logits.T

        # Calculate loss
        criterion = torch.nn.CrossEntropyLoss().to(device)
        pair_labels = torch.arange(embedded_img_feature.size(0), device=device)
        loss_img2tag = criterion(logits_per_img, pair_labels)
        loss_tag2img = criterion(logits_per_tag, pair_labels)
        loss = (loss_img2tag + loss_tag2img) / 2
        running_loss.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(running_loss)


def val(model_list, dataloader, tokenizer, device):
    img_encoder, clip_model, emb_img, emb_tag, temperature = model_list
    img_encoder.eval()
    clip_model.eval()
    emb_img.eval()
    emb_tag.eval()
    temperature.eval()

    embedded_img_feature = []
    embedded_tag_feature = []

    with torch.no_grad():
        for data in dataloader:
            # Forward pass
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
    logits = temperature(similarity_matrix)
    logits_per_img = logits
    logits_per_tag = logits.T

    # Calculate loss
    criterion = torch.nn.CrossEntropyLoss().to(device)
    pair_labels = torch.arange(embedded_img_feature.size(0), device=device)
    loss_img2tag = criterion(logits_per_img, pair_labels)
    loss_tag2img = criterion(logits_per_tag, pair_labels)
    loss = (loss_img2tag + loss_tag2img) / 2
    
    return loss.item()


# Hyperparameters
EPOCH = 10000
BATCH_SIZE = 8192
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 30

# Fix the random seed
utils.fix_seed(7)

# Prepare data
img_paths_train, tag_paths_train = utils.get_paths('train')
img_paths_val, tag_paths_val = utils.get_paths('val')
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
train_set = dataset.FontDataset(img_paths_train, tag_paths_train, tokenizer)
val_set = dataset.FontDataset(img_paths_val, tag_paths_val, tokenizer)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers = os.cpu_count(), pin_memory=True, drop_last=True)
valloader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

# Prepare models
device = torch.device('cuda:0')
img_autoencoder = models.ImageAutoencoder(models.ResidualBlock, [2, 2]).to(device)
img_autoencoder.load_state_dict(torch.load(IMG_AUTOENCODER_WEIGHTS_PATH))
img_encoder = img_autoencoder.encoder
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
emb_img = models.MLP().to(device)
emb_tag = models.MLP().to(device)
temperature = models.ExpMultiplier(initial_value=0.07).to(device)
model_list = [img_encoder, clip_model, emb_img, emb_tag, temperature]

# Prepare optimizer and early stopping
optimizer = torch.optim.Adam([
    {'params': emb_img.parameters(),     'lr': LEARNING_RATE},
    {'params': emb_tag.parameters(),     'lr': LEARNING_RATE},
    {'params': temperature.parameters(), 'lr': LEARNING_RATE}
    ])
earlystopping = utils.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=0)


val_loss_best = np.Inf
for epoch in tqdm(range(1, EPOCH+1)):
    # Training and validation
    train_loss = train(model_list, optimizer, trainloader, tokenizer, device)
    val_loss = val(model_list, valloader, tokenizer, device)
    print(f'[train] Epoch: {epoch}, loss: {train_loss:.4f}')
    print(f'[val]   Epoch: {epoch}, loss: {val_loss:.4f}')

    # Save model weights state if validation loss improves
    if val_loss < val_loss_best:
        best_emb_img = copy.deepcopy(emb_img.state_dict())
        best_emb_tag = copy.deepcopy(emb_tag.state_dict())
        best_temperature = copy.deepcopy(temperature.state_dict())
        val_loss_best = val_loss

    # Check early stopping criteria
    if EARLY_STOPPING_PATIENCE:
        earlystopping(val_loss)
        if earlystopping.early_stop:
            print('Early stopping')
            break

# Save best model weights
state = {
    'emb_img': best_emb_img,
    'emb_tag': best_emb_tag,
    'temperature': best_temperature
    }
torch.save(state, SAVE_PATH)