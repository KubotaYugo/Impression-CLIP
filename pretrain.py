import os
import torch
import copy
import numpy as np
from tqdm import tqdm

import utils
import models
import dataset


def train(model, optimizer, criterion, device, dataloader):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for inputs in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
    return total_loss / total_samples

def val(model, criterion, device, dataloader):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples


SAVE_PATH = 'ImageAutoencoder_best.pt'

# Hyperparameters
EPOCH = 10000
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 50

# Fix the random seed
utils.fix_seed(7)

# Prepare data
img_paths_train, _ = utils.get_paths('train')
img_paths_val, _ = utils.get_paths('val')
train_set = dataset.ImageDataset(img_paths_train)
val_set = dataset.ImageDataset(img_paths_val)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers = os.cpu_count(), pin_memory=True)
valloader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers = os.cpu_count(), pin_memory=True)

# Prepare model, loss function, and evaluation metric, earlystopping
device = torch.device('cuda:0')
model = models.ImageAutoencoder(models.ResidualBlock, [2, 2]).to(device)
criterion = torch.nn.L1Loss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
earlystopping = utils.EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=0)

# Training
val_loss_best = np.Inf
for epoch in tqdm(range(1, EPOCH+1)):
    train_loss = train(model, optimizer, criterion, device, trainloader)
    val_loss = val(model, criterion, device, valloader)
    print(f'[train] Epoch: {epoch}, loss: {train_loss:.4f}')
    print(f'[val]   Epoch: {epoch}, loss: {val_loss:.4f}')

    if val_loss < val_loss_best:
        best_model = copy.deepcopy(model)
        val_loss_best = val_loss
        best_epoch = epoch
    
    if EARLY_STOPPING_PATIENCE:
        earlystopping(val_loss)
        if earlystopping.early_stop:
            print('Early stopping')
            break

torch.save(best_model.state_dict(), SAVE_PATH)