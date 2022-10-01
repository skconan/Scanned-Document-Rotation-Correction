import os
import time
import numpy as np
from utils.file import *
from datetime import datetime, timedelta

from torchsummary import summary
from models.rotation_net import RotationNet

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def load_dataset(path, batch_size, device):
    if isinstance(path, list):
        dataset = []
        for p in path:
            dataset += torch.load(p, map_location=device)
            print(p, 'loaded')
    else:
        dataset = torch.load(path, map_location=device)
        print(path, 'loaded')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Dataset Length: {len(dataset)}")
    return dataloader


def train(dataloader, model, loss_fn, optimizer):
    loss = 0
    running_loss = 0.
    for batch, (X, y) in enumerate(dataloader):
        y = y.float().cuda().squeeze()
        X = X.float().cuda()

        # Compute prediction error
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)
        running_loss += loss.item()

        # Backpropagation
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # updates the gradients of the model, update the weights and bias
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Need Adjust depend on number of dataset
        if batch % 10 == 0:
            last_loss = running_loss / 10  # loss per batch
            running_loss = 0.

    return last_loss


def validate(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    valid_loss = 0

    # we do not want these actions to be recorded for our next calculation of the gradient
    with torch.no_grad():
        for X, y in dataloader:
            y = y.float().cuda().squeeze()
            X = X.float().cuda()

            # prediction
            pred = model(X).squeeze()

            # Compute prediction error
            loss = loss_fn(pred, y)
            valid_loss += loss.item()

    return valid_loss/num_batches


def angle_difference(y_predict, y_true):
    err = torch.abs(180 - torch.abs(torch.abs(y_true - y_predict) - 180))
    return err**2


def angle_error_regression(y_pred, y_true):
    return torch.mean(angle_difference(y_pred, y_true))


if __name__ == '__main__':
    gpu_id = 0
    img_size = 128
    batch_size = 128
    DATASET_DIR = './dataset'

    training_set_dir = os.path.join(
        DATASET_DIR, 'training_set_white_bg_128_90_degree_no_crop')
    validation_set_dir = os.path.join(
        DATASET_DIR, 'validation_set_white_bg_128_90_degree_no_crop')

    model_dir = './models'
    save_dir = os.path.join(
        model_dir, 'weight-white-bg-angle-regression-90-conv-fc-01')
    make_dirs(save_dir)

    device = torch.device('cuda', 0)
    torch.cuda.set_device(gpu_id)

    print("Set device", device)
    print("Current GPU Id:", torch.cuda.current_device())

    model = RotationNet(out_channels=64)
    model = model.cuda()
    summary(model, (1, img_size, img_size))

    train_set_path, training_length = list_files(training_set_dir)
    valid_set_path, validation_length = list_files(validation_set_dir)

    print(f'Batch Size: {batch_size}')
    print(f'Device: {device}')

    train_dataloader = load_dataset(train_set_path, batch_size, device)
    valid_dataloader = load_dataset(valid_set_path, batch_size, device)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    criterion = nn.MSELoss()
    # criterion = angle_error_regression
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(
        'runs/scanned_doc_rot_correction_{}'.format(timestamp))

    data_out = os.path.join(save_dir, "xhistory.csv")

    data = []
    data.append(['Epoch', 'TrainLoss', 'ValidLoss'])

    s_epochs = 0
    n_epochs = s_epochs + 1000
    save_epoch = 10
    minimum_loss = 1000
    training_time = 0

    for epoch in range(s_epochs, n_epochs):
        start_time = time.time()
        train_loss = train(train_dataloader,
                           model, criterion, optimizer)
        valid_loss = validate(valid_dataloader, model, criterion)
        data.append([str(epoch), "%.6f" % train_loss, "%.6f" % valid_loss])

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': train_loss, 'Validation': valid_loss},
                           epoch + 1)
        writer.flush()

        epoch_time = (time.time() - start_time)
        training_time += epoch_time
        print(
            'Epoch %d Train Loss: %.3f \tValid Loss: %.3f \tEpoch time: %s \tTotal time: %s'
            % (epoch+1, train_loss, valid_loss, timedelta(seconds=epoch_time),
               timedelta(seconds=training_time)))

        if epoch % save_epoch == 0 or valid_loss < minimum_loss:
            model_path = os.path.join(save_dir, "model_valid_loss_%.6f_loss_%.6f_epoch_%05d.pt" % (
                valid_loss, train_loss, epoch+1))
            torch.save(model.state_dict(), model_path)
            if valid_loss < minimum_loss:
                minimum_loss = valid_loss
            print("%s Saved" % model_path)
            np.savetxt(data_out, data, delimiter=',', fmt='%s')
    print("Done!")

    np.savetxt(data_out, data, delimiter=',', fmt='%s')


