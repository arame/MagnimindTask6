import time
import numpy as np
import torch
from _checkpoint import SaveBestModel
from _pytorchtools import EarlyStopping

def train_model(model, logging, criterion, optimizer, scheduler, dataloaders, dataset_sizes, checkpoint_path, patience, num_epochs=25):
    since = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_best_model = SaveBestModel(checkpoint_path)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    epoch_train_losses = []
    epoch_train_accuracy = []
    epoch_val_losses = []
    epoch_val_accuracy = []
    for epoch in range(1, num_epochs + 1):
        logging.info(f'Epoch {epoch}/{num_epochs}')
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # check for early stopping and if instead the best model, save it as checkpoint
            if phase == 'val':
                epoch_val_accuracy.append(epoch_acc)
                epoch_val_losses.append(epoch_loss)
                early_stopping(epoch_acc)
                if early_stopping.early_stop == False:
                    save_best_model(epoch_acc, epoch, model, optimizer, criterion)
            else:
                epoch_train_accuracy.append(epoch_acc)
                epoch_train_losses.append(epoch_loss)                

        if early_stopping.early_stop:
            logging.info(f"Early stopping for epoch: {epoch}")
            break

    time_elapsed = time.time() - since
    logging.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logging.info(f'Best val Acc: {early_stopping.best_score:4f}')

    # load best model weights
    model = torch.load(checkpoint_path)
    return model, epoch_train_accuracy, epoch_train_losses, epoch_val_accuracy, epoch_val_losses