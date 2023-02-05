import time, copy
import torch
from _checkpoint import SaveBestModel, save_model
from _pytorchtools import EarlyStopping

def train_model(model, logging, criterion, optimizer, scheduler, dataloaders, dataset_sizes, checkpoint_path, patience, num_epochs=25):
    since = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_acc = 0.0
    save_best_model = SaveBestModel(checkpoint_path)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
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
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val':
                early_stopping(epoch_loss)
                if early_stopping.early_stop:
                    logging.info(f"Early stopping for epoch: {epoch}")
                    break

                save_best_model(epoch_acc, epoch, model, optimizer, criterion)

        print()

    time_elapsed = time.time() - since
    logging.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logging.info(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model = torch.load(checkpoint_path)
    return model