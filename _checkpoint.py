import torch
import matplotlib.pyplot as plt

plt.style.use('ggplot')
# Code adapted from debuggercafe.com
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation accuracy is closer to 1 than the previous accuracy, then save the
    model state.
    """
    def __init__(
        self, checkpoint_path, best_valid_accuracy=float(0)
    ):
        self.best_valid_accuracy = best_valid_accuracy
        self.checkpoint_path = checkpoint_path
        
    def __call__(
        self, current_valid_accuracy, epoch, model, optimizer, criterion
    ):
        if current_valid_accuracy > self.best_valid_accuracy:
            self.best_valid_accuracy = current_valid_accuracy
            print(f"\nBest validation accuracy: {self.best_valid_accuracy}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, self.checkpoint_path)

def save_model(epochs, model, optimizer, criterion, final_checkpoint_path):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, final_checkpoint_path)

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')