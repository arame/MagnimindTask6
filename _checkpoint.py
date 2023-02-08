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
            print(f"Best validation accuracy: {self.best_valid_accuracy}")
            print(f"Saving best model for epoch: {epoch}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'accuracy': self.best_valid_accuracy,
                }, self.checkpoint_path)
