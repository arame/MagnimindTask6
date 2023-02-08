import matplotlib.pyplot as plt

def accuracy_loss_graph(epoch_train_accuracy, epoch_train_losses, epoch_val_accuracy, epoch_val_losses):
    epochs = []
    for i in range(len(epoch_train_losses)):
        epochs.append(i + 1)
    plt.title("Epoch readings for accuracy and losses")
    plt.plot(epochs, epoch_train_accuracy, label="Training Accuracy")
    plt.plot(epochs, epoch_train_losses, label="Training Losses")
    plt.plot(epochs, epoch_val_accuracy, label="Validation Accuracy")
    plt.plot(epochs, epoch_val_losses, label="Validation Losses")
    plt.legend()
    plt.show()