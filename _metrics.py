from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, matthews_corrcoef
import matplotlib.pyplot as plt

def display_metrics(y_test, y_pred, target_names, title = None):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot()
    plt.title(title)
    plt.show()
    cr = classification_report(y_test, y_pred, target_names=target_names)
    print(cr)


    