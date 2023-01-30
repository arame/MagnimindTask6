from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, matthews_corrcoef
import matplotlib.pyplot as plt

def display_metrics(y_test, y_pred, title = None):
    target_names = ['Class 1', 'Class 2']
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot()
    plt.title(title)
    plt.show()
    cr = classification_report(y_test, y_pred, target_names=target_names)
    print(cr)
    mc = matthews_corrcoef(y_test, y_pred)
    print(f"Taking into account that there are many more non-buiness entries than there are business entries")
    print(f"the Matthews Correlation Coeficient is calculated.")
    print(f"A value of +1 represents perfect prediction, 0 an average random prediction and -1 an inverse predication.")
    print(f"The Matthews Correlation Coefficient in this case is {round(mc, 2)}.")

    