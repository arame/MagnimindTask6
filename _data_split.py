from sklearn import train_test_split

def split_data(X, y):
    X_temp, y_temp, X_test, y_test = train_test_split(X, y, test_size=0.05, stratify=y)
    X_train, y_train, X_val, y_val = train_test_split(X_temp, y_temp, test_size=0.20, stratify=y_temp)
    return X_train, y_train, X_val, y_val, X_test, y_test