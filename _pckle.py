import pickle, os
import logging

def save_pickle_object(data, filename):
    # open a file where data need to be stored
    path = os.path.join("pickle", filename)
    logging.info(f"Saving pickle file from: {path}")
    with open(path, 'wb') as file:
        # dump information to the file
        pickle.dump(data, file)

def load_pickle_object(filename):
    # open a file where data need to be stored
    path = os.path.join("pickle", filename)
    if os.path.exists(path) == False:
        logging.info(f"Pickle file in: {path}")
        return None

    logging.info(f"Loading pickle file from: {path}")
    with open(path, 'rb') as file:
        # dump information to the file
        data = pickle.load(file)
        return data