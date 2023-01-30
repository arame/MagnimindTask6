import logging, importlib, sys

def set_logging(logging):
    # ensure logging is output in the Jupyter Notebook
    importlib.reload(logging)
    logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                        level=logging.INFO, stream=sys.stdout)