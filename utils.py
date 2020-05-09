from functools import wraps
from time import time
import logging


def timing(func):
    @wraps(func)
    def wrap(*args, **kw):
        ts = time()
        result = func(*args, **kw)
        te = time()
        logging.info('func:%r took: %2.4f sec' %
                     (func.__name__, te - ts))
        return result

    return wrap
