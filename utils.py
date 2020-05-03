from functools import wraps
from time import time


def timing(func):
    @wraps(func)
    def wrap(*args, **kw):
        ts = time()
        result = func(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
              (func.__name__, te - ts))
        return result

    return wrap
