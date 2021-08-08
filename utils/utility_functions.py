import time


def timeit(method):
    '''
    Decorator to measure the tiem of executing the function. Will be used to check how fast prediction of one image is done
    :param method: function on which it will be used
    :return: time in ms
    '''
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed