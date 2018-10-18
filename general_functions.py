### Specifying Graphics Card (SGC)###

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # this might be just for nvidia

### Decorator for printing time (DPT) ###

def my_timer(function_):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = function_(*args,**kwargs)
        time_taken = time.time() - start_time
        print('{}() ran in: {} seconds.'.format(function_.__name__, time_taken))
        return result
    return wrapper
