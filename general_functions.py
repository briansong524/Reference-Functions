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

### Timer class for efficiently getting checkpoint times (CHKTIME) ###

'''
Made a timer class that I find useful for grabbing different times at checkpoints of a script.
Wherever this class is called is when the timer begins. 
- my_timer.print_reset(some_str) will return the time between when this is called and when the class was created/when this method was last run.
- my_timer.time_from_init() returns total time since the class was called.
Just stick a bunch of these throughout a list of scripts. This can be easily searched/deleted after being done with this. 
Replace all the prints with logs if you would like to have it logged somewhere. 
'''

import time

class my_timer:
	def __init__(self):
		self.start_time = time.time()
		self.reset_time = time.time()

	def print_reset(self, some_str):
		# some_str to printout note: 'time to (some_str)' i.e. "time to run some_function()"
		
		tot_time = time.time() - self.reset_time

		m, s = divmod(tot_time, 60)
		h, m = divmod(m, 60)
		h, m, s = map(int, (h, m, s))
		print('time to ' + some_str + ': ' + str(h) + ' hour(s) ' + str(m) + ' minute(s) ' + str(s) + ' second(s).')	
		self.reset_time = time.time()

	def time_from_init(self):
		# get the total time from when this class was initiated 

		tot_time = time.time() - self.start_time

		m, s = divmod(tot_time, 60)
		h, m = divmod(m, 60)
		h, m, s = map(int, (h, m, s))
		print('total time since starting time: ' + str(h) + ' hour(s) ' + str(m) + ' minute(s) ' + str(s) + ' second(s).')

## Example of use

# timer = my_timer()
# time.sleep(1)
# timer.print_reset('wait one second later') # returns "time to wait one second later: 0 hour(s) 0 minute(s) 1 second(s)."
# time.sleep(3)
# timer.print_reset('wait three seconds later') # returns "time to wait three seconds later: 0 hour(s) 0 minute(s) 3 second(s)."
# time.sleep(1)
# timer.time_from_init() # returns "total time since starting time: 0 hour(s) 0 minute(s) 5 second(s)."


### class for indexing categorical variables (CATIND) ### 

class categorical_dictionary:
    'class containing all categorical variables indexed and converted'
    
    def __init__(self):
        self.cat_dict = {}
        self.rev_cat_dict = {}
    
    def add_col(self, vals, col_name, verbose = True):
        cat_vals = set(vals) # get classes
        temp_dict = dict(zip(cat_vals, range(1, len(cat_vals)+1)))
        temp_dict[col_name + '_UNK'] = 0 # adding an index for previously non-existant class 
        self.cat_dict[col_name] = temp_dict
        rev_temp_dict = {j:i for i,j in temp_dict.items()}
        self.rev_cat_dict[col_name] = rev_temp_dict
        if verbose:
            print('Added ' + col_name)
            
    def cat_to_ind(self, vals, col_name):
        def failsafe_mapper(val, col_name):
            'make mapping robust by handling previously unseen classes'
            try:
                mapped_val = self.cat_dict[col_name][val]
            except:
                print('Unknown value: "' + str(val) + '", appending as index 0 (general unknown class index)')
                mapped_val = 0
            return mapped_val
        
        mapped_list = list(map(lambda x: failsafe_mapper(x,col_name), vals))
        return mapped_list
    
    def ind_to_cat(self, vals, col_name):
        return list(map(lambda x: self.rev_cat_dict[col_name][x], vals))
    
# Example:
#
# df = read.csv('...') # "cat_col_1","cat_col_2","y"
# x_cols = ['cat_col_1','cat_col_2'] # categorical columns
# X = df[x_cols].copy()
# y = df['y'].values
# cat_dict = categorical_dictionary()
# 
# for col_ in x_cols:
#     cat_dict.add_col(X[col_].values, col_)
#     X[col_] = cat_dict.cat_to_ind(X[col_], col_)
#
# # The dataframe "X" should now be all numeric, which is now ready to be used for sklearn models or something

### check memory usage by Python script (GETMEM) ###
def check_memory():
	process = psutil.Process(os.getpid())
	rss = process.memory_info().rss / 1024.0 / 1024.0 # megabytes
	sfx = 'MB'
	if rss > 1024:
		rss = rss / 1024.0
		sfx = 'GB'
	print("Current memory usage: " + str(round(rss,2)) + " " + sfx)
