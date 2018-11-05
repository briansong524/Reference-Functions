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
		self.reset_time = self.start_time

	def print_reset(self, some_str):
		# some_str to printout note: 'time to (some_str)' i.e. "time to run some_function()"
		
		tot_time = time.time() - self.reset_time

		if tot_time < 120:
			print('time to ' + some_str + ': ' + str(round(tot_time,2)) + ' seconds.')
		else:
			print('time to ' + some_str + ': ' + str(round(tot_time / 60.0,2)) + ' minutes.')	


		self.reset_time = time.time()

	def time_from_init(self):
		# get the total time from when this class was initiated 

		tot_time = time.time() - self.start_time
		if tot_time < 120:
			print('total time since starting time: ' + str(round(tot_time,2)) + ' seconds.')
		else:
			print('total time since starting time: ' + str(round(tot_time / 60.0,2)) + ' minutes.')


## Example of use

# timer = my_timer()
# time.sleep(1)
# timer.print_reset('wait one second later') # returns "time to wait one second later: 1.0 seconds."
# time.sleep(3)
# timer.print_reset('wait three seconds later') # returns "time to wait three seconds later: 3.0 seconds."
# time.sleep(1)
# timer.time_from_init() # returns "total time since starting time: 5.0 seconds."
