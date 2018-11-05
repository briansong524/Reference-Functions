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



timer = my_timer()
time.sleep(1)
timer.print_reset('wait one second later')
time.sleep(3)
timer.print_reset('wait three seconds later')
time.sleep(1)
timer.time_from_init()