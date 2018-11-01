from sklearn.metrics import confusion_matrix
import pandas as pd

class conf_mat_summary:

	def __init__(self, y_true, y_pred, labels = None, sample_weight = None):
		self.y_true = list(y_true)
		self.y_pred = list(y_pred)
		self.confusion_matrix = confusion_matrix(y_true, y_pred, labels, sample_weight)
		self.tn, self.fp, self.fn, self.tp = map(float,self.confusion_matrix.ravel())


		self.error_rate  = (self.fp + self.fn) / (self.tn + self.fp + self.fn + self.tp + 0.00001)
		self.accuracy    = (self.tp + self.tn) / (self.tn + self.fp + self.fn + self.tp + 0.00001)
		self.sensitivity = self.tp / (self.tp + self.tn + 0.00001)
		self.specificity = self.tn / (self.tn + self.fp + 0.00001)
		self.precision   = self.tp / (self.tp + self.fp + 0.00001)
		self.fpr 		 = 1 - self.specificity
		self.f_score     = (2*self.precision*self.sensitivity) / (self.precision + self.sensitivity  + 0.00001)


	def summary(self):
		names_ = ['Accuracy','Sensitivity/TPR/Recall','Specificity/TNR','Precision/PPV','Error Rate','False Positive Rate','F-Score']
		values = [self.accuracy, self.sensitivity, self.specificity, self.precision, self.error_rate, self.fpr, self.f_score]
		values = map(lambda x: round(x,2), values)
		results = pd.DataFrame({'Measure':names_, 'Value':values})
		set_ = set(self.y_true + self.y_pred)
		labels = map(str, set_)
		max_len_name = max(map(len,list(labels)))
		labels = map(lambda x: x + ' '*(max_len_name - len(x)), labels)


		dis_bet_class = max([max_len_name, len(str(self.confusion_matrix[0][0])), len(str(self.confusion_matrix[1][0]))])
		extra_1 = dis_bet_class - len(str(self.confusion_matrix[0][0]))
		extra_2 = dis_bet_class - len(str(self.confusion_matrix[1][0]))

		print(' '*(6 + max_len_name) + 'pred')
		print(' '*(6 + max_len_name) + labels[0] + ' '*dis_bet_class + labels[1])
		print('true ' + labels[0] + ' ' + str(self.confusion_matrix[0][0]) + ' '*(extra_1 + dis_bet_class) + str(self.confusion_matrix[0][1]))
		print('     ' + labels[1] + ' ' + str(self.confusion_matrix[1][0]) + ' '*(extra_2 + dis_bet_class) + str(self.confusion_matrix[1][1]))
		print(results)

## Showing how to use the class 
# true = [0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,0,1,0,1,0]
# pred = [0,0,0,0,1,0,0,0,1,1,0,0,1,0,1,1,0,0,1,0,1,0]

## Even if you use named values or otherwise
# map_ = {0:'no',1:'yes'}
# true = map(lambda x: map_[x], true)
# pred = map(lambda x: map_[x], pred)

# a = conf_mat_summary(y_true = true, y_pred = pred)
# a.summary() # this should print out the values 
## Note that AUC, as important as it is, requires predicted probabilities, so run that separately. I could've 
## made it work with it, but I'm lazy.
