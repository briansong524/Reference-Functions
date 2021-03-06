### Printing heat map of confusion matrix (PHMCM) ###
# assume the confusion matrix is stored under "best_confmat"
# also required is total number of bins (here is labeled "total_bins")

import numpy as np
import matplotlib.pyplot as plt
sum_rows = np.sum(best_confmat,axis=1) # this line is summing up the total ground-truth values that were accounted for. 
nonzero_axis = [i for i in range(len(sum_rows)) if sum_rows[i] !=0] # in case some classes were never predicted (maybe very sparse), then this 
																	# makes it not so ugly
scaled_confmat = best_confmat[nonzero_axis] / sum_rows[nonzero_axis,None] # if the distribution of classes were skewed, then the heatmap would 
																		  # be a bit awkward (bigger classes would get most of the color weight)

plt.matshow(scaled_confmat)
plt.colorbar()
tick_marks = np.arange(total_bins)
plt.xticks(tick_marks, range(total_bins))
plt.yticks(tick_marks,nonzero_axis)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()



### Get a complete(-ish) summary of the confusion matrix (COMPCM) ### 

from sklearn.metrics import confusion_matrix
import pandas as pd

'''
Note that AUC, as important as it is, requires predicted probabilities, so run that separately. I could've 
made it work with it, but I'm lazy.
PLease help where it can be improved.
Made a class instead of a method so that each measure can be pulled out.

A realistic situation in which this can be used can look something like this:

lr = LogisticRegression()
lr.fit(X,y)
pred = lr.predict(X)
cms = conf_mat_summary(y_true = y, y_pred = pred)
cms.summary()
predprob = lr.predict_proba(X) 
auc = roc_auc_score(y,predprob[:,1])
print('AUC of training set:' + str(round(auc,4)))

'''

class conf_mat_summary:

    def __init__(self, y_true, y_pred): #, labels = None, sample_weight = None  # i am afraid these might break the code lol.
        self.y_true = list(y_true)
        self.y_pred = list(y_pred)
        self.confusion_matrix = confusion_matrix(y_true, y_pred)#, labels, sample_weight)
        self.tn, self.fp, self.fn, self.tp = list(map(float,self.confusion_matrix.ravel()))

        # Calculate the different measures (added 1e-5 at the denominator to avoid 'divide by 0')

        self.error_rate  = (self.fp + self.fn) / (self.tn + self.fp + self.fn + self.tp + 0.00001)
        self.accuracy    = (self.tp + self.tn) / (self.tn + self.fp + self.fn + self.tp + 0.00001)
        self.sensitivity = self.tp / (self.tp + self.fn + 0.00001)
        self.specificity = self.tn / (self.tn + self.fp + 0.00001)
        self.precision   = self.tp / (self.tp + self.fp + 0.00001)
        self.fpr 		 = 1 - self.specificity
        self.f_score     = (2*self.precision*self.sensitivity) / (self.precision + self.sensitivity  + 0.00001)


    def summary(self):

        # gather values 

        names_ = ['Accuracy','Precision/PPV','Sensitivity/TPR/Recall','Specificity/TNR','Error Rate','False Positive Rate (FPR)','F-Score']
        values = [self.accuracy, self.precision, self.sensitivity, self.specificity, self.error_rate, self.fpr, self.f_score]
        values = list(map(lambda x: round(x,4), values))
        results = pd.DataFrame({'Measure':names_, 'Value':values})


        # calculate some formatting stuff to make output nicer

        set_ = set(self.y_true + self.y_pred)
        labels = sorted(list(map(str, set_)))
        max_len_name = max(list(map(len,list(labels))))
        labels = list(map(lambda x: x + ' '*(max_len_name - len(x)), labels))
        dis_bet_class = max([max_len_name, len(str(self.confusion_matrix[0][0])), len(str(self.confusion_matrix[1][0]))])
        extra_0 = dis_bet_class - len(labels[0])
        extra_1 = dis_bet_class - len(str(self.confusion_matrix[0][0]))
        extra_2 = dis_bet_class - len(str(self.confusion_matrix[1][0]))

        # print outputs

        print(' ') # skips a line. idk, maybe it would look nicer in terminal or something
        print(' '*(6 + max_len_name) + 'pred')
        print(' '*(6 + max_len_name) + labels[0] + ' '*(extra_0 + dis_bet_class) + labels[1])
        print('true ' + labels[0] + ' ' + str(self.confusion_matrix[0][0]) + ' '*(extra_1 + dis_bet_class) + str(self.confusion_matrix[0][1]))
        print('     ' + labels[1] + ' ' + str(self.confusion_matrix[1][0]) + ' '*(extra_2 + dis_bet_class) + str(self.confusion_matrix[1][1]))
        print(results)
	
# ## Showing how to use the class 
# true = [0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,0,1,0,1,0] # 13 0's and 9 1's
# pred = [0,0,0,0,1,0,0,0,1,1,0,0,1,0,1,1,0,0,1,0,1,0] # 14 0's and 8 1's

# ## Even if you use named variables or otherwise
# map_ = {0:'no',1:'yes'}
# true = map(lambda x: map_[x], true)
# pred = map(lambda x: map_[x], pred)

# a = conf_mat_summary(y_true = true, y_pred = pred)
# a.summary() # prints the confusion matrix + measures
# print(' '); print('Accuracy: ' + str(a.accuracy)) # can pull individual measurements by calling the variable

### random forest print feature importance ###

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import seaborn as sns
from pylab import savefig

# X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size = 0.2)
# rf = RandomForestClassifier(n_jobs = 10)
# rf.fit(X_tr, y_tr)
# scores = cross_val_score(rf,X_tr,y_tr,cv=5)
feat_imp = rf.feature_importances_
var_imp = list(zip(list(X_tr),feat_imp))
var_imp.sort(key=lambda tup: tup[1], reverse=True)
max_imp = var_imp[0][1]
rel_imp = list(map(lambda x: x[1]/max_imp, var_imp))

sns.set_style("whitegrid")
sns.set_color_codes("muted")
fig = plt.figure(figsize=(15, 10))
plt.tight_layout()

sig_df = pd.DataFrame({'y':[i[0] for i in var_imp], 'x':[i[1] for i in var_imp]})

ax = sns.barplot( y='y',x='x',data=sig_df)
ax.set_title("Most important predictors in the " + 'Decision Tree' + " model")
ax.set( ylabel='Variables',xlabel='Importance')
counter = 0
for p in ax.patches:
    width = p.get_width()
    ax.text(width  +.007 ,
            p.get_y()+p.get_height()/2. + 0.2,
            '{:1.3f} / {:1.3f}'.format(var_imp[counter][1], rel_imp[counter]),
            ha="center")
    counter +=1
ax.text(np.max(sig_df['x'])-.024,22,'Average Training Accuracy: {:1.3f}'.format(np.mean(scores)), ha='center')
