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



### 