# Basic Structure for Cross Validation of ML Model

## Introduction

This is essentially a wrapper functions to run an exhaustive grid search 
k-fold cross validation that should generally work with any model training 
setup. It should be able to work with any sklearn models. I would advise 
swapping out sklearn models to something else in production, if minmaxing 
speed/memory use, but it is probably fine 99% of the time. 

## General Functionality 

Running *run.py* will take in a training set (and optionally a test set) and 
run an exhaustive grid search k-fold cross validation to output the best set 
of hyperparameters. A printout of the results will be available. If a test set 
is provided, the script will then use the best hyperparameters selected and 
train on the whole training set, then return the predicted values 
as *predictions.csv*.

## How to use

Define the configuration settings in *config.json* and edit the functions in
*cv_utils* as needed (most notably *data_manipulation()*). Then you can run 
*run.py* in command line. 

If you are providing a test set, remember to modify the test set function 
to have an expected output format. It is defaulted to what Kaggle generally 
expects. 

## Configuration Values

The keys in *config.json* is defined here

### Within base configs:

**model_type**: Define if it is "classification" or "regression"

**model**: Define which ML model will be used. (Options available: "rf" for 
random forest, "gbm" for gradient boost machine, and **TBD** "lr" for 
linear/logistic regression (depending on model_type))

**k-fold**: Define the number of folds to use in the grid search

**scoring_metric**: Define which scoring metric is used to optimize the 
hyperparameters from. Refer to 
[this](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) 
for options.

**label**: Define the name of the label column (that is being trained against)

**id**: Define the name of the id column. This is used to hide from the 
training set, and also output a prediction with this column if needed 
(e.g. expected submission for Kaggle)

### Within rf configs:

Note: These are just a few of the parameters from sklearn's 
RandomForestClassifier() and RandomForestRegressor(), so any additional 
parameters of interest can be found on their documentation. I just kept the 
standard go-to hyperparameters that generally are tuned.

You can write a string of comma-separated values to test different values for 
each hyperparameter. 

**n_estimators**: The number of trees in the forest. 

**max_depth**: The maximum depth of the tree. Removing from the config will 
use the default value defined by the class

**min_samples_split**: The minimum number of samples required to split an 
internal node:

* If int, then consider min_samples_split as the minimum number.

* If float, then min_samples_split is a fraction and ceil(min_samples_split * 
n_samples) are the minimum number of samples for each split.

**min_samples_leaf**: The minimum number of samples required to be at a leaf 
node. A split point at any depth will only be considered if it leaves at 
least min_samples_leaf training samples in each of the left and right 
branches. This may have the effect of smoothing the model, especially in 
regression.

* If int, then consider min_samples_leaf as the minimum number.

* If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * 
n_samples) are the minimum number of samples for each node.

### Within gbm configs

Note: These are just a few of the parameters from Microsoft's lightGBM class. 
More specifically, there is a sklearn compatible class LGBMClassifier() and 
LGBMRegressor(), so any additional parameters of interest can be found in the 
relevant documentation. 

You can write a string of comma-separated values to test different values for 
each hyperparameter. 

**n_estimators**: Number of boosted trees to fit.

**num_leaves**: Maximum tree leaves for base learners.

**max_depth**: Maximum tree depth for base learners, <=0 means no limit.