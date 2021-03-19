# Basic Structure for Cross Validation of ML Model

## Introduction

This is a set of skeleton functions to generally work with any model training 
setup. Should be able to work with any sklearn models. I would advise 
swapping out sklearn models to something else in production, if minmaxing 
speed/memory use, but it is probably fine 99% of the time.

## General Functionality 

*run.py* will take in a training set (and optionally a test set) and run a 
k-fold cross validation to output the best set of hyperparameters. A printout 
of the results will be available. If a test set is provided, the script will
then use the best hyperparameters selected and train on the whole training
set, then return the predicted values as *predictions.csv*.

## How to use

Define the configuration settings in *config.json* and edit the functions in
*cv_utils* as needed (most notably *data_manipulation()*). Then you can run 
*run.py* in command line. 

If you are providing a test set, remember to modify the test set function 
to have an expected output format. It is defaulted to what Kaggle generally 
expects. 

## Configuration Values

The keys in *config.json* is defined here

**model_type**: Define if it is "classification" or "regression"


**model**: Define which ML model will be used. (Options available: "rf" for random forest, "gbm" for gradient boost machine, "lr" for linear/logistic regression (depending on model_type))

