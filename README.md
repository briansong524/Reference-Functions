# Reference-Functions
Some functions that can be used for certain use cases

## General Functions I might like to (re)-use for projects ##

Basic skeletons and explanation of each function provided here.

Functions will be stored accordingly to its functionality:

 - General functions
 
 - Tensorflow specific general functions
 
 - Exploratory Analysis
 
 - General Preprocessing
 
 - Text Preprocessing
 
 - Feature Engineering 
 
 - Machine Learning (not deep learning)
 
 - Deep Learning
 
 - Post-model Analysis
 
 - more to come (maybe)...



## General functions ##

 - Specifying Graphics Card (SGC)
 
   - self explanatory really
   
 - Decorator for printing time (DPT)
   
   - a decorator function to print out time. just write @my_timer on the line above any method/class you want time taken to run 
     to be printed

## Tensorflow specific general functions ##

 - making an iterator to feed a tensorflow graph (tensoriter)
 
   - using placeholders to make it super space efficent 

 - training method to prevent overfitting (trainnoover)
 
   - general outline of continuously training until the validation set 
   
     stops getting more accurate / decreasing loss. 
     
   - saves best model
   
   - stop criteria is if the next n batches doesnt improve predictions

## Exploratory Analysis ##


## General Preprocessing ##
 - make one hot vector for response (onehot)
 
   - self explanatory

## Text Preprocessing ## 

 - Splitting text into tuple (split_tuple)
 
   - in case I want to split a column of multiple categories into 
     separate categories

 - Fill NaN values with a placeholder (fillnan)
 
   - self explanatory
   
 - Obtain vocab size by percentile (vocsizefind)
 
   - counts how many times words appear, then sum the most frequent words
     until the total count adds up to n% of the number of words appeared 

 - Building dictionary for a set of words (build_dict)
 
   - build a dictionary based on the n most common words that appear
   
   - the dictionary will have an index (key) that represents the word (value)
   
   - The reverse of the dictionary is also returned

 - Cleaning and tokenizing a list of string (cleantok)
 
   - a list of strings (like item description) is cleaned then tokenized
   
   - can include patterns and words here for filtering
   
   - uses RegexpTokenizer
 
 - Convert list of words to a list of lists of indices (word2ind)
 
   - words will be converted to indices based on a provided dictionary

 - deciding pad lengths (findpadlen)
 
   - use histogram or percentile to choose a potentially good pad length

 - Convert a list of words to a padded sequence of indices (word2padind)
 
   - words will be converted to indices and padded so all lines are 
     equally shaped
     
   - the resulting vector will be a numpy array

 - Generate batches to train a bag-of-words context model (batchbow) 
 
   - use to train words to have contextual understanding
   
   - bag of words part can be converted to sequential

## Feature Engineering ##



## Machine Learning ##



## Deep Learning ##

 - generate an embedding layer that can be trained on (emblayer)
 
   - used to make categorical variables usable in Wx+b for tensorflow
   
   - [i for i in range(vocabulary_size)] or something to define 
     indices for first input parameter

 - General Neural Network (DNN)
 
   - Basic neural network layer

 - Dropout layer (dropNN)
 
   - dropout layer to prevent overfitting

 - ReLU layer (relu)
 
   - Rectified linear unit used to prevent diminishing gradient

 - ResNet Simple Model (resnet)
 
   - helps deal with diminishing gradient for 50+ layer models

 - Convolutional Neural Network (CNN)
 
   - input 3d/4d x to get a 2d output 
   
   - uses tf.nn.conv2d. can use tf.layers.conv2d and avoid having to 
     expand the input


## Post-model Analysis ##

 - Printing heat map of confusion matrix (PHMCM)
 
   - nice visual showing how accurate the classification model is doing


##########################
## Links I found useful ##
##########################

https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/04_Save_Restore.ipynb

 - goes over using tensorflow to do CNN with many comments and explanations
 
 - has helper functions that help with visualization (like heat map)
 
 - includes method of preventing overfitting

https://blog.waya.ai/deep-residual-learning-9610bb62c355

 - basic resnet explanation and setup

http://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/ for reference

 - goes over simple contextual word learning through RNN using tensorflow

https://ahmedbesbes.com/how-to-mine-newsfeed-data-and-extract-interactive-insights-in-python.html
