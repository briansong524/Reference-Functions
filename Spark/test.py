#! /usr/bin/env python

"""
A Simple test using Kaggle's Titanic data (specifications stored in 
model_utils.py). Goes through the whole pipeline of loading data to
evaluating the model.
"""
import os
import json
import datetime
import argparse

import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, Imputer
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from model_utils import model_structure, model_config

def main(config):
    # Cookie cutter sequence of processes involved in running the 
    # necessary steps. Using the general pipeline outlined in Spark's
    # MLLib docs here: https://spark.apache.org/docs/latest/ml-pipeline.html

    spark = spark_initiate()

    # some data / tramsformer
    raw_data = config['base']['train_df']
    structure_schema = model_structure()
    data = load_data(spark, raw_data, 'df', structure_schema)
    # data.show()

    df, cat_dict = transformer(data)
    datatype_dict = dict(df.dtypes)
    features = config['base']['featuresCol'].split(',')
    list_str = [] # list of string columns
    for feature in features:
        if datatype_dict[feature] == 'string':
            list_str.append(feature)
            df = StringIndexer(inputCol=feature, 
                               outputCol=feature + '_index'
                               ) \
                 .fit(df) \
                 .transform(df)
    df = df.drop(*list_str)
    df.show()
    features = list(set(df.columns) - set(config['base']['labelCol']))
    assembler = VectorAssembler(inputCols=features,
                                outputCol='features')
    df = assembler.transform(df)
    (trainingData, testData) = df.randomSplit([0.7, 0.3])

    # estimator

    model = estimators(config)
    fitted_model = model.fit(trainingData)
    testData = fitted_model.transform(testData)
    predictionAndLabels = testData.select('probability','Survived') \
                                  .rdd.map(lambda x: (float(x[0][0]),
                                                      float(x[1])
                                                      )
                                          )
    metrics = BinaryClassificationMetrics(predictionAndLabels)

    # Area under precision-recall curve
    print("Area under PR = %s" % metrics.areaUnderPR)

    # Area under ROC curve
    print("Area under ROC = %s" % metrics.areaUnderROC)


def spark_initiate(master = 'local', appName = 'placeholder'):
    # Initiate a new Spark Session 
    spark = SparkSession.builder \
            .master('local') \
            .appName('placeholder') \
            .getOrCreate() 
            #.config('something','something')
    return spark


def load_data(spark, data, rdd_or_df, structure_schema = ''):
    # Load data from python environment into the Spark Session. Load as RDD
    # if the data is unstructured, and as dataset if structured (unless 
    # you're confident enough to write code more optimal than the built-in
    # optimizations for dataset type)

    if rdd_or_df == 'rdd':
        sc = spark.sparkContext
        rdd = sc.parallelize(data)
        return rdd
    elif rdd_or_df == 'df':
        if structure_schema == '':
            df = spark.createDataFrame(data)
        else:
            df = spark.createDataFrame(data,structure_schema)
        return df

def transformer(df, cat_dict={}):
    # Skeleton of a standard data manipulation method. This is probably where 
    # all the project-specific modifications should take place to maintain 
    # consistency.    
    # Using https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/
    # to do some basic feature engineering 
    
    # Drop unnecessary columns 

    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    if cat_dict == {}:
        cat_dict = categorical_dictionary()

    # df = df.withColumn('Title', 
    #                    df.Cabin.rdd.flatMap(
    #                                lambda x: 
    #                                     substrings_in_string(x,
    #                                                       title_list
    #                                                       )
    #                                )            
    #                    )
    substring_udf_title = udf(lambda x: substrings_in_string(x, title_list), 
                              StringType()
                              )
    replace_title_udf = udf(lambda x: replace_titles(x), StringType())
    substring_udf_cabin = udf(lambda x: substrings_in_string(x, cabin_list), 
                              StringType()
                              )

    df = Imputer().setInputCols(['Age']) \
                  .setOutputCols(['Age']) \
                  .setStrategy('mean') \
                  .fit(df).transform(df)

    df = df.withColumn('Title', 
                      substring_udf_title(col('Name'))  
                      ) \
           .withColumn('Title', 
                      replace_title_udf(col('Title'))
                      ) \
           .withColumn('Deck', 
                      substring_udf_cabin(col('Cabin'))
                      ) \
           .withColumn('Family_Size', col('SibSp') + col('Parch')) \
           .withColumn('Age_Class', col('Age') * col('Pclass')) \
           .withColumn('Fare_Per_Person', col('Fare') / (col('Family_Size') + 1)) \
           .withColumn('Sex',col('Sex')) \
           .drop('PassengerId','Name','Ticket','Cabin')


    # if 'Sex' not in cat_dict.keys():substring_udf = udf(lambda x: substrings_in_string(x, title_list), StringType())
    #     # replace with index, save indexer
    #     pass
    # if 'Embarked':
    #     # replace with index, save indexer
    #     pass

    return df, cat_dict

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if substring in big_string:
            return substring
    # print big_string
    return np.nan

def replace_titles(title):
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    else:
        return title

def convert_word_to_ind(dataset_col,dictionary): 
    # input the pandas column of texts and dictionary. This should be modular
    # each input should be a string of cleaned words tokenized into a list 
    # (ex. ['this', 'is', 'an', 'item'])
    # dictionary should be the dictionary obtained from build_dictionary
    list_of_lists = []
    unk_count = 0 # total 'unknown' words counted
    for word_or_words in dataset_col: # words is the list of all words
        list_of_inds = []
        for word in word_or_words:
            if word in dictionary:
                index = np.int(dictionary[word]) # dictionary contains top words, so if in, it gets an index
            else:
                index = 0  #  or dictionary['UNK']? can figure out later
                unk_count += 1
            list_of_inds.append(index)
        list_of_lists.append(list_of_inds)

    # make list_of_lists into something that can be put into pd.DataFrame
    #list_as_series = pd.Series(list_of_lists)
    list_as_series = np.array(list_of_lists)
    return list_as_series, unk_count

def estimators(config):
    # All models to choose amongst for simple regression/classification
    model_type = config['base']['model_type']    
    model = config['base']['model']

    if model == 'gbm':
        if model_type == 'classification':
            glm = GBTClassifier(
                        featuresCol = 'features',
                        labelCol = config['base']['labelCol'],
                        predictionCol = config['base']['predictionCol'],
                        lossType = config['model']['gbm']['lossType'],
                        maxDepth = int(config['model']['gbm']['maxDepth']),
                        stepSize = float(config['model']['gbm']['stepSize'])
                        )


    return glm


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
        self.fpr         = 1 - self.specificity
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
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config', type=str, default='ml.conf',
        help = 'Configuration file')
    parser.add_argument(
        '--train_df', type=str, default='',
        help = 'Input training file (whole directory if not in same \
                location as script)')
    parser.add_argument(
        '--output_dir', type=str, default='outputs',
        help = 'Output directory for results')
    parser.add_argument(
        '--save_results', type=bool, default=True,
        help = 'Save results to output directory')

    FLAGS, unparsed = parser.parse_known_args()

    # with open(FLAGS.config,'r') as in_:
    #     config = json.load(in_)
    config = model_config()

    config['base']['train_df'] = pd.read_csv(FLAGS.train_df)

    today_dt = str(datetime.datetime.today()).split('.')[0].replace(' ','_')
    output_dir = FLAGS.output_dir + '_' + today_dt
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config['base']['output_dir'] = output_dir 
    config['base']['save_results'] = FLAGS.save_results
    main(config)

