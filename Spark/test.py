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
import strings

import pandas as pd
import pyspark
from pyspark.sql import SparkSession

from model_utils import model_structure, model_config

def main(config):
    # Cookie cutter sequence of processes involved in running the 
    # necessary steps. Using the general pipeline outlined in Spark's
    # MLLib docs here: https://spark.apache.org/docs/latest/ml-pipeline.html

    spark = spark_initiate()

    # some data / tramsformer
    raw_data = config['base']['train_df']
    data = load_data(spark, raw_data, 'df')

    df, cat_dict = transformer(data)
    df.show()
    (trainingData, testData) = df.randomSplit([0.7, 0.3])

    # estimator

    model = estimators(model=config['model'], 
                       model_type=config['model_type']
                       )
    fitted_model = model.fit(df)

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

    df = df.withColumn('Title', 
                       df.Cabin.map(
                                   lambda x: substrings_in_string(x,
                                                                  title_list
                                                                  )
                                   )
                       )
    df = df.withColumn('Title', df.Title.map(replace_titles))
    df = df.withColumn('Deck', 
                       df.Cabin.map(
                                   lambda x: substrings_in_string(x,
                                                                  cabin_list
                                                                  )
                                   )
                       )
    df = df.withColumn('Family_Size',df.SibSp + df.Parch)
    df = df.withColumn('Age_Class', df.Age * df.Pclass)
    df = df.withColumn('Fare_Per_Person', df.Fare / (df.Family_Size + 1))
    df = df.withColumn('Sex',df.Sex)

    if 'Sex' not in cat_dict.keys():
        # replace with index, save indexer

    if 'Embarked':
        # replace with index, save indexer

    df = df.drop('PassengerId','Name','Ticket','Cabin')

    return df

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    print big_string
    return np.nan

def replace_titles(title):
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
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

