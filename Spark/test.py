#! /usr/bin/env python
import os
import json
import datetime
import argparse

import pandas as pd


def main(config):
    # Cookie cutter sequence of processes involved in running the 
    # necessary steps. Using the general pipeline outlined in Spark's
    # MLLib docs here: https://spark.apache.org/docs/latest/ml-pipeline.html

    spark = spark_initiate()

    # some data / tramsformer
    raw_data = config['base']['train_df']
    data = load_data(spark, raw_data, 'df')
    df = transformer(data)

    # estimator

    model = estimators(model=config['model'], 
                       model_type=config['model_type']
                       )
    fitted_model = model.fit(data)

def spark_initiate(master = 'local', appName = 'placeholder'):
    # Initiate a new Spark Session 
    spark = SparkSession.builder \
            .master('local') \
            .appName('placeholder') \
            .getOrCreate() \
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

def transformer(data):
    # Skeleton of a standard data manipulation method. This is probably where 
    # all the project-specific modifications should take place to maintain 
    # consistency.    
    
    pass

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

    with open(FLAGS.config,'r') as in_:
        config = json.load(in_)

    config['base']['train_df'] = pd.read_csv(FLAGS.train_df)

    today_dt = str(datetime.datetime.today()).split('.')[0].replace(' ','_')
    output_dir = FLAGS.output_dir + '_' + today_dt
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config['base']['output_dir'] = output_dir 
    config['base']['save_results'] = FLAGS.save_results
    main(config)

