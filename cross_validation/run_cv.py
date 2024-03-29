"""
Run Grid-Search Cross Validation

Script to run an exhaustive grid search cross validation on a given training 
set. Can output predictions if test_df is defined.
"""

import os
import json
import datetime
import argparse

import pandas as pd
from cv_utils import cv

parser = argparse.ArgumentParser()

parser.add_argument(
    '--config', type=str, default='config.json',
    help = 'Configuration file')
parser.add_argument(
    '--train_df', type=str, default='',
    help = 'Input training file (whole directory if not in same \
            location as script)')
parser.add_argument(
    '--test_df', type=str, default='',
    help = 'Input testing file (whole directory if not in same \
            location as script). Blank would skip outputting predictions.')
parser.add_argument(
    '--output_dir', type=str, default='outputs',
    help = 'Output directory for cv results')
parser.add_argument(
    '--save_results', type=bool, default=True,
    help = 'Save cv results to output directory')

FLAGS, unparsed = parser.parse_known_args()

if __name__ == '__main__':
    with open(FLAGS.config,'r') as in_:
        config = json.load(in_)

    config['base']['train_df'] = pd.read_csv(FLAGS.train_df)
    if FLAGS.test_df != "":
        config['base']['test_df'] = pd.read_csv(FLAGS.test_df)
        config['base']['run_pred'] = True
    else:
        config['base']['run_pred'] = False

    today_dt = str(datetime.datetime.today()).split('.')[0].replace(' ','_')
    output_dir = FLAGS.output_dir + '_' + today_dt
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config['base']['output_dir'] = output_dir 
    config['base']['save_results'] = FLAGS.save_results
    cv(config)