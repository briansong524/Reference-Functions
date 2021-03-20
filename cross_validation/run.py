import pandas as pd
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    '-i', type=str, default='',
    help = 'input file (whole directory if not in same location as script)')
FLAGS, unparsed = parser.parse_known_args()

if __name__ == '__main__':
    with open(config.json,r) as in_:
        config = json.read(in_)


    config['train_df'] = pd.read_csv(FLAGS.train_df)
    if FLAGS.test_df != "":
        config['test_df'] = pd.read_csv(FLAGS.test_df)
    else:
        config['test_df'] = ""

    cv(config)