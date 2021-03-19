
with open(config.json,r) as in_:
    config = json.read(in_)


train_df = pd.read_csv(FLAGS.train_df)
if FLAGS.test_df != "":
    config['test_df'] = pd.read_csv(FLAGS.test_df)
else:
    config['test_df'] = ""

cv(config)