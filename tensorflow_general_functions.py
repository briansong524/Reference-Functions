### making an iterator to feed a tensorflow graph (tensoriter) ###

batch_len = 1000
num_epoch = 25
tot_iter = train_raw.shape[0]* num_epoch // batch_len + 1 # might not use this 


print('splitting labels and features...')
features_input = train_temp_set.astype(np.int32)
label_input = train_price
# make some placeholders to avoid GraphDef exceeding 2GB
feat_placeholder = tf.placeholder(features_input.dtype, features_input.shape)
label_placeholder = tf.placeholder(label_input.dtype, label_input.shape)
print('making tensor slices...')
dataset = tf.data.Dataset.from_tensor_slices((feat_placeholder, label_placeholder))
print('shuffling...')
#np.random.shuffle(temp_set) # shuffle the data
dataset = dataset.shuffle(buffer_size =10000)
print('making epochs...')
dataset = dataset.repeat() # epoch
print('making batches...')
dataset = dataset.batch(batch_len) 
iterator = dataset.make_initializable_iterator()
next_batch = iterator.get_next()
sess.run(iterator.initializer, {feat_placeholder: features_input, label_placeholder: label_input})
    while counter <= 1001:
        features_, label_ = sess.run(next_batch) #etcetc




### training method to prevent overfitting (trainnoover) ###

print('Start training...')

start_time = time.time()

learn_rate = 1e-4
counter = 0 
i = 1
best_acc = 0 # just has to be greater than 2 really
with tf.Session() as sess:
    sess.run(iterator.initializer, {feat_placeholder: features_input, label_placeholder: label_input})
    init = tf.global_variables_initializer()
    sess.run(init)  
    while counter <= 1001:
        features_, label_ = sess.run(next_batch)
        sess.run(train_step,{input_x: features_, input_y: label_, dropout_keep_prob:.7})
        if i % 100 == 0:
            print('calculating accuracy')
            acc_val, pred_val = sess.run([accuracy,argmax_pred], {input_x: val_temp_set, input_y: val_price, dropout_keep_prob:1})
            print('Accuracy of validation set at step %1d: %5.3f' % (i, acc_val))
            if acc_val > best_acc:
                best_acc = acc_val
                saver.save(sess,save_path=export_dir)
                best_confmat = confusion_matrix(pred_val, np.argmax(val_price,1))
                print('resetting counter after ' + str(counter) + ' steps')
                counter = 0
                
            tot_time = time.time() - start_time
            print('One hundred steps took %5.3f seconds.' % tot_time)
            print(' ')
            start_time = time.time()
        counter += 1
        i += 1 
        if i % 500 == 0: 
            learn_rate = learn_rate/10
print('Best validation accuracy is: ' + str(best_acc))
print('Done')