### generate an embedding layer that can be trained on (emblayer) ###

def embed(inputs, size, dim,name):
    # inputs is a list of indices
    # size is the number of unique indices (look for max index to achieve this if ordered)
    # dim is the number of embedded numbers 
    std = np.sqrt(2 / dim)
    emb = tf.Variable(tf.random_uniform([size, dim], -std, std))
    lookup = tf.nn.embedding_lookup(emb, inputs,name = name)
    #print(lookup.shape)
    return lookup
    # input_x_cat1 = input_x[:,(input_name_len + input_itemdesc_len)] # just a row of categories. can be multiple indices
    # cat1_emb = embed([i for i in range(dict_cat1_len)],dict_cat1_len,cat1_emb_size, name= 'cat1_emb') 
    # cat1_emb_lookup = tf.nn.embedding_lookup(cat1_emb,input_x_cat1)
    # cat1_emb_lookup is the matrix that can be used as input for NN now




### General Neural Network (DNN) ###

def dense_NN(x,out_len, name_w, name_b):

    tot_nodes = x.shape[1]
    W_dense = tf.Variable(tf.truncated_normal([int(tot_nodes) , out_len], stddev=0.1), name=name_w) #"W2"
    b_dense = tf.Variable(tf.constant(0.1, shape=[out_len]), name=name_b) # "b2"
    return tf.matmul(x,W_dense) + b_dense




### Dropout layer (dropNN) ###

def dropout_layer(layer, dropout_keep_prob):
    return tf.nn.dropout(layer, dropout_keep_prob)




### ReLU layer (relu) ###

def relu_layer(layer):
    return tf.nn.relu(layer)





### ResNet Simple Model (resnet) ###

def residual_block(layer, out_len):
    shortcut = layer
    
    layer_ = dense_NN(layer, out_len, 'res_w', 'res_b')
    layer_ = relu_layer(layer_)
    layer_ = dense_NN(layer_, out_len, 'res_w2', 'res_b2')
    layer_ = relu_layer(layer_)
    
    shortcut = dense_NN(shortcut, out_len, 'shortcut_w','shortcut_b')
    
    added_ = tf.add(layer_, shortcut)
    added_ = relu_layer(added_)
    return added_
    




### Convolutional Neural Network (CNN) ###

def CNN(x,W_shape,b_shape,pad_length, name_w, name_b):
    # x is the expanded lookup tables that will be trained
    W1 = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1), name= name_w) #"W1"
    b1 = tf.Variable(tf.constant(0.1, shape=[b_shape]), name = name_b) # "b1"
    conv = tf.nn.conv2d( #tf.layers.conv2d is also used, with  more parameters. Probably a slightly higher API because of that.
        x,
        W1,
        strides = [1,1,1,1],
        padding="VALID",
        name="conv")
    #print('shape of CNN output:' + str(conv.shape))
    h = tf.nn.relu(tf.nn.bias_add(conv, b1), name="relu")
    #print('shape after ReLU: ' + str(h.shape))
    pooled = tf.nn.max_pool(
                h,
                ksize=[1, pad_length, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
    #print('shape after max pooling: ' + str(pooled.shape))
    pool_flat = tf.reshape(pooled, [-1, out_nodes])
    #print("shape after flattening:" + str(pool_flat.shape))

    #h_drop = tf.nn.dropout(pool_flat, dropout_keep_prob)
    #print('shape after dropout: ' + str(h_drop.shape))
    return pool_flat
    # name_emb_lookup_expand = tf.expand_dims(name_emb_lookup,-1)
    # W_shape_name = [1,name_emb_size,1,out_nodes] 
	# b_shape_name = out_nodes # same as last dimension in W
	# layers_name = CNN(name_emb_lookup_expand,W_shape_name,b_shape_name,name_pad_size,"W_name", "b_name")



