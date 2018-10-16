
### make one hot vector for response (onehot) ###
onehot_pricebins = np.zeros((train_raw.shape[0],np.max(train_raw.price_bins.values)))
onehot_pricebins[np.arange(train_raw.shape[0]),train_raw.price_bins.values-1] = 1