
### make one hot vector for response (onehot) ###
onehot_pricebins = np.zeros((train_raw.shape[0],np.max(train_raw.price_bins.values)))
onehot_pricebins[np.arange(train_raw.shape[0]),train_raw.price_bins.values-1] = 1

### make pandas columns from categorical variables (mkcatcol) ###
'''
- For data with multiple rows referring to same index (such as different items purchased by one user),
  get the count and ratio of any categorical columns (such as item type).
- This assumes all categories from the dataset
  - Please preprocess columns if necessary (like grouping less frequent categories)
  - This means if a very infrequent class is not existent in the dataset, then it will not get a column
    even though it technically exists as 0 count
- NA's will be classified as "missing" and counted accordingly
'''
def make_col_from_cat(df, index_col, col, name_split = ' '):
    'for a categorical column, get count/ratio of each category'
    ## check na's and replace with "missing"
    if df[col].isna().sum() > 0:
        print('found some NAs for ' + col + ', replacing with "missing"')
        df[col].fillna('missing',inplace = True)
        
    count_cats = pd.crosstab(df[index_col],df[col]) # get counts of each category per index
    count_names = list(map(lambda x: col + name_split + x, list(count_cats))) # add base column name to all columns 
    count_cats = pd.DataFrame(count_cats.values, index = count_cats.index, columns = count_names)
    ratio_cats = count_cats.divide(count_cats.sum(axis = 1), axis = 0) # divide row values by row total
    ratio_names = list(map(lambda x: x + name_split + 'Ratio', list(ratio_cats))) # rename columns
    ratio_cats = pd.DataFrame(ratio_cats.values, index = ratio_cats.index, columns = ratio_names)
    return pd.concat([count_cats, ratio_cats], axis = 1)
    
def append_pd_cols(df, df_index_col, append_df):
    'append by df_index_col and index of append_df robustly'
    col_names = list(append_df)
    for i in col_names:
        dict_ = dict(zip(append_df.index, append_df[i]))
        df[i] = df[df_index_col].map(dict_)
    return df

def mult_cat_col(df_append_to, df_cat_from, df_append_to_index_col, df_cat_from_index_col, cat_col_list):
    for cat_col in cat_col_list:
        new_cols = make_col_from_cat(df_cat_from, df_cat_from_index_col, cat_col)
        df_append_to = append_pd_cols(df_append_to, df_append_to_index_col, new_cols)
    return df_append_to
  
### make pandas columns from numerical variables (mknumcol) ###
'''
- Get the max/min/mean/sum of a numeric column from data containing multiple rows per index
- Currently set up to only impute with mean (can be adjusted simply)
- Use "append_pd_cols" defined above (within section mkcatcol) to append values to main dataframe used for training
'''

def make_col_from_num(df, index_col, col, name_split = ' '):
    '''
    for numeric column, get max/min/mean/sum. 
    '''
    
    # check/handle NA's
    has_NAs = df[col].isna().sum() > 0
    if has_NAs:
        print('Found NA values in ' + col + '. Replacing NA values with mean: ' + str(np.mean(df[col])))
        df[col].fillna(np.mean(df[col]), inplace = True)
    
    # calculate descriptive statistics
    vals = ['max','min','mean','sum']
    piv = df.pivot_table(index = "Account ID", values = col, aggfunc = [np.max, np.min, np.mean, np.sum])
    col_names = list(map(lambda x: col + name_split + x, vals))
    piv = pd.DataFrame(piv.values, index = piv.index, columns = col_names)
    return piv
  
 
