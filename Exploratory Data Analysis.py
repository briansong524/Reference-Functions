def check_na(df):
    'return pandas dataframe of description of NAs in a pandas df'
    na_count = df.isna().sum()
    na_df = pd.DataFrame({'column':na_count.index, 'count_na':na_count.values})
    na_df['prop_na'] = na_df['count_na']/df.shape[0]
    na_df = na_df.sort_values('prop_na', ascending = False)
    na_df.reset_index(inplace=True)
    na_df.drop('index', axis = 1, inplace = True)
    return na_df
