def cv():

    ## data manipulation

    ## setup train data
    # define 'label' as y, everything else as x

    train_df

    ## setup cross validation parameters

    ## display results

    ## output test set results (if needed)



def data_manipulation(df):
    # Skeleton of a standard data manipulation method. This is probably where 
    # all the project-specific modifications should take place to maintain 
    # consistency.    
    
    y_name = 'label'
    col_names = list(df)
    x_names = list(set(col_names) - y_name)

    return df[x_names], df[y_name]

def setup_cv(config):
    # Set up the cross-validation by using the configs defined in config.json

    model = all_models(config['model_type'], config['model'])
    
def all_models(model_type, model):
    # defining which model is being called here
