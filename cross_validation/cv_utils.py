""" Utility Functions
All the necessary methods that are used to run the grid search k-fold 
cross validation. These methods are compartmentalized to different steps for
easier modifications.
"""


def cv(inputs):
    # The series of steps involved in a grid search k-fold cross validation
    
    param_dict = inputs['base']
    train_df = param_dict['train_df']
    x_tr, y_tr = data_manipulation(train_df)
    
    # Setup cv
    K = inputs['k-fold']
    model = all_models(inputs['model_type']['model'])
    gscv = GridSearchCV(model, 
                        param_grid=inputs[inputs['model']],
                        scoring=inputs['']
                        )

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
    
def all_models(model_type, model):
    # defining which model is being called here
    model_dict = {
        'classification':{
            'rf':RandomForestClassifier(),
            'gbm':LGBMClassifier()
        },
        'regression':{
            'rf':RandomForestRegressor(),
            'gbm':LGBMRegressor()
        }

    }
    return model_dict[model_type][model]