""" Utility Functions
All the necessary methods that are used to run the grid search k-fold 
cross validation. These methods are compartmentalized to different steps for
easier modifications.
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, confusion_matrix

def cv(inputs):
    # The series of steps involved in a grid search k-fold cross validation
    
    params = inputs['base']
    x_tr, y_tr = data_manipulation(params)
    
    # Setup cv
    K = inputs['k-fold']
    model = all_models(inputs['model_type']['model'])
    param_grid = inputs[params['model']]
    for key in param_grid.keys():
        param_grid[key] = list(map(int,param_grid[key].split(',')))
        
    gscv = GridSearchCV(model, 
                        param_grid = inputs[inputs['model']],
                        scoring = params['scoring_metric'],
                        cv = int(params['k-fold'])
                        )
    gscv.fit(x_tr,y_tr)
    
    ## display results

    print("Best parameters set found on development set:")
    print()
    print(gscv.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = gscv.cv_results_['mean_test_score']
    stds = gscv.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    ## output test set results (if needed)
    print('predicting on best params')

    x_te, _ = data_manipulation(params, test_set = True)
    preds = gscv.predict(x_te)
    test_df['target'] = preds
    output = test_df[['id','target']].copy()
    output.to_csv(data_dir + '/predictions/text_filter_and_keyword_lgb_model.csv', index = False)

def data_manipulation(params, test_set = False):
    # Skeleton of a standard data manipulation method. This is probably where 
    # all the project-specific modifications should take place to maintain 
    # consistency.    
    
    y_name = params['label']
    id_name = params['id']
    if test_set:
        col_names = list(df)
        x_names = list(set(col_names) - y_name - id_name)
        return df[x_names], ''
    else:
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
