""" 
Utility Functions

All the necessary methods that are used to run the grid search k-fold 
cross validation. These methods are compartmentalized to different steps for
easier modifications.
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GridSearchCV

def cv(inputs):
    # The series of steps involved in a grid search k-fold cross validation
    
    params = inputs['base']
    x_tr, y_tr = data_manipulation(params)
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
    cv_results(gscv)
    
    if params['test_df'] != "":
        print('predicting on best params')

        x_te, _ = data_manipulation(params, test_set = True)
        preds = gscv.predict(x_te)
        test_df[params['label']] = preds
        output = test_df[[params['id'],params['label']]].copy()
        output.to_csv(params['output_dir'] + '/cv_predictions.csv',
                      index = False
                      )
    
def all_models(model_type, model):
    # Defining which model is being called here. Can easily modify/add more 
    # models here.

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

def cv_results(gscv, params):
    # Printing out the cv results and write to file if configured as such

    means = gscv.cv_results_['mean_test_score']
    stds = gscv.cv_results_['std_test_score']
    results = 'Best parameters set found on development set:\n\n'
    results += str(gscv.best_params_) + '\n\n'
    results += 'Grid scores on development set:\n\n'
    for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
        results += "%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params) + '\n'
    print(results)
    if params['save_results']:
        with open(params['output_dir'] + '/cv_results.txt','w') as out_:
            out_.write(results)

def data_manipulation(params, test_set = False):
    # Skeleton of a standard data manipulation method. This is probably where 
    # all the project-specific modifications should take place to maintain 
    # consistency.    
    


    ## Separating predictors and label, removing id column, if any

    y_name = params['label']
    id_name = params['id']

    if test_set:
        df = params['test_df']
        y = ''
    else:
        df = params['train_df']
        y = df[y_name]

    ## Write any data aggregation/modification here ##

    pass

    ## Formatting expected output

    col_names = list(df)
    x_names = list(set(col_names) - y_name - id_name)
    return df[x_names], y