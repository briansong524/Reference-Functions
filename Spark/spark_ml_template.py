"""
Spark ML Template

General template for using Apache's MLlib API with compartmentalized steps 
to optimize modularity for each project. To keep it consistent, the naming
schemes here will be done with regard to Apache Spark's.



"""

import pyspark
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation, ChiSquareTest
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import LinearRegression


from spark_template import spark_initiate



def transformer(data):
    # Skeleton of a standard data manipulation method. This is probably where 
    # all the project-specific modifications should take place to maintain 
    # consistency.    
    
    
    return 

def estimators(config):
    # All models to choose amongst for simple regression/classification
    model_type = config['base']['model_type']    
    model = config['base']['model']
    if model == 'rf':
        if model_type == 'classification':
            glm = RandomForestClassifier(
                        featuresCol = config['base']['featuresCol'],
                        labelCol = config['base']['labelCol'],
                        predictionCol = config['base']['predictionCol'],
                        numTrees = config['model']['numTrees'],
                        maxDepth = config['model']['maxDepth']
                        )
        elif model_type == 'regression':
            glm = RandomForestRegressor(
                        featuresCol = config['base']['featuresCol'],
                        labelCol = config['base']['labelCol'],
                        predictionCol = config['base']['predictionCol'],
                        numTrees = config['model']['numTrees'],
                        maxDepth = config['model']['maxDepth']
                        )
    if model == 'gbm':
        if model_type == 'classification':
            glm = GBTClassifier(
                        featuresCol = config['base']['featuresCol'],
                        labelCol = config['base']['labelCol'],
                        predictionCol = config['base']['predictionCol'],
                        lossType = config['model']['lossType'],
                        maxDepth = config['model']['maxDepth'],
                        stepSize = config['model']['stepSize']
                        )
        elif model_type == 'regression':
            glm = GBTRegressor(
                        featuresCol = config['base']['featuresCol'],
                        labelCol = config['base']['labelCol'],
                        predictionCol = config['base']['predictionCol'],
                        lossType = config['model']['lossType'],
                        maxDepth = config['model']['maxDepth'],
                        stepSize = config['model']['stepSize']
                        )
    if model == 'logistic':
        glm = LogisticRegression(
                    featuresCol = config['base']['featuresCol'],
                    labelCol = config['base']['labelCol'],
                    predictionCol = config['base']['predictionCol'],
                    threshold = config['model']['threshold'],
                    regParam = config['model']['regParam'],
                    elasticNetParam = config['model']['elasticNetParam']
                    )
    if model == 'linear':
        glm = LinearRegression(
                    featuresCol = config['base']['featuresCol'],
                    labelCol = config['base']['labelCol'],
                    predictionCol = config['base']['predictionCol'],
                    regParam = config['model']['regParam'],
                    elasticNetParam = config['model']['elasticNetParam']
                    )
    return glm

def display_results():
    # Display an overview of the model output to verify the process has run 
    # as expected.

    if config['base']['model_type']

    pass


