"""
Spark ML Template

General template for using Apache's MLlib API with compartmentalized steps 
to optimize modularity for each project. To keep it consistent, the naming
schemes here will be done with regard to Apache Spark's.



"""

import pyspark
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation, ChiSquareTest

from spark_template import spark_initiate

def run_process():
    # Cookie cutter sequence of processes involved in running the 
    # necessary steps. Using the general pipeline outlined in Spark's
    # MLLib docs here: https://spark.apache.org/docs/latest/ml-pipeline.html

    spark = spark_initiate()



    
def estimators():
    # All models to choose amongst for simple regression/classification
    pass

def transformer():
    # Skeleton of a standard data manipulation method. This is probably where 
    # all the project-specific modifications should take place to maintain 
    # consistency.    
    
    pass


def display_results():
    # Display an overview of the model output to verify the process has run 
    # as expected.
    pass


