"""
Spark ML Template

General template for using Apache's MLlib API with compartmentalized steps 
to optimize modularity for each project.



"""

import pyspark
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation, ChiSquareTest

def run_process():
    # Cookie cutter sequence of processes involved in running the 
    # necessary steps

    spark = SparkSession.builder \
            .master('local') \
            .appName('placeholder') \
            .getOrCreate()


    
def all_models():
    # All models to choose amongst for simple regression/classification
    pass


def data_manipulation():
    # Skeleton of a standard data manipulation method. This is probably where 
    # all the project-specific modifications should take place to maintain 
    # consistency.    
    
    pass


def display_results():
    # Display an overview of the model output to verify the process has run 
    # as expected.
    pass


