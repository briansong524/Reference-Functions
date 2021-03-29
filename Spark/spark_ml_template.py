"""
Spark ML Template

General template for using Apache's MLlib API with compartmentalized steps 
to optimize modularity for each project.

"""

import pyspark
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation, ChiSquareTest

