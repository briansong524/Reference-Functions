"""
Spark Template

General template for a main() function that will reside in the driver
process of the Spark app. The goal is to keep this compartmentalized for
maximum modularity - whether it is used for a machine learning process or
data manipulation.
"""

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import udf, col



def spark_initiate(master = 'local', appName = 'placeholder'):
    # Initiate a new Spark Session 
    spark = SparkSession.builder \
            .master('local') \
            .appName('placeholder') \
            .getOrCreate() \
            #.config('something','something')
    return spark



def load_data(spark, data, rdd_or_df, structure_schema = ''):
    # Load data from python environment into the Spark Session. Load as RDD
    # if the data is unstructured, and as dataset if structured (unless 
    # you're confident enough to write code more optimal than the built-in
    # optimizations for dataset type)

    if rdd_or_df == 'rdd':
        sc = spark.sparkContext
        rdd = sc.parallelize(data)
        return rdd
    elif rdd_or_df == 'df':
        if structure_schema == '':
            df = spark.createDataFrame(data)
        else:
            df = spark.createDataFrame(data,structure_schema)
        return df

def process_template():
    # An example of a general setup for working with Spark. 
    # Steps: 
    # - Start up a Spark Session
    # - Load data into the session (as RDD or dataset/dataframe)
    # - Execute some process on the data (ML, data preprocess, etc.)


    # Start up a Spark Session 
    
    spark = spark_initiate()
    
    # Load a dataset into the Spark environment (as RDD or dataset/dataframe)

    data = [('a',1,[1,2,3]),('b',2,[4,5,6]),('c',3,[7,8,9])]
    structure_schema = StructType([
                            StructField('col1',StringType(),True),
                            StructField('col2',IntegerType(),True),
                            StructField('col3',StructType([
                                    StructField('val1',IntegerType()),
                                    StructField('val2',IntegerType()),
                                    StructField('val3',IntegerType())
                                    ])
                                )
                            ])
    df = load_data(spark = spark,
                   data = data,
                   rdd_or_df = 'df',
                   structure_schema = structure_schema
                   )

    # Display details of the data (optional, just good practice)

    df.printSchema() # check if data is interpreted correctly
    df.show() # check if data looks correct

    # Row-wise Data Manipulation

    def row_process(df):
        # Row-wise data processing. Easier to comprehend if written as a 
        # mapping function. This is done by writing row-based functions, 
        # then using map() with this. I have yet to determine if this is 
        # more efficient than column based processes 
        # Modify as needed.
        caps = df['col1'].capitalize()
        add5 = df['col2'] + 5
        sum_list = sum(df['col3'])
        return caps, add5, sum_list

    rdd = df.rdd.map(lambda x: row_process(x))
    colnames = ['COL1','col2plus5','col3sum']
    df2 = rdd.toDF(colnames)
    df2.show()

    # Column-wise Data Manipulation

    def col_process(df):
        # Column-wise data processing. I would say this is similar to using 
        # panda's .map() column operation. I am not sure if this is more
        # efficient than row-wise. If anything, it is a bit
        # messier to write, but avoids the necessity to convert dataframe 
        # to RDD to process and convert back into dataframe. Also, 
        # column-wise functions can be reused across different datasets 

        caps_udf = udf(lambda x: x.capitalize(), StringType())
        add5_udf = udf(lambda x: x + 5, IntegerType())
        sum_udf = udf(lambda x: sum(x), IntegerType())

        df = df.withColumn('COL1', caps_udf(col('col1'))) # replaces 'col1'
        df = df.withColumn('col2plus5', add5_udf(col('col2'))) # adds column
        df = df.withColumn('col3sum', sum_udf(col('col3'))) #adds column
        return df

    df2 = col_process(df)
    df2.show()
