import pyspark
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.types import DoubleType

def model_structure():
    structure_schema = StructType([
                            StructField('PassengerId',IntegerType(),True),
                            StructField('Survived',IntegerType(),True),
                            StructField('Pclass',IntegerType(),True),
                            StructField('Name',StringType(),True),
                            StructField('Sex',StringType(),True),
                            StructField('Age',DoubleType(),True),
                            StructField('SibSp',IntegerType(),True),
                            StructField('Parch',IntegerType(),True),
                            StructField('Ticket',StringType(),True),
                            StructField('Fare',DoubleType(),True),
                            StructField('Cabin',StringType(),True),
                            StructField('Embarked',StringType(),True),
                            ])

    return structure_schema

def model_config():
    conf = {
            "base":{
                "model_type":"classification",
                "model":"gbm",
                "featuresCol":"features",
                "labelCol":"Survived",
                "predictionCol":"prediction",
            },
            "model":{
                "rf":{
                    "numTrees":"20",
                    "maxDepth":"5"
                    },
                "gbm":{
                    "maxDepth":"5",
                    "lossType":"logistic",
                    "stepSize":"0.1"
                    },
                "linear":{
                    "loss":"squaredError",
                    "regParam":"0",
                    "elasticNetParam":"0"
                    },
                "logistic":{
                    "threshold":"0.5",
                    "regParam":"0",
                    "elasticNetParam":"0"
                    }
                }
            }
    return conf