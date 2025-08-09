# Step 1: Data Preprocessing with PySpark
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,lit,concat_ws
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col
import pandas as pd
from pyspark.sql.functions import col, mean as _mean, count as _count, when

# Set SPARK_HOME if running locally
os.environ["SPARK_HOME"] = "C:/Users/arhod/spark-3.5.5"

# Initialize Spark session
spark = SparkSession.builder.appName("ResearcherStartupMatching").getOrCreate()

# Load datasets
#researchers_df = spark.read.option("header", "true").csv("C:/Users/arhod/Desktop/Diploma-vscode/indian_faculty_dataset.csv")

startups_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("C:/Users/arhod/Desktop/Diploma-vscode/INC 5000 Companies 2019.csv")

researchers_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("C:/Users/arhod/Desktop/Diploma-vscode/indian_faculty_dataset.csv")



def replace_missing_spark(df, columns, method="mean"):
    for column in columns:
        # Check for null or NaN values
        null_count = df.filter(col(column).isNull() | (col(column) != col(column))).count()
        if null_count > 0:
            if method == "mean":
                mean_value = df.select(_mean(col(column))).first()[0]
                if mean_value is not None:
                    df = df.withColumn(
                        column,
                        when(col(column).isNull() | (col(column) != col(column)), mean_value).otherwise(col(column))
                    )
            elif method == "count":
                count_value = df.count()
                df = df.withColumn(
                    column,
                    when(col(column).isNull() | (col(column) != col(column)), count_value).otherwise(col(column))
                )
    # Add more methods if needed
    return df