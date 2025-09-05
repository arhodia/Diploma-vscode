# Step 1: Data Preprocessing with PySpark
""" import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,lit,concat_ws
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col
import pandas as pd """

# Set SPARK_HOME if running locally
""" os.environ["SPARK_HOME"] = "C:/Users/arhod/spark-3.5.5" """

# Initialize Spark session
""" spark = SparkSession.builder.appName("ResearcherStartupMatching").getOrCreate() """

# Load datasets
#researchers_df = spark.read.option("header", "true").csv("C:/Users/arhod/Desktop/Diploma-vscode/indian_faculty_dataset.csv")

""" startups_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("C:/Users/arhod/Desktop/Diploma-vscode/INC 5000 Companies 2019.csv")
 """
""" researchers_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("C:/Users/arhod/Desktop/Diploma-vscode/indian_faculty_dataset.csv") """

# Εμφάνισε τις στήλες για τους ερευνητες(schema)
""" researchers_df.printSchema() """



""" researchers_df = researchers_df.withColumn(
    "text_features", concat_ws(" | ", col("Position"), col("Department"), col("Expertise"), col("Highest Qualification"))
).withColumn("type", lit("researcher")).withColumnRenamed("Name", "entity_name")
 """

# Keep only specific columns in researchers_df
#columns_to_keep = ["Vidwan-ID", "entity_name", "text_features", "type"]
#researchers_df = researchers_df.select(*columns_to_keep)


# Εμφάνισε τις στήλες για τις εταιρίες(schema)
#startups_df.printSchema()

# Εμφάνισε τις στήλες για τους ερευνητες(schema)
""" researchers_df.printSchema() """

# Εμφάνισε τα 10 πρώτα rows, σαν πίνακα (tabular)
#startups_df.show(5, truncate=False)

# Εμφάνισε τα 5 πρώτα rows, σαν πίνακα (tabular)
""" researchers_df.show(5, truncate=False) """



# Step 2: Data Cleaning
# Check nulls in each column
#null_counts = startups_df.select([sum(col(c).isNull().cast('int')).alias(c) for c in startups_df.columns])

# Show the result
#null_counts.show()
# Print the row with Vidwan-ID = 60818 in researchers_df
""" researchers_df.filter((col("Vidwan-ID") == 60818) | (col("Vidwan-ID") == 556358)).show(truncate=False) """


# main.py
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,lit,concat_ws
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col
from pyparsing import col
from replace_missing_spark import load_and_clean_data
from pyspark.sql.functions import col
from pyspark.sql.functions import isnan, when, count, col, trim
from pyspark.mllib.feature import Word2Vec

first_blank_workers_row,null_nan_counts_startups,null_nan_counts_researchers,startups_df, researchers_df = load_and_clean_data()

# Now you can use them
print("First 'workers' value where blank:")
print(first_blank_workers_row)
print("Null/NaN counts in startups dataset:")
print(null_nan_counts_startups)
print("Null/NaN counts in researchers dataset:")
print(null_nan_counts_researchers)
# Show the schema of the DataFrames
researchers_df.printSchema()
startups_df.printSchema()

# Show the size in MB of each DataFrame
def dataframe_size_mb(df):
    # Estimate size by converting to Pandas and checking memory usage
    pdf = df.limit(1000000).toPandas()  # limit to avoid OOM for very large datasets
    size_mb = pdf.memory_usage(deep=True).sum() / (1024 * 1024)
    return size_mb

print(f"Researchers DataFrame size (MB): {dataframe_size_mb(researchers_df):.2f}")
print(f"Startups DataFrame size (MB): {dataframe_size_mb(startups_df):.2f}")

#confirm 
startups_df.show(5)
researchers_df.show(5)


def count_null_nan_blank(df):
    return df.select([
        count(
            when(
                col(c).isNull() | isnan(col(c)) | (trim(col(c)) == ''), c
            )
        ).alias(c) for c in df.columns
    ])

print("Null/NaN/blank counts in startups dataset:")
count_null_nan_blank(startups_df).show()

print("Null/NaN/blank counts in researchers dataset:")
count_null_nan_blank(researchers_df).show()

researchers_df.filter((col("Vidwan-ID") == 56586) | (col("Vidwan-ID") == 556358)).show(truncate=False) 
startups_df.filter((col("rank") == 4)).show(truncate=False) 




# 1. Παίρνουμε τα text features από Spark
#researchers_pd = researchers_df.select("Vidwan-ID", "Department").toPandas()
#startups_pd = startups_df.select("rank", "industry_features").toPandas()








