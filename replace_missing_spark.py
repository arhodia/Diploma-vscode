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
from pyspark.sql.types import NumericType
from pyspark.sql.functions import isnan
from pyspark.sql.types import IntegerType
# Initialize Spark session
spark = SparkSession.builder.appName("ResearcherStartupMatching").getOrCreate()

# Load datasets
# researchers_df = spark.read.option("header", "true").csv("C:/Users/arhod/Desktop/Diploma-vscode/indian_faculty_dataset.csv")
# Count the number of rows for each dataset
researchers_count = spark.read.option("header", "true").csv("C:/Users/arhod/Desktop/Diploma-vscode/indian_faculty_dataset.csv").count()
startups_count = spark.read.option("header", "true").csv("C:/Users/arhod/Desktop/Diploma-vscode/INC 5000 Companies 2019.csv").count()


print(f"Number of rows in researchers dataset: {researchers_count}")
print(f"Number of rows in startups dataset: {startups_count}") 


startups_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("C:/Users/arhod/Desktop/Diploma-vscode/INC 5000 Companies 2019.csv")

researchers_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("C:/Users/arhod/Desktop/Diploma-vscode/indian_faculty_dataset.csv")

# count how many nan, null, and ' ' values are in the dataframe
def count_null_nan(df):
    null_nan_counts = {}
    for column in df.columns:
        count = df.filter(
            col(column).isNull() | (col(column) != col(column)) | (col(column) == ' ')
        ).count()
        null_nan_counts[column] = count
    return null_nan_counts

# call the count_null_nan function and calculate null and nan values
null_nan_counts_startups = count_null_nan(startups_df)
null_nan_counts_researchers = count_null_nan(researchers_df)
print("Startups null/nan counts:", null_nan_counts_startups)
print("Researchers null/nan counts:", null_nan_counts_researchers)



print("Researchers dataframe columns are:")
researchers_df.printSchema()
print("Startups dataframe columns are:")
startups_df.printSchema()


# Μετατροπή των δύο στηλών από string (με μορφή 3.0, 4.0, κλπ.) σε integer

researchers_df_new = (
    researchers_df
    .withColumn("Start Year", col("Start Year").cast(IntegerType()))
    .withColumn("Years of Experience", col("Years of Experience").cast(IntegerType()))
)


# replace the nan and null values with a parameter mean
#def replace_missing_spark(df, columns, method="mean"):
#    for column in columns:
        # Check for null, NaN, or space (' ') values
#        null_count = df.filter(
#            col(column).isNull() | (col(column) != col(column)) | (col(column) == ' ')
#       ).count()
#        if null_count > 0:
#            if method == "mean":
#                mean_value = df.select(_mean(col(column))).first()[0]
#                if mean_value is not None:
#                    df = df.withColumn(
#                        column,
#                        when(
#                            col(column).isNull() | (col(column) != col(column)) | (col(column) == ' '),
#                           mean_value
#                        ).otherwise(col(column))
#                    )
#            elif method == "count":
#                count_value = df.count()
#                df = df.withColumn(
#                    column,
#                    when(
#                        col(column).isNull() | (col(column) != col(column)) | (col(column) == ' '),
#                        count_value
#                    ).otherwise(col(column))
#                )
    # Add more methods if needed
#    return df


def replace_missing_spark(df, columns, numeric_fill="max", string_fill="Unknown"):
    for column in columns:
        # Detect type
        if isinstance(df.schema[column].dataType, NumericType):
            if numeric_fill == "max":
                max_value = df.select(col(column)).agg({"{}".format(column): "max"}).first()[0]
                if max_value is not None:
                    df = df.withColumn(
                        column,
                        when(
                            col(column).isNull() | isnan(col(column)) | (col(column) == ' '),
                            max_value
                        ).otherwise(col(column))
                    )
            else:
                df = df.fillna({column: numeric_fill})
        else:
            # String column → replace null, NaN, empty
            df = df.withColumn(
                column,
                when(
                    col(column).isNull() | (col(column) == '') | isnan(col(column)),
                    string_fill
                ).otherwise(col(column))
            )
    return df


# insert columns to replace 
columns_to_replace_startups = ["workers", "metro"]
startups_df = replace_missing_spark(startups_df, columns_to_replace_startups, numeric_fill="max", string_fill="Unknown")

columns_to_replace_researchers = ["Department", "Location", "Expertise", "Experience", "Qualification", "Honours and Awards", "Start Year", "Years of Experience"]
researchers_df_new = replace_missing_spark(researchers_df_new, columns_to_replace_researchers, numeric_fill="max", string_fill="Unknown")


# check if there is null values in the new dataframes
null_nan_counts_startups = count_null_nan(startups_df)
null_nan_counts_researchers = count_null_nan(researchers_df_new)
print("Startups null/nan counts:", null_nan_counts_startups)
print("Researchers null/nan counts:", null_nan_counts_researchers) 


# Print the row from researchers_df where 'vidwan-id' equals 56586
researchers_df_new.filter(col("vidwan-id") == 56586).show()

print("Researchers dataframe columns are:")
researchers_df_new.printSchema() 