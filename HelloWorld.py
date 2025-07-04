# Step 1: Data Preprocessing with PySpark
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col

# Set SPARK_HOME if running locally
os.environ["SPARK_HOME"] = "C:/Users/arhod/spark-3.5.5"

# Initialize Spark session
spark = SparkSession.builder.appName("ResearcherStartupMatching").getOrCreate()

# Load datasets
researchers_df = spark.read.option("header", "true").csv("C:/Users/arhod/Desktop/Diploma-vscode/indian_faculty_dataset.csv")
startups_df = spark.read.option("header", "true").csv("C:/Users/arhod/Desktop/Diploma-vscode/INC 5000 Companies 2019.csv")

# Εμφάνισε τις στήλες για τους ερευνητες(schema)
#researchers_df.printSchema()

# Εμφάνισε τα 10 πρώτα rows, σαν πίνακα (tabular)
#researchers_df.show(10, truncate=False)

startups_df.printSchema()
print("Rows before cleaning:", startups_df.count())
startups_df.show(10, truncate=False)


startups_df = startups_df.select("rank", "profile", "name", "url", "industry")
startups_df = startups_df.filter(
    col("rank").isNotNull() & (col("rank") != "") &
    col("profile").isNotNull() & (col("profile") != "") &
    col("name").isNotNull() & (col("name") != "") &
    col("url").isNotNull() & (col("url") != "") &
    col("industry").isNotNull() & (col("industry") != "")
)


print("Rows after cleaning:", startups_df.count())
startups_df.show(10, truncate=False)

# Save the cleaned startups_df to a new CSV file
output_path = "C:/Users/arhod/Desktop/Diploma-vscode/cleaned_startups.csv"
startups_df.write.mode("overwrite").option("header", "true").csv(output_path)

