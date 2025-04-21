import findspark 
findspark.init()
from pyspark import SparkFiles
from pyspark.sql import SparkSession
import sys
import os

os.environ['HADOOP_HOME'] = "C:\hadoop"
sys.path.append("C:/Mine/Spark/hadoop-2.6.0/bin")

# Create a Spark session
spark = SparkSession.builder.appName("Mastering K-means Clustering with PySpark MLlib").getOrCreate()

# URL of the dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/Iris.csv"
spark.sparkContext.addFile(url)

# Read the dataset
df = spark.read.csv(SparkFiles.get("Iris.csv"), header=True, inferSchema=True)
df.show(5)
