import findspark
from matplotlib import pyplot as plt
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler

findspark.init()
from pyspark import SparkFiles
from pyspark.sql import SparkSession


spark = SparkSession.builder \
    .appName("Mastering K-means Clustering with PySpark MLlib") \
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
    .getOrCreate()
url = "C://Users//arhod//Desktop//Διπλωματικη 24-25//aug_test.csv"
spark.sparkContext.addFile(url)
print(spark.version)

#data frame for job-seekers
df = spark.read.csv(SparkFiles.get("aug_test.csv"), header=True, inferSchema=True)
df.show()
spark.sparkContext.setLogLevel("ERROR")
# Assembling features into a single column
assembler = VectorAssembler(inputCols=["enrollee_id","city_development_index" ], outputCol="features")
data_df = assembler.transform(df)

# Scaling the features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(data_df)
data_df = scaler_model.transform(data_df)

data_df.show(5)

# Computing WSSSE for K values from 2 to 8
wssse_values =[]
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='scaled_features', \
                                metricName='silhouette', distanceMeasure='squaredEuclidean')

for i in range(2,8):
    KMeans_mod = KMeans(featuresCol='scaled_features', k=i)
    KMeans_fit = KMeans_mod.fit(data_df)
    output = KMeans_fit.transform(data_df)
    score = evaluator.evaluate(output)
    wssse_values.append(score)
    print("Silhouette Score:",score)

plt.plot(range(1, 7), wssse_values)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within Set Sum of Squared Errors (WSSSE)')
plt.title('Elbow Method for Optimal K')
plt.grid()
plt.show()

#data frame for companies
url = "C://Users//arhod//Desktop//Διπλωματικη 24-25//INC 5000 Companies 2019.csv"
spark.sparkContext.addFile(url)

df = spark.read.csv(SparkFiles.get("INC 5000 Companies 2019.csv"), header=True, inferSchema=True)
df.show()
spark.sparkContext.setLogLevel("ERROR")
