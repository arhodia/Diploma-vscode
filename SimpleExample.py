import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import findspark

from pyspark.sql.functions import col, lower, regexp_replace
findspark.init()
from pyspark import SparkFiles
from pyspark.sql import SparkSession
from sklearn.feature_extraction.text import TfidfVectorizer
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.ml.clustering import KMeans
spark = SparkSession.builder \
    .appName("Mastering K-means Clustering with PySpark MLlib") \
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
    .getOrCreate()
urlStartups = "C://Users//arhod//Desktop//Διπλωματικη 24-25//startups.csv"
spark.sparkContext.addFile(urlStartups)

urlResearchers = "C://Users//arhod//Desktop//Διπλωματικη 24-25//researchers.csv"
spark.sparkContext.addFile(urlResearchers)

#data frame for startups
dfStartups = spark.read.csv(SparkFiles.get("startups.csv"), header=True, inferSchema=True,sep=';')
dfStartups.show()

#data frame for researchers
dfResearchers = spark.read.csv(SparkFiles.get("researchers.csv"), header=True, inferSchema=True,sep=';')
dfResearchers.show()

# Preprocessing: Convert to lowercase and remove punctuation
dfStartups = dfStartups.withColumn("Description", lower(col("Description")))
dfStartups = dfStartups.withColumn("Description", regexp_replace(col("Description"), "[^a-zA-Z\\s]", ""))

dfResearchers = dfResearchers.withColumn("Description", lower(col("Description")))
dfResearchers = dfResearchers.withColumn("Description", regexp_replace(col("Description"), "[^a-zA-Z\\s]", ""))

# Show results
dfStartups.show(truncate=False)
dfResearchers.show(truncate=False)

startup_pd = dfStartups.select("Description").toPandas()
researcher_pd = dfResearchers.select("Description").toPandas()

# Combine text columns for startup and researcher
startup_text = startup_pd["Description"].tolist()
researcher_text = researcher_pd["Description"].tolist()

# Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

# Fit and transform the startup descriptions
startup_vectors = vectorizer.fit_transform(startup_text)

# Transform the researcher descriptions using the same vocabulary
researcher_vectors = vectorizer.transform(researcher_text)

startup_df_tfidf = pd.DataFrame(startup_vectors.toarray(), columns=vectorizer.get_feature_names_out())
researcher_df_tfidf = pd.DataFrame(researcher_vectors.toarray(), columns=vectorizer.get_feature_names_out())

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


print("Startup TF-IDF Matrix:")
print(startup_df_tfidf)
print("\nResearcher TF-IDF Matrix:")
print(researcher_df_tfidf)

# Convert the TF-IDF matrices to Spark DataFrames

# Convert startup TF-IDF to Spark DataFrame
startup_spark_df = spark.createDataFrame([
    Row(features=Vectors.dense(row)) for row in startup_df_tfidf.values
])
# Convert researcher TF-IDF to Spark DataFrame
researcher_spark_df = spark.createDataFrame([
    Row(features=Vectors.dense(row)) for row in researcher_df_tfidf.values
])








# Computing WSSSE for K values from 2 to 8
# wssse_values =[]
# evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='scaled_features', \
#                                 metricName='silhouette', distanceMeasure='squaredEuclidean')

# for i in range(2,8):
#     KMeans_mod = KMeans(featuresCol='scaled_features', k=i)
#     KMeans_fit = KMeans_mod.fit(data_df)
#     output = KMeans_fit.transform(data_df)
#     score = evaluator.evaluate(output)
#     wssse_values.append(score)
#     print("Silhouette Score:",score)



# Fill NaN values
# startups = startups.fillna("")
# researchers = researchers.fillna("")

# # Combine text data for vectorization
# startup_text = startups['Description']
# researcher_text = researchers['Research_Area']
# # Vectorize the text
# vectorizer = TfidfVectorizer(stop_words="english")
# startup_vectors = vectorizer.fit_transform(startup_text)
# researcher_vectors = vectorizer.transform(researcher_text)

# # Apply KMeans
# kmeans_startups = KMeans(n_clusters=2, random_state=0).fit(startup_vectors)
# kmeans_researchers = KMeans(n_clusters=2, random_state=0).fit(researcher_vectors)
#
# # Add cluster labels
# startups["Cluster"] = kmeans_startups.labels_
# researchers["Cluster"] = kmeans_researchers.labels_
#
# # Find matches based on clusters
# matches = []
# for _, startup in startups.iterrows():
#     for _, researcher in researchers.iterrows():
#         if startup["Cluster"] == researcher["Cluster"]:
#             matches.append({
#                 "Startup": startup["Name"],
#                 "Researcher": researcher["Name"],
#                 "Cluster": startup["Cluster"]
#             })
#
# # Display matches
# for match in matches:
#     print(match)
