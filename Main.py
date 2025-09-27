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

import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, concat_ws, regexp_replace, lower, trim, isnan, when, count
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec as MLWord2Vec, VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from replace_missing_spark import load_and_clean_data
from pyspark.sql.functions import isnan, when, count, col, trim
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt

first_blank_workers_row,null_nan_counts_startups,null_nan_counts_researchers,startups_df, researchers_df = load_and_clean_data()

# Now you can use them - 
#print("First 'workers' value where blank:")
#print(first_blank_workers_row)
#print("Null/NaN counts in startups dataset:")
#print(null_nan_counts_startups)
#print("Null/NaN counts in researchers dataset:")
#print(null_nan_counts_researchers)

# Show the schema of the DataFrames - Εκτυπώνει 
#researchers_df.printSchema()
#startups_df.printSchema()

# Show the size in MB of each DataFrame
#def dataframe_size_mb(df):
    # Estimate size by converting to Pandas and checking memory usage
#    pdf = df.limit(1000000).toPandas()  # limit to avoid OOM for very large datasets
#    size_mb = pdf.memory_usage(deep=True).sum() / (1024 * 1024)
#    return size_mb

#print(f"Researchers DataFrame size (MB): {dataframe_size_mb(researchers_df):.2f}")
#print(f"Startups DataFrame size (MB): {dataframe_size_mb(startups_df):.2f}")

#confirm 
#startups_df.show(5)
#researchers_df.show(5)


def count_null_nan_blank(df):
    return df.select([
        count(
            when(
                col(c).isNull() | isnan(col(c)) | (trim(col(c)) == ''), c
            )
        ).alias(c) for c in df.columns
    ])

#Καλώ τη συνάρτηση για να εκτυπώσω και να επιβεβαιώσω ότι δεν υπάρχουν null/nan/blank τιμές και στα 2 αρχεία
#print("Null/NaN/blank counts in startups dataset:")
#count_null_nan_blank(startups_df).show()

#print("Null/NaN/blank counts in researchers dataset:")
#count_null_nan_blank(researchers_df).show()


#Εμφανίζει μια γραμμή με συγκεκριμένο Vidwan-ID για να επαληθεύσουμε ότι η null/nan τιμή αντικαταστάθηκε με unkown
#researchers_df.filter((col("Vidwan-ID") == 56586) | (col("Vidwan-ID") == 556358)).show(truncate=False)
#startups_df.filter((col("rank") == 4)).show(truncate=False)


# 1) Φτιάχνουμε ενιαίο κείμενο από τις στήλες σου
text_cols = ["Position", "Department", "Expertise", "Highest Qualification"]
researchers_df2 = researchers_df.withColumn(
    "text_features",
    concat_ws(" | ", *[trim(col(c)).cast("string") for c in text_cols])
)


# Προαιρετικό μικρό καθάρισμα για tokenization
researchers_df2 = researchers_df2.withColumn(
    "text_features",
    regexp_replace(lower(col("text_features")), r"\s+", " ")
)

# 2) Pipelines για Word2Vec → Assembler → Scaler → KMeans
def build_word2vec_kmeans_pipeline(k=6, vector_size=200, min_count=2):
    tok = RegexTokenizer(inputCol="text_features", outputCol="tokens", pattern="\\W+")
    rmv = StopWordsRemover(inputCol="tokens", outputCol="tokens_no_sw")
    w2v = MLWord2Vec(inputCol="tokens_no_sw", outputCol="w2v", vectorSize=vector_size,
                   minCount=min_count, maxIter=20, windowSize=5, stepSize=0.025)

    df_step = tok.transform(researchers_df2)
    df_step = rmv.transform(df_step)
    w2v_model = w2v.fit(df_step)
    df_w2v = w2v_model.transform(df_step)
    df_w2v.select("tokens_no_sw", "w2v").show(20, truncate=False)
    # Συνδυάζουμε embedding + numeric
    assembler = VectorAssembler(inputCols=["w2v"] , outputCol="features", handleInvalid="keep")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
    kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="prediction", k=k, seed=42)

    return Pipeline(stages=[tok, rmv, w2v, assembler, scaler, kmeans])

# 3) Helper: τρέχεις διάφορα k και παίρνεις Silhouette + WSSSE (trainingCost)
def try_k_values_word2vec(df, k_range=range(2, 13), vector_size=200, min_count=2):
    results = []  # κάθε στοιχείο: (k, silhouette, wssse, model)
    evaluator = ClusteringEvaluator(featuresCol="scaledFeatures")  # Silhouette (euclidean)
    for k in k_range:
        pipe = build_word2vec_kmeans_pipeline(k=k, vector_size=vector_size, min_count=min_count)
        model = pipe.fit(df)
        pred = model.transform(df)
        sil = float(evaluator.evaluate(pred))
        wssse = float(model.stages[-1].summary.trainingCost)  # “inertia”/WSSSE
        results.append((k, sil, wssse, model))
        print(f"k={k:2d} | silhouette={sil:.4f} | WSSSE={wssse:.2f}")

    # === Plots ===
    ks      = [r[0] for r in results]
    sils    = [r[1] for r in results]
    wsss    = [r[2] for r in results]
"""
    plt.figure()
    plt.plot(ks, sils, marker='o')
    plt.xlabel("k"); plt.ylabel("Silhouette"); plt.title("Silhouette vs k"); plt.grid(True)

    plt.figure()
    plt.plot(ks, wsss, marker='o')
    plt.xlabel("k"); plt.ylabel("WSSSE (trainingCost)"); plt.title("Elbow (WSSSE) vs k"); plt.grid(True)

    return results"""

# === Παράδειγμα χρήσης ===
"""results_w2v = try_k_values_word2vec(researchers_df2, k_range=[6])
for k, sil, wssse, _ in results_w2v:
    print(f"[Word2Vec] k={k}: silhouette={sil:.4f}, WSSSE={wssse:.2f}")


# Πάρε το καλύτερο μοντέλο (με max silhouette)
best_w2v = max(results_w2v, key=lambda t: t[1])[3]
clusters_w2v = best_w2v.transform(researchers_df2)
clusters_w2v.select("Vidwan-ID", "Name", "prediction").show(100, truncate=False)"""





"""
# Πάρε το καλύτερο μοντέλο (με max silhouette)
best_w2v = max(results_w2v, key=lambda t: t[1])[3]
clusters_w2v = best_w2v.transform(researchers_df2)
clusters_w2v.select("Vidwan-ID", "Name", "prediction").show(10, truncate=False)

"""




#4. Finding the Optimal Number of Clusters (K)
"""
# Elbow Method
wssse = []
for k in range(2, 11):
    kmeans = KMeans(k, seed=1)
    model = kmeans.fit(researchers_df)
    wssse.append(model.computeCost(researchers_df))

# Plotting the Elbow Curve
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(range(2, 11), wssse, marker='o')
plt.title("Elbow Method For Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Within Set Sum of Squared Errors (WSSSE)")
plt.grid()
plt.show()


# 5. Performing K-means Clustering

# Define the K-means clustering model
kmeans = KMeans(k=4, featuresCol="scaled_features", predictionCol="cluster")
kmeans_model = kmeans.fit(researchers_df)
# Assigning the data points to clusters
clustered_data = kmeans_model.transform(researchers_df)

# 6. Evaluating the Model
kmeans = KMeans(k=model, seed=1) 

# 7. Visualizing the Results
# Converting to Pandas DataFrame
clustered_data_pd = clustered_data.toPandas()
# Visualizing the results
plt.scatter(clustered_data_pd["SepalLengthCm"], clustered_data_pd["SepalWidthCm"], c=clustered_data_pd["cluster"], cmap='viridis')
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.title("K-means Clustering with PySpark MLlib")
plt.colorbar().set_label("Cluster")
plt.show()

"""



# 3A) Καθαρισμός κειμένου (μικρά γράμματα, αφαίρεση συμβόλων, trims) + αντικατάσταση null
startups_regex = (
    startups_df.withColumn(
        "industry_clean",
        trim(
            lower(
                regexp_replace(col("industry"), r"[^A-Za-z0-9\s]+", " ")
            )
        )
    )
)

# Προβολή πριν το tokenization
startups_regex.select("industry", "industry_clean").show(2, truncate=False)

# 3B) Tokenization (σπάμε τη φράση της industry σε λέξεις)
tok = RegexTokenizer(inputCol="industry_clean", outputCol="tokens", pattern=r"\W+")
df_tok = tok.transform(startups_regex)

df_tok.select("industry_clean", "tokens").show(2, truncate=False)

# (προαιρετικό) Αφαίρεση stopwords για να μείνουν πιο «ουσιαστικές» λέξεις
rem = StopWordsRemover(inputCol="tokens", outputCol="tokens_nostop")
df_tok2 = rem.transform(df_tok)

df_tok2.select("tokens", "tokens_nostop").show(10, truncate=False)

# 3C) Word2Vec -> Μετατροπή tokens σε διάνυσμα (μέσος όρος λέξεων)
# - vectorSize: μέγεθος embedding (π.χ. 100)
# - minCount=1 για να μη χαθούν σπάνιοι όροι
w2v = MLWord2Vec(
    inputCol="tokens_nostop",
    outputCol="features",
    vectorSize=10,
    minCount=1,
    seed=42
)
w2v_model = w2v.fit(df_tok2)
df_feats = w2v_model.transform(df_tok2)

df_feats.select("industry", "tokens_nostop", "features").show(3, truncate=False)


# 3D) StandardScaler πάνω στο διάνυσμα features → scaled_features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
scaler_model = scaler.fit(df_feats)
data_df = scaler_model.transform(df_feats)

data_df.select("industry", "scaled_features").show(1, truncate=False)




'''
# Computing WSSSE for K values from 2 to 8
# 4) Εύρεση του βέλτιστου αριθμού clusters με Silhouette Score
cost = []
for k in range(2, 15):
    kmeans = KMeans(featuresCol="scaled_features", k=k, seed=42)
    model = kmeans.fit(data_df)
    cost.append(model.summary.trainingCost)

plt.plot(range(2, 15), cost, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WSSSE (Cost)")
plt.title("Elbow Method for Optimal k")
plt.show()


scores = []
evaluator = ClusteringEvaluator(
    predictionCol="prediction",
    featuresCol="scaled_features",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

for k in range(2, 15):
    kmeans = KMeans(featuresCol="scaled_features", k=k, seed=42)
    model = kmeans.fit(data_df)
    predictions = model.transform(data_df)
    score = evaluator.evaluate(predictions)
    scores.append(score)

plt.plot(range(2, 15), scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method for Optimal k")
plt.show()'''