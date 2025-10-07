# main.py
from pyspark.sql.functions import col, concat_ws, regexp_replace, lower, trim, isnan, when, count
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec as MLWord2Vec, VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from replace_missing_spark import load_and_clean_data
from pyspark.sql.functions import isnan, when, count, col, trim
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


# 1) Δημιουργία νέας στήλης text_features που συνδυάζει τις text στήλες
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

#print("Μετά τη δημιουργία της στήλης text_features:")
#researchers_df2.select("text_features").show(2, truncate=False)


tok = RegexTokenizer(inputCol="text_features", outputCol="tokens", pattern="\\W+")
df_step = tok.transform(researchers_df2)
#df_step.select("text_features", "tokens").show(2, truncate=False)


rmv = StopWordsRemover(inputCol="tokens", outputCol="tokens_no_sw")
df_step = rmv.transform(df_step)
#df_step.select("tokens", "tokens_no_sw").show(2, truncate=False)


# Word2Vec -> Μετατροπή tokens σε διάνυσμα (μέσος όρος λέξεων)
w2v = MLWord2Vec(
    inputCol="tokens_no_sw", 
    outputCol="w2v", 
    vectorSize=10,
    minCount=1,
    maxIter=20,
    windowSize=5,
    stepSize=0.025
)
w2v_model = w2v.fit(df_step)
df_w2v = w2v_model.transform(df_step)
#df_w2v.select("tokens_no_sw", "w2v").show(5, truncate=False)


# Συνδυάζουμε embedding + numeric
scaler = StandardScaler(inputCol="w2v", outputCol="scaledFeaturesFaculty", withMean=True, withStd=True)
scaler_model = scaler.fit(df_w2v)
df_w2v = scaler_model.transform(df_w2v)
#df_w2v.select("w2v", "scaledFeaturesFaculty").show(2, truncate=False)

# Computing WSSSE for K values from 2 to 8
# 4) Εύρεση του βέλτιστου αριθμού clusters με Silhouette Score
cost = []
scores = []
evaluator = ClusteringEvaluator(
    predictionCol="prediction",
    featuresCol="scaledFeaturesFaculty",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

for k in range(2, 15):
    kmeans = KMeans(featuresCol="scaledFeaturesFaculty", k=k, seed=42)
    model = kmeans.fit(df_w2v)
    cost.append(model.summary.trainingCost)
    predictions = model.transform(df_w2v)
    score = evaluator.evaluate(predictions)
    scores.append(score)

# --- Elbow ---
plt.figure()
plt.plot(range(2, 15), cost, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WSSSE (Cost)")
plt.title("Elbow Method for Optimal k for Faculty")

# --- Elbow ---
plt.figure()
plt.plot(range(2, 15), scores, marker='o')
plt.xlabel("Number of Clusters (k) for Faculty")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method for Optimal k for Faculty")

plt.show()

#plt.show()
#5. Performing K-means Clustering
# Define the K-means clustering model
kmeans = KMeans(k=7, featuresCol="scaledFeaturesFaculty", predictionCol="clusterFaculty")
kmeans_model = kmeans.fit(df_w2v)

# Assigning the data points to clusters
clusters = kmeans_model.transform(df_w2v)
#Εμφάνιση των ερευνητών και σε ποιο cluster ανήκουν αυτοί οι ερευνητές-Ενδεικτικά 10 γραμμές
clusters.select("Name", "tokens_no_sw", "clusterFaculty").show(10, truncate=False)
#Εμφάνιση των clusters και πόσοι ερευνητές ανήκουν σε κάθε cluster
clusters.groupBy("clusterFaculty").count().show()


'''
##### START-UPS DATASET #####
'''
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
#startups_regex.select("industry", "industry_clean").show(2, truncate=False)

# 3B) Tokenization (σπάμε τη φράση της industry σε λέξεις)
tok = RegexTokenizer(inputCol="industry_clean", outputCol="tokens", pattern=r"\W+")
df_tok = tok.transform(startups_regex)
#df_tok.select("industry_clean", "tokens").show(2, truncate=False)


# (προαιρετικό) Αφαίρεση stopwords για να μείνουν πιο «ουσιαστικές» λέξεις
rem = StopWordsRemover(inputCol="tokens", outputCol="tokens_nostop")
df_tok2 = rem.transform(df_tok)
#df_tok2.select("tokens", "tokens_nostop").show(10, truncate=False)


# 3C) Word2Vec -> Μετατροπή tokens σε διάνυσμα (μέσος όρος λέξεων)
# - vectorSize: μέγεθος embedding (π.χ. 100)
# - minCount=1 για να μη χαθούν σπάνιοι όροι-Εκπαιδεύουμε το word2vec, το σχεδιάζουμε και το εκτελούμε στο αρχείο
w2v = MLWord2Vec(
    inputCol="tokens_nostop",
    outputCol="features",
    vectorSize=10,
    minCount=1,
    seed=42
)
w2v_model = w2v.fit(df_tok2)
df_feats = w2v_model.transform(df_tok2)

#df_feats.select("industry", "tokens_nostop", "features").show(3, truncate=False)


# 3D) StandardScaler πάνω στο διάνυσμα features → scaled_features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
scaler_model = scaler.fit(df_feats)
data_df = scaler_model.transform(df_feats)

#data_df.select("industry", "scaled_features").show(1, truncate=False)


# Computing WSSSE for K values from 2 to 8
# 4) Εύρεση του βέλτιστου αριθμού clusters με Silhouette Score
cost = []
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
    cost.append(model.summary.trainingCost)
    predictions = model.transform(data_df)
    score = evaluator.evaluate(predictions)
    scores.append(score)


# --- Elbow ---
plt.figure()
plt.plot(range(2, 15), cost, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WSSSE (Cost)")
plt.title("Elbow Method for Optimal k")

# --- Elbow ---
plt.figure()
plt.plot(range(2, 15), scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method for Optimal k")

#plt.show()
#5. Performing K-means Clustering
# Define the K-means clustering model
kmeans = KMeans(k=7, featuresCol="scaled_features", predictionCol="clusterIndustry")
kmeans_model = kmeans.fit(data_df)

# Assigning the data points to clusters
clusters = kmeans_model.transform(data_df)
#Εμφάνιση των εταιρειών και σε ποιο cluster ανήκουν αυτες οι εταιρείες-Ενδεικτικά 10 γραμμές
clusters.select("industry", "clusterIndustry").show(20, truncate=False)
#Εμφάνιση των clusters και πόσες εταιρείες ανήκουν σε κάθε cluster
clusters.groupBy("clusterIndustry").count().show()
