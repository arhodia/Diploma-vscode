"""
print("\nclean_combined columns:")
print(clean_combined.columns)

first_5 = clean_combined.drop("scaled_features").limit(5)
last_5 = clean_combined.drop("scaled_features").orderBy(F.monotonically_increasing_id(), ascending=False).limit(5)

row_count = clean_combined.count()
last_5 = clean_combined.drop("scaled_features").withColumn("row_idx", F.monotonically_increasing_id()).orderBy(F.desc("row_idx")).limit(5)


print("First 5:")
first_5.show(truncate=False)
print("Last 5:")
last_5.show(truncate=False)

print("\nclusters_all 5 rows:")
clusters_all.drop("scaled_features").show(5, truncate=False)
   # Λήψη όλων των μοναδικών τιμών ως λίστα
    unique_values = clusters_all.select("topic_merged_norm").rdd.flatMap(lambda x: x).distinct().collect()

    #print("Πλήθος:", number_of_unique_values)
    #print("Τιμές:", unique_values)

last_clean_combined_5 = clusters_all.drop("scaled_features").orderBy(F.monotonically_increasing_id(), ascending=False).limit(10)

row_count_clusters_all = clusters_all.count()
last_clean_combined_5 = clusters_all.drop("scaled_features").withColumn("row_idx", F.monotonically_increasing_id()).orderBy(F.desc("row_idx")).limit(20)
print("Last 20:")
last_clean_combined_5.show(truncate=False)

print("\nfeats_all columns:")
print(feats_all.columns)

print("\ntrain_df columns:")
print(train_df.columns)

print("\nclusters_all columns:")
print(clusters_all.columns)


# Εμφανίζουμε μερικές εγγραφές για επαλήθευση
    combined.filter(col("id") == 123).show(truncate=False)
    combined.filter(col("company_rank") == 1).show(truncate=False)
    combined.filter(col("id") == 17.0).show(truncate=False)
    combined.filter(col("id") == 4999).show(truncate=False)
    combined.filter(col("company_rank") == 587).show(truncate=False)
    combined.filter(col("company_rank") == 789).show(truncate=False)
    """

    #Ελέγχω τα χαρατηριστικα των 2 dataframes
    #r2.printSchema()
    #s2.printSchema()

    #r2.count()
    #s2.count()

    #combined.printSchema() 
    #combined.toPandas().to_csv(final_csv, index=False, encoding="utf-8-sig")
    # Εκτυπώνω πληροφορίες για το αρχείο
    # print(f"Saved → {final_csv}")
    # rows = combined.count()
    # print(rows)
    # combined.printSchema()

    # Κειμενικές στήλες που έχουμε στο ενιαίο DF
    #text_cols = [c for c in ["researchField", "industry", "city"] if c in combined.columns]

    #λαμβάνω υπόψη όλες τις μορφές που μπορεί να πάρει μια κενή τιμή
    # Εμφάνισε τις πρώτες 10 εγγραφές των αποτελεσμάτων
    #final_results.show(10, truncate=False)
    #final_results.show(5, truncate=False)
    #final_results.show(final_results.count(), truncate=False)


    #print(clusters_all.select("topic_merged_norm").rdd.flatMap(lambda x: x).collect())
    #print(selected_option)
#clean_combined
#"""