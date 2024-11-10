from pyspark.sql import SparkSession
import os
os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.11"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.11"

from pyspark.sql.functions import col, when, lower, count, avg, desc, to_date, substring, datediff, current_date, length, min, sum

# Initialize Spark session
spark = SparkSession.builder.appName("App Review Analysis").getOrCreate()

# Load the data
reviews_df = spark.read.option("header", "true").csv("/Users/shooqalsu/Desktop/6457824/raw/reviews.csv") \
    .withColumn("rating", col("rating").cast("double")) \
    .withColumn("review_date", to_date(substring(col("at"), 0, 10), "yyyy-MM-dd"))

print("# Question 1: Feature Analysis")
reviews_with_features = reviews_df.withColumn("feature",
    when(lower(col("content")).contains(" سريع "), " سريع ")
    .when(lower(col("content")).contains(" بطارية "), " بطارية ")
    .when(lower(col("content")).contains(" سهل "), " سهل ")
    .when(lower(col("content")).contains(" جميل "), " جميل ")
    .when(lower(col("content")).contains(" ممتع "), " ممتع ")
    .when(lower(col("content")).contains(" تحديث "), " تحديث ")
    .otherwise(" أخرى ")
)

features_with_ratings = reviews_with_features \
    .groupBy("feature") \
    .agg(count("*").alias("mentions"), avg("rating").alias("avg_rating")) \
    .dropna()  # Drop rows with null values in the result

features_with_ratings.orderBy(desc("mentions")).show()

print("# Question 2: Average Ratings by Category")
# Note: You'll need to adjust the file paths for each category
categories = ["art_and_design", "tools", "travel_and_local", "video_players", "weather"]

for category in categories:
    df = spark.read.option("header", "true").csv(f"/Users/shooqalsu/Desktop/6457824/raw/categories/{category}.csv") \
        .withColumn("score", col("score").cast("double"))
    avg_rating_df = df.agg(avg("score").alias("average_rating")).dropna()  # Drop rows with null values in the result
    print(f"Average rating for {category}:")
    avg_rating_df.show()

print("# Question 3: Rating Evolution")
rating_evolution = reviews_df.groupBy("review_created_version", "review_date") \
    .agg(avg("rating").alias("avg_rating")) \
    .dropna()  # Drop rows with null values in the result

sorted_evolution = rating_evolution.orderBy("review_date")
sorted_evolution.show()

print("# Question 4: App Engagement Analysis")
engagement_df = reviews_df.withColumn("has_reply", when(col("reply_content").isNotNull(), 1).otherwise(0))
app_engagement = engagement_df.groupBy("app_id") \
    .agg(sum("has_reply").alias("total_replies"), avg("rating").alias("avg_rating")) \
    .dropna()  # Drop rows with null values in the result

engagement_analysis = app_engagement.orderBy(col("total_replies").desc())
engagement_analysis.show()

print("# Question 5: Rating by App Age")
app_first_review_date = reviews_df.groupBy("review_id").agg(min("at").alias("first_review_date"))
reviews_with_app_age = reviews_df.join(app_first_review_date, "review_id") \
    .withColumn("app_age", datediff(current_date(), col("first_review_date")) / 365.0)
rating_by_app_age = reviews_with_app_age.groupBy("app_age") \
    .agg(avg("rating").alias("avg_rating")) \
    .orderBy("app_age") \
    .dropna()  # Drop rows with null values in the result
rating_by_app_age.show()

print("# Question 6: Sentiment Analysis")
reviews_with_length = reviews_df.withColumn("review_length", length(col("content")))
sentiment_df = reviews_with_length.withColumn("sentiment",
    when(lower(col("content")).rlike(" جيد|عظيم|ممتاز|رائع|جميل "), "Positive")
    .when(lower(col("content")).rlike(" سيء|ضعيف|فظيع|بطيء|فاشل "), "Negative")
    .otherwise("Neutral")
)

review_stats = sentiment_df.agg(
    avg("review_length").alias("avg_review_length"),
    count(when(col("sentiment") == "Positive", True)).alias("positive_count"),
    count(when(col("sentiment") == "Negative", True)).alias("negative_count"),
    count(when(col("sentiment") == "Neutral", True)).alias("neutral_count")
).dropna()  # Drop rows with null values in the result

review_stats.show()

print("# Question 7: Ratings by Version")
reviews_with_version = reviews_df.withColumn("review_created_version", col("review_created_version").cast("double"))
ratings_by_version = reviews_with_version.groupBy("app_id", "review_created_version") \
    .agg(
        avg("rating").alias("avg_rating"),
        count("rating").alias("rating_count")
    ) \
    .orderBy("app_id", "review_created_version") \
    .dropna()  # Drop rows with null values in the result
ratings_by_version.show(50)

print("# Question 8: Monetization Potential")
for category in categories:
    df = spark.read.option("header", "true").csv(f"/Users/shooqalsu/Desktop/6457824/raw/categories/{category}.csv") \
        .withColumn("score", col("score").cast("double"))
    positive_reviews_df = df.filter(col("score") > 3)
    monetization_potential = positive_reviews_df.agg(count("appId").alias("positive_reviews")).dropna()  # Drop rows with null values in the result
    ranked_monetization_potential = monetization_potential.orderBy(col("positive_reviews").desc())
    print(f"Monetization potential for {category}:")
    ranked_monetization_potential.show()

print("# Question 10: Negative Review Analysis")
negative_reviews_df = reviews_df.filter(col("rating") <= 2)
negative_reviews_with_reasons = negative_reviews_df.withColumn("reason", 
    when(lower(col("content")).contains("crash"), "Crash")
    .when(lower(col("content")).contains("bug"), "Bug")
    .when(lower(col("content")).contains("slow"), "Slow Performance")
    .when(lower(col("content")).contains("freeze"), "Freeze")
    .otherwise("Other")
)

issues_by_category = negative_reviews_with_reasons.groupBy("reason").count().dropna()  # Drop rows with null values in the result
issues_by_category.show()

print("the end")
# Stop the Spark session
spark.stop()
