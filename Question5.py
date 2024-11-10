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

print("# Question 5: Rating by App Age")
app_first_review_date = reviews_df.groupBy("review_id").agg(min("review_date").alias("first_review_date"))
reviews_with_app_age = reviews_df.join(app_first_review_date, "review_id") \
    .withColumn("app_age", datediff(current_date(), col("first_review_date")) / 365.0)
rating_by_app_age = reviews_with_app_age.groupBy("app_age") \
    .agg(avg("rating").alias("avg_rating")) \
    .orderBy("app_age")\
        .dropna() 

rating_by_app_age.show()