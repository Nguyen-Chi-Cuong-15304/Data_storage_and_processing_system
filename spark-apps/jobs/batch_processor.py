from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, min as spark_min, max as spark_max, count, lit, date_format, to_date, current_date, date_sub
from pyspark.sql.types import DoubleType
import logging

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Cấu hình ---
HDFS_RAW_DATA_PATH = "hdfs://namenode-h2dn:9000/user/data/gold_raw_parquet"
ES_NODES = "elasticsearch-h2dn"
ES_PORT = "9200"
ES_INDEX = "gold_prices_prod"

def main():
    logger.info("Starting Spark Batch job: HDFS to Elasticsearch Batch Views")

    # Khởi tạo Spark Session với cấu hình HDFS
    spark = SparkSession \
        .builder \
        .appName("HDFSToESBatchViews") \
        .config("spark.jars.packages", "org.elasticsearch:elasticsearch-spark-30_2.12:8.9.0") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode-h2dn:9000") \
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession created.")

    # Đọc dữ liệu từ HDFS (Parquet)
    try:
        hdfs_df = spark.read.parquet(HDFS_RAW_DATA_PATH)
        logger.info(f"Successfully read data from HDFS path '{HDFS_RAW_DATA_PATH}'. Schema:")
        hdfs_df.printSchema()
    except Exception as e:
        logger.error(f"ERROR reading data from HDFS: {e}")
        spark.stop()
        exit()

    # Lọc dữ liệu trong 30 ngày gần nhất
    hdfs_df = hdfs_df.filter(
        to_date(col("price_date"), "yyyy-MM-dd") >= date_sub(current_date(), 30)
    )

    # Tính toán Batch Views (Giá TB, Min, Max hàng ngày theo loại vàng)
    daily_agg_df = hdfs_df \
        .groupBy("price_date", "gold_type") \
        .agg(
            avg("buy_price").alias("avg_buy_price"),
            avg("sell_price").alias("avg_sell_price"),
            spark_min("buy_price").alias("min_buy_price"),
            spark_max("buy_price").alias("max_buy_price"),
            spark_min("sell_price").alias("min_sell_price"),
            spark_max("sell_price").alias("max_sell_price"),
            count("*").alias("record_count")
        ) \
        .withColumn("view_type", lit("batch"))

    # Tạo ID duy nhất cho mỗi bản ghi Batch View
    batch_view_df = daily_agg_df \
        .withColumn("doc_id", date_format(col("price_date"), "yyyy-MM-dd_") + col("gold_type"))

    logger.info("Batch views calculated. Schema:")
    batch_view_df.printSchema()
    batch_view_df.show(5, truncate=False)

    # Ghi Batch Views vào Elasticsearch
    try:
        batch_view_df.write \
            .format("org.elasticsearch.spark.sql") \
            .mode("append") \
            .option("es.nodes", ES_NODES) \
            .option("es.port", ES_PORT) \
            .option("es.resource", f"{ES_INDEX}/_doc") \
            .option("es.mapping.id", "doc_id") \
            .option("es.write.operation", "upsert") \
            .option("es.nodes.wan.only", "true") \
            .option("es.index.auto.create", "true") \
            .save()
        logger.info("Successfully wrote batch views to Elasticsearch.")
    except Exception as e:
        logger.error(f"ERROR writing batch views to Elasticsearch: {e}")
    finally:
        spark.stop()
        logger.info("Spark Batch job finished.")

if __name__ == "__main__":
    main()