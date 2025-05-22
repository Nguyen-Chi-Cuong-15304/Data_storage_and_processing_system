from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql import DataFrame
import logging

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Cấu hình ---
HDFS_RAW_DATA_PATH = "hdfs://namenode-h2dn:9000/user/data/device_raw_parquet"
MODEL_PATH = "hdfs://namenode-h2dn:9000/user/models/sales_prediction_model"

def main():
    logger.info("Starting Spark job to train sales prediction model")

    # Khởi tạo Spark Session
    spark = SparkSession.builder \
        .appName("TrainSalesPredictionModel") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode-h2dn:9000") \
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession created.")

    try:
        # Đọc dữ liệu từ HDFS
        logger.info(f"Reading Parquet data from {HDFS_RAW_DATA_PATH}")
        df = spark.read.parquet(HDFS_RAW_DATA_PATH)

        # Hiển thị schema và một vài dòng dữ liệu
        logger.info("Schema of the DataFrame:")
        df.printSchema()
        logger.info("Sample data (5 rows):")
        df.show(5, truncate=False)

        # --- Tiền xử lý dữ liệu ---

        # 1. Chuyển đổi các cột phân loại thành dạng số
        categorical_columns = ["brand", "category", "color", "RAM", "ROM"]
        indexers = [
            StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
            for col in categorical_columns
        ]

        # 2. Tạo vector đặc trưng từ các cột số và cột phân loại đã được mã hóa
        numeric_columns = ["price", "price_old", "percent", "rating"]
        indexed_columns = [f"{col}_index" for col in categorical_columns]
        feature_columns = numeric_columns + indexed_columns

        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="raw_features",
            handleInvalid="skip"
        )

        # 3. Chuẩn hóa các đặc trưng số
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withStd=True,
            withMean=True
        )

        # 4. Pipeline tiền xử lý
        preprocessing_stages = indexers + [assembler, scaler]
        preprocessing_pipeline = Pipeline(stages=preprocessing_stages)

        # Áp dụng pipeline tiền xử lý
        logger.info("Applying preprocessing pipeline...")
        preprocessed_df = preprocessing_pipeline.fit(df).transform(df)

        # Loại bỏ các dòng có giá trị null trong cột mục tiêu (sold)
        preprocessed_df = preprocessed_df.filter(col("sold").isNotNull())

        # --- Chia dữ liệu thành tập huấn luyện và tập kiểm tra ---
        #train_df, test_df = preprocessed_df.randomSplit([0.8, 0.2], seed=42)
        categories = preprocessed_df.select("category").distinct().collect()
        train_dfs = []
        test_dfs = []

        for row in categories:
            category_value = row["category"]
            category_df = preprocessed_df.filter(col("category") == category_value)
            train_df_part, test_df_part = category_df.randomSplit([0.8, 0.2], seed=42)
            train_dfs.append(train_df_part)
            test_dfs.append(test_df_part)

        # Gộp các tập con thành tập huấn luyện và tập kiểm tra chung
        train_df = train_dfs[0]
        test_df = test_dfs[0]
        for i in range(1, len(train_dfs)):
            train_df = train_df.union(train_dfs[i])
            test_df = test_df.union(test_dfs[i])

        logger.info(f"Training data size: {train_df.count()}")
        logger.info(f"Testing data size: {test_df.count()}")
        logger.info(f"Training data size: {train_df.count()}")
        logger.info(f"Testing data size: {test_df.count()}")

        # --- Huấn luyện mô hình ---
        # Sử dụng LinearRegression
        # lr = LinearRegression(featuresCol="features", labelCol="sold", predictionCol="prediction")
        lr = GeneralizedLinearRegression(
            featuresCol="features",
            labelCol="sold",
            predictionCol="prediction",
            family="gaussian",  # Phân phối: gaussian, poisson, gamma, tweedie
            link="identity",  # Hàm liên kết: identity, log, inverse
            maxIter=100,
            regParam=0.01
        )
        logger.info("Training Linear Regression model...")
        lr_model = lr.fit(train_df)

        # (Tùy chọn) Sử dụng RandomForestRegressor nếu muốn mô hình phức tạp hơn
        # rf = RandomForestRegressor(featuresCol="features", labelCol="sold", predictionCol="prediction", numTrees=50)
        # logger.info("Training Random Forest Regressor model...")
        # rf_model = rf.fit(train_df)

        # --- Đánh giá mô hình trên tập kiểm tra ---
        predictions = lr_model.transform(test_df)
        evaluator_rmse = RegressionEvaluator(labelCol="sold", predictionCol="prediction", metricName="rmse")
        evaluator_r2 = RegressionEvaluator(labelCol="sold", predictionCol="prediction", metricName="r2")

        rmse = evaluator_rmse.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)

        logger.info(f"Model evaluation on test set:")
        logger.info(f"RMSE: {rmse}")
        logger.info(f"R²: {r2}")

        # Hiển thị một vài dự đoán
        logger.info("Sample predictions (5 rows):")
        predictions.select("sold", "prediction", "features").show(5, truncate=False)
        predictions.select(
            "sold",
            "prediction",
            "brand",
            "category",
            "color",
            "RAM",
            "ROM",
            "price",
            "price_old",
            "percent",
            "rating"
        ).show(25, truncate=False)

        # --- Lưu mô hình ---
        logger.info(f"Saving model to {MODEL_PATH}")
        lr_model.write().overwrite().save(MODEL_PATH)

    except Exception as e:
        logger.error(f"ERROR during model training: {e}", exc_info=True)
    finally:
        logger.info("Stopping SparkSession...")
        spark.stop()
        logger.info("Spark job finished.")

if __name__ == "__main__":
    main()