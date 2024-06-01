from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("spark") \
    .master("spark://spark-master:7077")\
    .getOrCreate()

# Path to the input data
data_path = "hdfs://namenode:9000/home_credit/data/"

# Reading each CSV file into a Spark DataFrame
data_app_train = spark.read.csv(
    data_path + "application_train.csv", header=True, inferSchema=True)
data_app_test = spark.read.csv(
    data_path + "application_test.csv", header=True, inferSchema=True)

# Writing DataFrames to Parquet format
parquet_path = "hdfs://namenode:9000/home_credit/data_parquet/"
data_app_train.write.parquet(parquet_path + "application_train.parquet")
data_app_test.write.parquet(parquet_path + "application_test.parquet")

data_app_train_parquet = spark.read.parquet(
    parquet_path + "application_train.parquet")
data_app_train_parquet.show()
data_app_test_parquet = spark.read.parquet(
    parquet_path + "application_test.parquet")
data_app_test_parquet.show()

# Assuming data_app_train is your Spark DataFrame
data = data_app_train_parquet.drop('SK_ID_CURR')

# Rename 'TARGET' to 'label'
data = data.withColumnRenamed('TARGET', 'label')

# Handling null values for categorical and numeric columns
categorical_columns = [t[0] for t in data.dtypes if t[1] == 'string']
numerical_columns = [t[0]
                     for t in data.dtypes if t[1] != 'string' and t[0] != 'label']

# Fill missing values for categorical and numerical data
data = data.fillna('unknown', subset=categorical_columns)
data = data.fillna(0, subset=numerical_columns)

# Index and encode categorical columns
indexers = [StringIndexer(
    inputCol=c, outputCol=f"{c}_indexed", handleInvalid='keep') for c in categorical_columns]
encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(
), outputCol=f"{indexer.getOutputCol()}_encoded") for indexer in indexers]

# Assemble all the features together
assembler_inputs = [encoder.getOutputCol()
                    for encoder in encoders] + numerical_columns
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# Chia dữ liệu
# Split the data into training and test sets
(train_data, test_data) = data.randomSplit([0.8, 0.2], seed=1234)

# Logistic Regression
# Define the Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Create the pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])
# Fit the model
model_lgt = pipeline.fit(train_data)
model_lgt.write().overwrite().save("model_lgt")
# Make predictions
predictions = model_lgt.transform(test_data)

# Đánh giá
# Evaluate the model
binary_evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
multi_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction")
# Compute metrics
roc_auc = binary_evaluator.evaluate(predictions)
accuracy = multi_evaluator.evaluate(
    predictions, {multi_evaluator.metricName: "accuracy"})
f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
precision = multi_evaluator.evaluate(
    predictions, {multi_evaluator.metricName: "weightedPrecision"})
recall = multi_evaluator.evaluate(
    predictions, {multi_evaluator.metricName: "weightedRecall"})

# Confusion matrix
prediction_and_labels = predictions.select("prediction", "label").rdd.map(
    lambda row: (float(row["prediction"]), float(row["label"])))
metrics = MulticlassMetrics(prediction_and_labels)
confusion_matrix = metrics.confusionMatrix()

print("Logistic Regression Model Evaluation Results:")
print("ROC AUC: %f", roc_auc)
print("Accuracy: %f", accuracy)
print("F1 Score: %f", f1)
print("Precision: %f", precision)
print("Recall: %f", recall)
print("Confusion Matrix:\n%s", confusion_matrix)

# Random Forest
# Define the Random Forest model
rf = RandomForestClassifier(featuresCol="features", labelCol="label")

# Create the pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])
# Fit the model
model_rf = pipeline.fit(train_data)
model_rf.write().overwrite().save("model_rf")
# Make predictions
predictions = model_rf.transform(test_data)

# Đánh giá
# Evaluate the model
binary_evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
multi_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction")
# Compute metrics
roc_auc = binary_evaluator.evaluate(predictions)
accuracy = multi_evaluator.evaluate(
    predictions, {multi_evaluator.metricName: "accuracy"})
f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
precision = multi_evaluator.evaluate(
    predictions, {multi_evaluator.metricName: "weightedPrecision"})
recall = multi_evaluator.evaluate(
    predictions, {multi_evaluator.metricName: "weightedRecall"})

# Confusion matrix
prediction_and_labels = predictions.select("prediction", "label").rdd.map(
    lambda row: (float(row["prediction"]), float(row["label"])))
metrics = MulticlassMetrics(prediction_and_labels)
confusion_matrix = metrics.confusionMatrix()

print("Random Forest Model Evaluation Results:")
print("ROC AUC: %f", roc_auc)
print("Accuracy: %f", accuracy)
print("F1 Score: %f", f1)
print("Precision: %f", precision)
print("Recall: %f", recall)
print("Confusion Matrix:\n%s", confusion_matrix)
spark.stop()
