# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, PolynomialExpansion
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

def main():
    # Initialize SparkSession
    spark = SparkSession.builder.appName("PolynomialRegression").getOrCreate()

    # Load data
    data = (spark.read.csv("Y:\Documents\GitHub\CarSmartConsultancy\Data\cleaned_data\Predictive_Modeling.csv", header=True, inferSchema=True))
    # convert datetime to numeric
    data = data.withColumn("registration_date", unix_timestamp("registration_date"))
    data = data.withColumn("registration_date", data["registration_date"].cast(DoubleType()))

    # drop missing price value's rows
    data_subset = data.dropna(subset=['price'])

    # Split data into training and test sets
    train, test = data_subset.randomSplit([0.7, 0.3], seed=42)

    # select relevant columns and combine into features column
    inputCols = ["arf", "registration_date", "power", "omv"]
    # Assemble features
    assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
    train_data = assembler.transform(train).select("price", "features")
    test_data = assembler.transform(test).select("price", "features")

    # Apply polynomial expansion to features
    poly_exp = PolynomialExpansion(degree=2, inputCol="features", outputCol="poly_features")
    train_data = poly_exp.transform(train_data)
    test_data = poly_exp.transform(test_data)

    # Initialize linear regression model
    lr = LinearRegression(labelCol="price", featuresCol="poly_features")

    # Fit model to training data
    lr_model = lr.fit(train_data)

    # Make predictions on test data
    test_predictions = lr_model.transform(test_data)

    # Print model summary metrics on training set
    lr_summary_train = lr_model.summary
    print(f"Polynomial term performance on train data:")
    print(f"Polynomial RMSE for train data: {lr_summary_train.rootMeanSquaredError}")
    print(f"Polynomial R2 for test data: {lr_summary_train.r2}")
    print(f"----------------------------------------------------")

    # Print model summary metrics on test set
    lr_summary_test = lr_model.evaluate(test_data)
    test_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
    test_rmse = test_evaluator.evaluate(test_predictions)
    print(f"Polynomial term performance on test data:")
    print(f"Polynomial RMSE for test data: " + str(test_rmse))
    print(f"Polynomial R2 for test data: {lr_summary_test.r2}")
    print(f"----------------------------------------------------")

if __name__ == '__main__':
    main()