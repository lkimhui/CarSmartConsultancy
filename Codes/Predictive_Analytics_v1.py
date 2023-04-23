# Import required libraries
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression

# Create a SparkSession
spark = SparkSession.builder.appName("car_price_prediction").getOrCreate()

# Load dataset (CSV file) into Spark DataFrame
car_data = spark.read.csv("TableauData_V1.1.csv",inferSchema=True,header=True)
# car_data = spark.read.format('csv').option('header', True).load('TableauData_V1.csv')

# Print the schema of the DataFrame
car_data.printSchema()

# Show the first 5 rows of the DataFrame
car_data.show(5)
# car_data.head() # this line is not working

# for item in car_data.head():
#     print(item)

# Setting up DataFrame for Machine Learning
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
car_data.columns

# We need to change the data into the form of two columns ("label", "features")
assembler = VectorAssembler(inputCols=["depreciation",
"reg_age",
"mileage",
"road_tax",
"omv",
"engine_cap",
"power",
"curb_weight"],
outputCol="features")
output = assembler.transform(car_data)
output.select("features").show()
output.show()

# Final dataset with two columns ("features", "label")
new_car_data = output.select("features","price")
new_car_data.show()

# Split into training and testing datasets
train_data, test_data = new_car_data.randomSplit([0.7,0.3])
train_data.describe().show()
test_data.describe().show()

# Create a linear regression model object
lr = LinearRegression(labelCol='price')

# Fit the model to the data and call this model lrModel
lrModel = lr.fit(train_data,)

# print the coefficients and intercept for linear regression
print("Coefficients: {} Intercept {}".format(lrModel.coefficients, lrModel.intercept))

test_result = lrModel.evaluate(test_data)
test_result.residuals.show()

print("RMSE:{}".format(test_result.rootMeanSquaredError))
print("MSE: {}".format(test_result.meanSquaredError))

# Stop the SparkSession
spark.stop()

