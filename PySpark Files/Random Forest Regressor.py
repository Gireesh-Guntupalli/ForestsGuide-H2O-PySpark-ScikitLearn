import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('StudeinArbeit').getOrCreate()
from pyspark.sql.functions import *
from sklearn.metrics import r2_score
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorSlicer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv("housing.csv") #Importing data as dataframe
print(df.head())
print(" ")
print(f"Shape of the dataset: {df.shape}")
print(" ")
print(f"Is there any null values in the dataset: {df.isnull().values.any()}. ")
print(" ")
df.dropna(inplace=True) #dropping null values.
print(f"The shape of the data after cleaning null values is: {df.shape}.")
print(df.dtypes)

print(f"Different categories in the Ocean column: {df.ocean_proximity.unique()}")
##ocean_proximity column is object type and has following unique values
## ['<1H OCEAN' ,'INLAND','NEAR OCEAN', 'NEAR BAY', 'ISLAND'] which are encoded to numericals.
df['ocean_proximity'].replace(['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'], [0,1,2,3,4], inplace=True)

print(df)
print(" ")

##Preparing Independent and dependent variables
x=df.drop(['longitude','latitude','median_house_value'],axis=1)
y = df['median_house_value']

x_np = np.array(x)
y_np = np.array(y)

x_names = list(x.columns.values)
y_names = 'median_house_value'
print(f'x_names:{x_names}')
print(f'y_names:{y_names}')
print(" ")

mse = []
mae =[]
r2 = []

from sklearn.model_selection import KFold
k = 20
cv = KFold(n_splits=k)

for train_index, test_index in cv.split(df):
    x_np_train, y_np_train = x_np[train_index], y_np[train_index]
    x_np_test, y_np_test = x_np[test_index], y_np[test_index]

    scaler = StandardScaler()
    x_np_train = scaler.fit_transform(x_np_train)
    x_np_test = scaler.transform(x_np_test)

    model = RandomForestRegressor(
        featuresCol='Independent_features', labelCol=y_names,
        numTrees=100,featureSubsetStrategy='onethird', maxDepth=8,
        subsamplingRate= 1,impurity='variance',
        minInfoGain=0, minInstancesPerNode =1,
        seed=42
    )
    x_train = pd.DataFrame(x_np_train, columns=x_names)
    x_test = pd.DataFrame(x_np_test, columns=x_names)
    y_train = pd.DataFrame(y_np_train, columns=[y_names])
    y_test = pd.DataFrame(y_np_test, columns=[y_names])

    ##CONCATING x_train + y_train AND x_test + y_test TO APPLY VECTOR ASSEMBLER
    train_set = pd.concat([x_train, y_train], axis=1)
    test_set = pd.concat([x_test, y_test], axis=1)

    ##COVERTING train_set AND test_set FROM PANDAS DF TO SPARK DF AS VECTOR ASSEMBLER
    ##ONLY ACCEPTS SPARK DATAFRAMES FOR TRANSFORMATION
    spark_trainingData = spark.createDataFrame(train_set)
    spark_testData = spark.createDataFrame(test_set)

    ##APPLYING VECTOR ASSEMBLER TO PERFORM SPARK FIT AND TRANSFORM OPERATIONS
    featureassembler = VectorAssembler(inputCols=x_names, outputCol='Independent_features')
    mod_trainingData = featureassembler.transform(spark_trainingData)
    mod_trainingData.show()
    trainingData = mod_trainingData.select(["Independent_features", y_names])
    trainingData.show(n=5, truncate=10)
    featureassembler = VectorAssembler(inputCols=x_names, outputCol='Independent_features')
    mod_testData = featureassembler.transform(spark_testData)
    testData = mod_testData.select(["Independent_features", y_names])

    ##MODEL FITTING
    model = model.fit(trainingData)

    predictions = model.transform(testData)
    preds_model = predictions.select("prediction")
    pandas_preds = preds_model.toPandas()
    preds = pandas_preds.to_numpy()

    MAE = mean_absolute_error(y_np_test, preds)
    MSE = mean_squared_error(y_np_test, preds, squared=False) #gives RMSE value
    R2 = r2_score(y_np_test, preds)

    # accu_score.append(accur)
    mse.append(MSE)
    mae.append(MAE)
    r2.append(R2)

print("The model is built and the evaluation metrics on the test data are as follows: ")
print(" ")
print(f"M A E    : {np.mean(mae)}")
print(f"M S E    :{np.mean(mse)}")
print(f"R2_Score :{np.mean(r2)}")
