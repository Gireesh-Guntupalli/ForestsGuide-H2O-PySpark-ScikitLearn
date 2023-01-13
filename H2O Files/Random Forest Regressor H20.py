import pandas as pd
import numpy as np
import h2o
h2o.init() ##Connecting to H2O server
from h2o.estimators import H2ORandomForestEstimator
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

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
## ['<1H OCEAN' ,'INLAND','NEAR OCEAN', 'NEAR BAY', 'ISLAND'] which are enocoded to numericals.
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

    model = H2ORandomForestEstimator(  # binomial_double_trees= False,
        ntrees=100, mtries=-1,
        max_depth=8, sample_rate=1,
        min_split_improvement=0,
        min_rows=1,  ##use to compare sklearn
        seed=42)
    x_train = pd.DataFrame(x_np_train, columns=x_names)
    x_test = pd.DataFrame(x_np_test, columns=x_names)
    y_train = pd.DataFrame(y_np_train, columns=[y_names])
    y_test = pd.DataFrame(y_np_test, columns=[y_names])

    train_set = pd.concat([x_train, y_train], axis=1)
    test_set = pd.concat([x_test, y_test], axis=1)

    ##Coverting pandas frames to H2O  frames
    train = h2o.H2OFrame(train_set)
    test = h2o.H2OFrame(test_set)

    ##Model building..
    model.train(x=x_names,
                y=y_names,
                training_frame=train,
                validation_frame=test)
    test_preds = model.predict(test_data=test)
    pandas_preds = test_preds.as_data_frame()
    predictionsCSV = pd.concat([pandas_preds, y_test], axis=1)
    pandas_preds = pandas_preds["predict"]
    preds = pandas_preds.to_numpy() ##coverting h2o predictions to numpy array.

    MAE = mean_absolute_error(y_np_test, preds)
    MSE = mean_squared_error(y_np_test, preds, squared=False) ##squared = False gives RMSE values.
    R2 = r2_score(y_np_test, preds)

    # accu_score.append(accur)
    mse.append(MSE)
    mae.append(MAE)
    r2.append(R2)

print("The model is built and the evaluation metrics on the test data are as follows: ")
print(" ")
print(f"M A E    : {np.mean(mae)}")
print(f"R M S E    :{np.mean(mse)}")
print(f"R2_Score :{np.mean(r2)}")
