import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
## ['<1H OCEAN' ,'INLAND','NEAR OCEAN', 'NEAR BAY', 'ISLAND'] which are encoded to numerical.
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
        n_estimators=100,
        max_features=0.3333,max_depth=8,max_samples= None,
        criterion='squared_error',
        min_impurity_decrease=0,
        min_samples_leaf= 1, ##only used for H2o comparison
        # min_samples_split=1,  ##only used for Pyspark comparison
        random_state=42)
    model.fit(x_np_train, y_np_train)
    preds = model.predict(x_np_test)
    pred_frame = pd.DataFrame(preds, columns = ["predictions"])
    y_test_frame = pd.DataFrame(y_np_test, columns = ["true values"])
    temp_frame = pd.concat([pred_frame, y_test_frame], axis=1)
    print(temp_frame)

    MAE = mean_absolute_error(y_np_test, preds)
    MSE = mean_squared_error(y_np_test, preds, squared=False) #gives RMSE values
    R2 = r2_score(y_np_test, preds)

    mse.append(MSE)
    mae.append(MAE)
    r2.append(R2)

print("The model is built and the evaluation metrics on the test data are as follows: ")
print(" ")
print(f"M A E    : {np.mean(mae)}")
print(f"R M S E    :{np.mean(mse)}")
print(f"R2_Score :{np.mean(r2)}")