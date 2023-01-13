from build import *

df = pd.read_csv("rain_in_australia.csv")
print(df.head())
print(" ")
print(f"Shape of the dataset: {df.shape}")
print(" ")
print(f"Is there any null values in the dataset: {df.isnull().values.any()}. ")
print(" ")

# One Hot encoding for categorical columns
df = pd.get_dummies(data=df, columns=['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'])
#dropping columns that are not so significant.
df.drop(['Date', 'Location', 'Cloud9am', 'Cloud3pm', 'Evaporation', 'Sunshine'], axis=1, inplace=True)

# Replace 'RainTomorrow' with [0,1] inplace of [No, Yes]
df['RainTomorrow'].replace(['No', 'Yes'], [0, 1], inplace=True)
df = df.dropna(subset=(['RainTomorrow'])) ##Droping null values from target column

##IMputing null values in these columns with mean values of that column respectively.
df['MinTemp'].fillna(int(df['MinTemp'].mean()), inplace=True)
df['MaxTemp'].fillna(int(df['MaxTemp'].mean()), inplace=True)
df['Rainfall'].fillna(int(df['Rainfall'].mean()), inplace=True)
df['WindGustSpeed'].fillna(int(df['WindGustSpeed'].mean()), inplace=True)
df['WindSpeed3pm'].fillna(int(df['WindSpeed3pm'].mean()), inplace=True)
df['WindSpeed9am'].fillna(int(df['WindSpeed9am'].mean()), inplace=True)
df['Humidity3pm'].fillna(int(df['Humidity3pm'].mean()), inplace=True)
df['Humidity9am'].fillna(int(df['Humidity9am'].mean()), inplace=True)
df['Pressure3pm'].fillna(int(df['Pressure3pm'].mean()), inplace=True)
df['Pressure9am'].fillna(int(df['Pressure9am'].mean()), inplace=True)
df['Temp3pm'].fillna(int(df['Temp3pm'].mean()), inplace=True)
df['Temp9am'].fillna(int(df['Temp9am'].mean()), inplace=True)


# Defining dependant and independent variables
x= df.drop(columns='RainTomorrow', axis=1)
y= df['RainTomorrow']

##CONVERTING PANDAS x AND y TO NUMPY ARRAY FOR KFOLD APPLICATION
x_np = np.array(x)
y_np = np.array(y)

x_names = list(x.columns.values)
y_names = "Class"
print(f'x_names:{x_names}')
print(f'y_names:{y_names}')
print(" ")

# BUILDING AND EVALUATING THE MODEL
build(df, x_np, y_np, x_names, y_names)
