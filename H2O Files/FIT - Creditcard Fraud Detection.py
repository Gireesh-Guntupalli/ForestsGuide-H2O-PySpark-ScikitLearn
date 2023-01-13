from build import *

df = pd.read_csv("creditcard_fraud_detection.csv")
print(df.head())
print(" ")

print(f"Shape of the dataset: {df.shape}")
print(" ")
print(f"Is there any null values in the dataset: {df.isnull().values.any()}. ")
print(" ")

fraud = df[df['Class']==1]
normal = df[df['Class']==0]
print(f"The shape of fraud is: {fraud.shape}, The shape of normal is:  {normal.shape}")
print("")

x = df.drop(["Class"], axis=1)
y = df["Class"]

##CONVERTING PANDAS x AND y TO NUMPY ARRAY FOR KFOLD APPLICATION
x_np = np.array(x)
y_np = np.array(y)

x_names = list(x.columns.values)
y_names = "CreditCard"
print(f'x_names:{x_names}')
print(f'y_names:{y_names}')
print(" ")

# ##BUILDING AND EVALUATING THE MODEL
build("Random Forest", df,x_np,y_np, x_names, y_names)