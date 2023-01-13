##this code build random forest model on creditcard_fraud_detection dataset,
##and the accuracy of the model is found using Precesion-Recall curve analysis

from build import *

df = pd.read_csv("bank_loan_classification.csv")
print(df.head())
print(" ")
print(f"Shape of the dataset: {df.shape}")
print(" ")
print(f"Is there any null values in the dataset: {df.isnull().values.any()}.")
print(" ")

df = df.drop(["ID","ZIP Code"], axis = 1)
category_0 = df[df.CreditCard==0]
print(f"The shape of category_0 is:{category_0.shape}.")
category_1 = df[df.CreditCard==1]
print(f"The shape of category_1 is:{category_1.shape}.")
print(" ")

x = df.drop(["Personal Loan"], axis=1)
y = df["Personal Loan"]

##CONVERTING PANDAS x AND y TO NUMPY ARRAY FOR KFOLD APPLICATION
x_np = np.array(x)
y_np = np.array(y)

x_names = list(x.columns.values)
y_names = "CreditCard"
print(f'x_names:{x_names}')
print(f'y_names:{y_names}')
print(" ")

#BUILDING AND EVALUATING THE MODEL
build(df,x_np,y_np, x_names, y_names)




