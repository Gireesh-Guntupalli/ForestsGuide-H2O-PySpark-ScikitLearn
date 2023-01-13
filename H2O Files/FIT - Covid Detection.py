from build import *

df = pd.read_csv("covid_detection.csv", low_memory = False)
print(df.head())
print(" ")
print(f"Shape of the dataset: {df.shape}")
print(" ")
print(f"Is there any null values in the dataset: {df.isnull().values.any()}.")
print(" ")

##Checking for uniques values in each column
for i in df.columns:
    print("Number of unique values in '{}' are '{}' and these values are:{}\n".format(i, len(df[i].unique()),
                                                                                      df[i].unique()))
##This data is very clumsy. There are many categoricals even for a binary feature.
##These features need to be cleaned to train the model
df = df.loc[(df.CLASIFFICATION_FINAL <= 3)] ##Considering only 1,2 and 3 classes.
df['SEX'].replace([1,2],[0,1], inplace = True) ##replacing class  1 with o and 2 with 1
df['USMER'].replace([2,1],[0,1], inplace = True) # no = 0, yes = 1
df['PATIENT_TYPE'].replace([1,2],[0,1], inplace = True)

df = df.loc[(df.PNEUMONIA == 1) | (df.PNEUMONIA == 2)] ##Considering only 1 and 2 classes out of [ 1  2 99]
df['PNEUMONIA'].replace([1,2],[0,1], inplace = True) ##Encoding column

#Number of unique values in 'DIABETES' are '3' and these values are:[ 2  1 98], emitting 98
df = df.loc[(df.DIABETES == 1) | (df.DIABETES == 2)]
df['DIABETES'].replace([2,1],[0,1], inplace = True)

##Similary cleaning the foolwing columns of the dataset:
df = df.loc[(df.COPD == 1) | (df.COPD == 2)]
df['COPD'].replace([2,1],[0,1], inplace = True)

df = df.loc[(df.ASTHMA == 1) | (df.ASTHMA == 2)]
df['ASTHMA'].replace([2,1],[0,1], inplace = True)

df = df.loc[(df.INMSUPR == 1) | (df.INMSUPR == 2)]
df['INMSUPR'].replace([2,1],[0,1], inplace = True)

df = df.loc[(df.HIPERTENSION == 1) | (df.HIPERTENSION == 2)]
df['HIPERTENSION'].replace([1,2],[0,1], inplace = True)

df = df.loc[(df.OTHER_DISEASE == 1) | (df.OTHER_DISEASE == 2)]
df['OTHER_DISEASE'].replace([2,1],[0,1], inplace = True)

df = df.loc[(df.CARDIOVASCULAR == 1) | (df.CARDIOVASCULAR == 2)]
df['CARDIOVASCULAR'].replace([2,1],[0,1], inplace = True)

df = df.loc[(df.OBESITY == 1) | (df.OBESITY == 2)]
df['OBESITY'].replace([2,1],[0,1], inplace = True)

df = df.loc[(df.RENAL_CHRONIC == 1) | (df.RENAL_CHRONIC == 2)]
df['RENAL_CHRONIC'].replace([2,1],[0,1], inplace = True)

df = df.loc[(df.TOBACCO == 1) | (df.TOBACCO == 2)]
df['TOBACCO'].replace([2,1],[0,1], inplace = True)

df.DATE_DIED = df.DATE_DIED.apply(lambda x: 0 if x == "9999-99-99" else 1)
df.PREGNANT = df.PREGNANT.apply(lambda x: x if x == 1 else 0)
df.INTUBED = df.INTUBED.apply(lambda x: x if x == 1 else 0)
df.ICU = df.ICU.apply(lambda x: x if x == 1 else 0)

df['AT_RISK'] = df['DATE_DIED'] + df['INTUBED'] + df['ICU']
print(df.AT_RISK.unique())
##Coverting classes of AT_RISK [1, 2, 0, 3] TO BINARY CLASSES O for not at risk AND 1 means at risk
df.AT_RISK = df.AT_RISK.apply(lambda x: 1 if x > 0 else 0)

print(df.info())
print(df.head())

y = df["AT_RISK"]
x = df.drop(["AT_RISK",'CLASIFFICATION_FINAL', 'INTUBED', 'ICU', 'DATE_DIED'], axis =1)
print(f"Shape of X features: {x.shape}")
print(f"Shape of Y targets: {y.shape}")
print(" ")

x_np = np.array(x)
y_np = np.array(y)
x_names = list(x.columns.values)
y_names = "AT_RISK"

#BUILDING AND EVALUATING THE MODEL
build("Random Forest", df,x_np,y_np, x_names, y_names)

