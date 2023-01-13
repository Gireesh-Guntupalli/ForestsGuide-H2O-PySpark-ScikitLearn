from build import *

df = pd.read_csv("term_deposit_subscription.csv", sep =";")
df.columns = ['age', 'job', 'marital', 'education', 'credit', 'housing', 'loan','contact', 'month', 'day_of_week',
               'duration', 'campaign', 'pdays','previous', 'poutcome', 'emp.var.rate', 'cons.price.idx','cons.conf.idx',
               'euribor3m', 'nr.employed', 'subscribed']

print(df.head())
print(" ")
print(f"Shape of the dataset: {df.shape}")
print(" ")
print(f"Is there any null values in the dataset: {df.isnull().values.any()}. ")
print(" ")

##dropping object type data columns which are unimportant for training.
df.drop(columns=[
    'nr.employed','pdays','euribor3m','emp.var.rate',
    'cons.price.idx','contact','month','campaign', 'day_of_week'
    ], axis=0, inplace=True)

##Ranking columns as per intuation.
df['job'].replace(["unemployed", "student",
                   "housemaid","blue-collar","retired","unknown",
                    "admin.", "technician",
                    "services", "self-employed",
                    "management","entrepreneur",
                   ],[0,0.5,1,1,1,1,2,2,3,3,4,4], inplace = True)

##Ranking columns as per intuation.
df["education"].replace([ "basic.4y","basic.6y","basic.9y","high.school","professional.course",
                          "university.degree","illiterate","unknown"],[1,1,1,1,2,2,0.9,0.9], inplace=True)

df["poutcome"].replace(["nonexistent","failure","success" ],[0,0,1], inplace=True)
df["subscribed"].replace(["no", "yes"], [0,1], inplace= True)
print(" ")

#Encoding 'marital','credit','housing','loan'
encoder = LabelEncoder()
lst = ['marital','credit','housing','loan']
for i in lst:
    df[i] = encoder.fit_transform(df[i])
print(df.head())
print(" ")

df.rename(columns={"cons.conf.idx": 'consconfidx'}, inplace=True)

x = df.drop(columns = 'subscribed',axis=1)
y = df['subscribed']

print(f"Shape of X features: {x.shape}")
print(f"Shape of Y targets: {y.shape}")
print(" ")

x_np = np.array(x)
y_np = np.array(y)

x_names = list(x.columns.values)
y_names = "subscribed"
print(f'x_names:{x_names}')
print(f'y_names:{y_names}')
print(" ")

#BUILDING AND EVALUATING THE MODEL
build(df,x_np,y_np,  x_names, y_names)
