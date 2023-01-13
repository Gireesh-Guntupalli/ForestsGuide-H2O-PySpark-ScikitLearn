##this code build random forest model on creditcard_fraud_detection dataset,
##and the accuracy of the model is found using Precesion-Recall curve analysis

from imports import *
from sklearn.metrics import precision_recall_curve, auc

df = pd.read_csv("creditcard.csv")
print(df.head())
print(" ")

print(f"Shape of the dataset: {df.shape}")
print(" ")
print(f"Is there any null values in the dataset: {df.isnull().values.any()}. ")
print(" ")

fraud = df[df['Class']==1]
normal = df[df['Class']==0]
print(f"The shape of fraud is: {fraud.shape}, The shape of normal is:  {normal.shape}")
print(" ")

x = df.drop(["Class"], axis=1)
y = df["Class"]

##CONVERTING PANDAS x AND y TO NUMPY ARRAY FOR KFOLD APPLICATION
x_np = np.array(x)
y_np = np.array(y)

x_names = list(x.columns.values)
y_names = "Class"
print(f'x_names:{x_names}')
print(f'y_names:{y_names}')
print(" ")

f, axes = plt.subplots(1, figsize=(5, 5))

from sklearn.model_selection import KFold
k = 10
cv = KFold(n_splits=k)

accu_score = []
f1_score = []
precision =[]
mse = []
mae =[]
recall = []
y_real = []
y_proba = []

for i, (train_index, test_index) in enumerate(cv.split(df)):
    ##GETTING TRAIN AND TESTS FROM KFOLD SPLITS
    x_np_train, y_np_train = x_np[train_index], y_np[train_index]
    x_np_test, y_np_test = x_np[test_index], y_np[test_index]

    scaler = StandardScaler()
    x_np_train = scaler.fit_transform(x_np_train)
    x_np_test = scaler.transform(x_np_test)

    model = H2ORandomForestEstimator(  # binomial_double_trees= False,
        ntrees=100, mtries=-1,
        max_depth=4, sample_rate=1,
        min_split_improvement=0,
        min_rows=1,  ##use to compare sklearn
        seed=42)

    x_train = pd.DataFrame(x_np_train, columns=x_names)
    x_test = pd.DataFrame(x_np_test, columns=x_names)
    y_train = pd.DataFrame(y_np_train, columns=[y_names])
    y_test = pd.DataFrame(y_np_test, columns=[y_names])

    train_set = pd.concat([x_train, y_train], axis=1)
    test_set = pd.concat([x_test, y_test], axis=1)

    train = h2o.H2OFrame(train_set)
    train[y_names] = train[y_names].asfactor()  # Factoring "Class" column values to make the algorithm
    # understand it is a classification problem, or it assumes as regression task and throws a
    # warning to convert numerical values [0,1] to categorical values
    test = h2o.H2OFrame(test_set)
    test[y_names] = test[y_names].asfactor()

    model.train(x=x_names,
                y=y_names,
                training_frame=train,
                validation_frame=test)

    test_preds = model.predict(test_data=test)
    pandas_preds = test_preds.as_data_frame()
    predictionsCSV = pd.concat([pandas_preds, y_test], axis=1)
    print(predictionsCSV.head(50))
    pandas_preds = pandas_preds["predict"]
    preds = pandas_preds.to_numpy()

    pre, re, thre = precision_recall_curve(y_np_test, preds)
    lab = 'Fold %d AUC=%.4f' % (i + 1, auc(re, pre))
    axes.step(re, pre, label=lab)
    y_real.append(y_np_test)
    y_proba.append(preds)

    ##FINDING METRICS
    accur = accuracy_score(y_np_test, preds)
    precise = precision_score(y_np_test, preds)
    MAE = mean_absolute_error(y_np_test, preds)
    MSE = mean_squared_error(y_np_test, preds)
    rcall = recall_score(y_np_test, preds)
    F1 = 2 * (precise * rcall) / (precise + rcall)

    accu_score.append(accur)
    precision.append(precise)
    mse.append(MSE)
    mae.append(MAE)
    recall.append(rcall)
    f1_score.append(F1)

y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
pre, re, thre = precision_recall_curve(y_real, y_proba)
lab = 'Overall AUC=%.4f' % (auc(re, pre))
axes.step(re, pre, label=lab, lw=2, color='black')
axes.set_title("Precision-Recall curve for H2O Model.")
axes.set_xlabel('Recall', fontsize='12')
axes.set_ylabel('Precision', fontsize='12')
axes.legend(loc='lower left', fontsize='12')

f.tight_layout()
f.savefig('result2.png')
