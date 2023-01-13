from imports import *

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

f, axes = plt.subplots(1,figsize=(5, 5))

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

    x_np_train, y_np_train = x_np[train_index], y_np[train_index]
    x_np_test, y_np_test = x_np[test_index], y_np[test_index]

    scaler = StandardScaler()
    x_np_train = scaler.fit_transform(x_np_train)
    x_np_test = scaler.transform(x_np_test)

    # if 1 ==1:
    model = RandomForestClassifier(
            n_estimators=100,
            max_features="sqrt",max_depth=4,max_samples= None,
            criterion="gini",
            min_impurity_decrease=0,
            # min_samples_leaf= 1, ##only used for H2o comparison
            min_samples_split=1,  ##only used for Pyspark comparison
            random_state=42)

    # elif algorithm == "SVM":
    #     model = SVC(C=1.0, kernel="rbf", gamma="auto", random_state=42)
    #
    # else:
    #     model = LogisticRegression(max_iter=100, fit_intercept=True,
    #                                multi_class="ovr", solver="lbfgs",
    #                                random_state=42)

    ##TRAINING THE MODEL
    model.fit(x_np_train, y_np_train)
    preds = model.predict(x_np_test)

    pre, re, thre = precision_recall_curve(y_np_test, preds)
    lab = 'Fold %d AUC=%.4f' % (i + 1, auc(re, pre))
    axes.step(re, pre, label=lab)
    y_real.append(y_np_test)
    y_proba.append(preds)
    # if algorithm == "Random Forest":
    #     if i == i:
    #         importances = model.feature_importances_
    #         df_fimp = pd.DataFrame({"Features": x_names, 'Feat.Importances': importances})
    #         print(df_fimp.sort_values(by="Feat.Importances", ascending=False))
    #         df_fimp.to_csv("FeatureImportancesFold10SkLearnUni.csv")

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
axes.set_title('Precision-Recall curve for Scikit-Learn Model.')
axes.set_xlabel('Recall', fontsize='12')
axes.set_ylabel('Precision', fontsize='12')
axes.legend(loc='lower left', fontsize='12')

f.tight_layout()
f.savefig('result2.png')

# print(f"Model - {algorithm} is built and evaluation report is as follows >>>")
print(" ")
print(f"Accuracy : {np.mean(accu_score)}")
print(f"F1_Score : {np.mean(f1_score)}")
print(f"Precision: {np.mean(precision)}")
print(f"M A E    : {np.mean(mae)}")
print(f"M S E    :{np.mean(mse)}")
print(f"Recall   : {np.mean(recall)}")
print(" ")