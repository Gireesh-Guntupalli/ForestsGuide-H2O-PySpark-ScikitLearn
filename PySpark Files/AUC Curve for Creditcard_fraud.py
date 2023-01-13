##this code build random forest model on creditcard_fraud_detection dataset,
##and the accuracy of the model is found using Precesion-Recall curve analysis

from imports import *

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

    # if 1 == 1:
    #     ##CALLING RANDOM FOREST CLASSIFIER
    model = RandomForestClassifier(featuresCol='Independent_features', labelCol=y_names,
                                       numTrees=100,featureSubsetStrategy="sqrt", maxDepth=4,
                                       subsamplingRate= 1,impurity="gini",
                                       minInfoGain=0, minInstancesPerNode =1,
                                       seed=42)
    # elif algorithm == "SVM":
    #     model = LinearSVC(featuresCol='Independent_features', labelCol=y_names,
    #                       tol=0.001)
    #
    # else:
    #     model = LogisticRegression(featuresCol='Independent_features', labelCol=y_names,
    #                                maxIter=100, fitIntercept=True, family="binomial")

    ##CONVERTING NUMPY TRAIN AND TEST SETS TO PANDAS DATAFRAMES
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
    trainingData = mod_trainingData.select(["Independent_features", y_names])
    featureassembler = VectorAssembler(inputCols=x_names, outputCol='Independent_features')
    mod_testData = featureassembler.transform(spark_testData)
    testData = mod_testData.select(["Independent_features", y_names])

    ##MODEL FITTING
    model = model.fit(trainingData)
    # if algorithm == "Random Forest":
    #     pipe = Pipeline(stages = [model])
    #     mod = pipe.fit(trainingData)
    #     trainingData2 =  mod.transform(trainingData)
    #     print(f"Feature importnces: {mod.stages[-1].featureImportances}")
    #
    #     def ExtractFeatureImp(featureImp, dataset, featuresCol):
    #         list_extract = []
    #         for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
    #             list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    #         varlist = pd.DataFrame(list_extract)
    #         varlist['score'] = varlist['id'].apply(lambda x: featureImp[x])
    #         return (varlist.sort_values('score', ascending=False))
    #
    #     ExtractFeatureImp(mod.stages[-1].featureImportances, trainingData2, 'Independent_features').to_csv("FeatureImportancesFold10SkLearnUni.csv")
    #     print(ExtractFeatureImp(mod.stages[-1].featureImportances, trainingData2, 'Independent_features'))
    #     ##Refered from https://www.timlrx.com/blog/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator
    predictions = model.transform(testData)
    preds_model = predictions.select("prediction")
    pandas_preds = preds_model.toPandas()
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
axes.set_title("Precision-Recall curve for PySpark Model.")
axes.set_xlabel('Recall', fontsize = '12')
axes.set_ylabel('Precision', fontsize = '12')
axes.legend(loc='lower left', fontsize ='12')

f.tight_layout()
f.savefig('result2.png')


print(f"Accuracy : {np.mean(accu_score)}")
print(f"F1_Score : {np.mean(f1_score)}")
print(f"Precision: {np.mean(precision)}")
print(f"M A E    : {np.mean(mae)}")
print(f"M S E    :{np.mean(mse)}")
print(f"Recall   : {np.mean(recall)}")

print("Finished.")