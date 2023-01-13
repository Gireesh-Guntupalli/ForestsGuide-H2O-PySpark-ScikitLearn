from imports import *
from sklearn.model_selection import KFold
k = 10
cv = KFold(n_splits=k)

accu_score = []
f1_score = []
precision =[]
mse = []
mae =[]
recall = []

def build(df, x_np, y_np, x_names, y_names):

    for train_index, test_index in cv.split(df):
        ##GETTING TRAIN AND TESTS FROM KFOLD SPLITS
        x_np_train, y_np_train = x_np[train_index], y_np[train_index]
        x_np_test, y_np_test = x_np[test_index], y_np[test_index]

        scaler = StandardScaler()
        x_np_train = scaler.fit_transform(x_np_train)
        x_np_test = scaler.transform(x_np_test)


        ##CALLING RANDOM FOREST CLASSIFIER
        model = RandomForestClassifier(featuresCol='Independent_features',labelCol=y_names,
                                           numTrees=100,featureSubsetStrategy="sqrt", maxDepth=4,
                                           subsamplingRate= 1,impurity="gini",
                                           minInfoGain=0, minInstancesPerNode =1,
                                           seed=42)

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
        mod_trainingData.show()
        trainingData = mod_trainingData.select(["Independent_features", y_names])
        trainingData.show(n=5,truncate=10)
        featureassembler = VectorAssembler(inputCols=x_names, outputCol='Independent_features')
        mod_testData = featureassembler.transform(spark_testData)
        testData = mod_testData.select(["Independent_features", y_names])

        ##MODEL FITTING
        model = model.fit(trainingData)

        predictions = model.transform(testData)
        preds_model = predictions.select("prediction")
        pandas_preds = preds_model.toPandas()
        preds = pandas_preds.to_numpy()

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

    print(f"Accuracy : {np.mean(accu_score)}")
    print(f"F1_Score : {np.mean(f1_score)}")
    print(f"Precision: {np.mean(precision)}")
    print(f"M A E    : {np.mean(mae)}")
    print(f"M S E    :{np.mean(mse)}")
    print(f"Recall   : {np.mean(recall)}")

    print("Finished.")