import pandas as pd
from imports import *

accu_score = []
f1_score = []
precision =[]
mse = []
mae =[]
recall = []
time_array = []

from sklearn.model_selection import KFold
k = 10
cv = KFold(n_splits=k)

def build(df, x_np, y_np, x_names, y_names):

    for train_index, test_index in cv.split(df):
        x_np_train, y_np_train = x_np[train_index], y_np[train_index]
        x_np_test, y_np_test = x_np[test_index], y_np[test_index]

        scaler = StandardScaler()
        x_np_train = scaler.fit_transform(x_np_train)
        x_np_test = scaler.transform(x_np_test)

        model = H2ORandomForestEstimator(#binomial_double_trees= False,
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

        ##MODEL FITTING
        model.train(x=x_names,
                 y=y_names,
                 training_frame=train,
                 validation_frame=test)
        perf = model.model_performance(test)
        print(perf)
        test_preds = model.predict(test_data=test)
        pandas_preds = test_preds.as_data_frame()
        predictionsCSV = pd.concat([pandas_preds, y_test], axis =1)
        print(predictionsCSV.head(50))
        pandas_preds = pandas_preds["predict"]
        preds = pandas_preds.to_numpy()

        ##FINDING METRICS
        accur = accuracy_score(y_np_test, preds)
        precise = precision_score(y_np_test, preds)
        MAE = mean_absolute_error(y_np_test, preds)
        MSE = mean_squared_error(y_np_test, preds)
        rcall = recall_score(y_np_test, preds)

        accu_score.append(accur)
        precision.append(precise)
        mse.append(MSE)
        mae.append(MAE)
        recall.append(rcall)


    print(f"Accuracy : {np.mean(accu_score)}")
    print(f"Precision: {np.mean(precision)}")
    print(f"M A E    : {np.mean(mae)}")
    print(f"M S E    :{np.mean(mse)}")
    print(f"Recall   : {np.mean(recall)}")

    print("Finished.")

