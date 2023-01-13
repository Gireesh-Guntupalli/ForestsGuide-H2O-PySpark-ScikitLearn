from imports import *

accu_score = []
f1_score = []
precision =[]
mse = []
mae =[]
recall = []

from sklearn.model_selection import KFold
k = 10
cv = KFold(n_splits=k)

def build(df, x_np, y_np, x_names, y_names):

    for train_index, test_index in cv.split(df):
        x_np_train, y_np_train = x_np[train_index], y_np[train_index]
        x_np_test, y_np_test = x_np[test_index], y_np[test_index]

        scaler = StandardScaler()
        x_np_train= scaler.fit_transform(x_np_train)
        x_np_test = scaler.transform(x_np_test)

        model = RandomForestClassifier(
                                            n_estimators=100,
                                           max_features="sqrt",max_depth=4,max_samples= None,
                                           criterion="gini",
                                           min_impurity_decrease=0,
                                           # # min_samples_leaf= 1, ##only used for H2o comparison
                                           min_samples_split=1,  ##only used for Pyspark comparison
                                           random_state=42)

        ##TRAINING THE MODEL
        model.fit(x_np_train, y_np_train)
        preds = model.predict(x_np_test)

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

    print(" ")
    print(f"Accuracy : {np.mean(accu_score)}")
    print(f"F1_Score : {np.mean(f1_score)}")
    print(f"Precision: {np.mean(precision)}")
    print(f"M A E    : {np.mean(mae)}")
    print(f"M S E    :{np.mean(mse)}")
    print(f"Recall   : {np.mean(recall)}")
    print(" ")