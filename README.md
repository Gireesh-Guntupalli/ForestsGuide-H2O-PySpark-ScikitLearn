# Consistency of Alogorithm-Impementations of different Machine Learning Frameworks

## To meet the study goal, this project creates random forest models in the machine learning libraries Scikit-Learn, H20, and PySpark and compares their performances to assess differences and determine what is causing these variances

This project consists of three folder H2O, PySpark and Scikit-Learn which contains 
5 random forest classifier models and a Random forest regressor model implementations,

In each folder is named after the machine learning framework, that is used to build the Random Forest model.
Ecah folder conists of Build.py, imports.py and FIT - {task name}, python code files.

### Build.py module:
The real algorithm implementation occurs 'build()' function of Build.py file. The build() method is comprised up of a 'for loop' that iterates for K-Fold times.
Within this for loop, the model is trained, predictions are obtained and evaluation metrics are computed.

### import.py:
This python module imports all the library requirements need to preprcosess, build model and evulate the model.

### FIT module:
Each folder has 5 FIT files, each one for different classification task.
In this module the dataset is imported and any preprocessing is done. 
After preprocessing the data build() function of Build.py os called to build the model and get predictions.
(It has to be noted that this project focuses more on algorithm implemention part than the data wrangling portion.)

### Docker Files:
Docker file executes the requirement installations. 
Each framework has its own docker file. This project is implemented in PyCharm IDE with each docker image used to set up the project interpreter.
This is done to avoid library dependency errors.

### Datases used for the models:
The datasets used for the project can be downloaded from this [link](https://1drv.ms/u/s!AtfAgPR4VDcEwtVMZcFZlcvRKB4fpg?e=FUcmIj)

