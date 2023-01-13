##Contains all the library imports required for building model

import pandas as pd
import numpy as np
import h2o
h2o.init() ##Connecting to H2O server
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error,mean_absolute_error
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators import H2OSupportVectorMachineEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt