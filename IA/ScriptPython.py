# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
import seaborn as sns

# Load dataset
url = r'C:\Users\crism\OneDrive\Desktop\Tesis Final\CSV\DataSet.csv'
dataset = pandas.read_csv(url)  

ros = RandomOverSampler(random_state=40)
sm = SMOTE(random_state=42,k_neighbors=1)
array = dataset.values
X = array[:,0:9]
Y = array[:,9]
X_res, Y_res = ros.fit_resample(X,Y)
#X_res, Y_res = sm.fit_resample(X,Y)
#print(len(X_ros))
#print(len(Y_ros))
#X_res, Y_res = sm.fit_resample(X,Y)
print(len(X_res))

#plt.figure(figsize=(12,10))
#cor = dataset.corr()
#sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
#plt.show()

#scaler = MinMaxScaler()
#scaler = RobustScaler()
#scaler.fit(X_res)
#X_res = scaler.transform(X_res)


# shape
print(dataset.shape)

print(dataset.groupby('emotionLevel').size())
'''
# head
#print(dataset.head(20))

# descriptions
#print(dataset.describe())

# class distribution


# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
#scatter_matrix(dataset)
#plt.show()
'''
'''
X, y = make_classification(n_samples=5000, n_features=11, n_informative=11,
                                n_redundant=0, n_repeated=0, n_classes=5,
                                n_clusters_per_class=1,
                                class_sep=0.8, random_state=0)

sm = SMOTE(random_state=42)
array = dataset.values
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
print(sorted(Counter(y_resampled).items()))
'''
'''
#RF RF RF
rf = RandomForestRegressor(random_state = 42)
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model

'''

validation_size = 0.3
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_res, Y_res, test_size=validation_size, random_state=seed,shuffle=True)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('CART', DecisionTreeClassifier()))
models.append(('ClusterKM', KMeans(n_clusters=5, random_state=0)))
models.append(('RF', RandomForestClassifier(n_estimators=500)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('SVM', SVC(gamma='auto')))

'''
rf_random.fit(X_res, Y_res)

rf_random.best_params_

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_res, Y_res)
base_accuracy = evaluate(base_model, X_res, Y_res)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_res, Y_res)

'''

'''
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_res, Y_res)
grid_search.best_params_

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_res, Y_res)

#print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

print('Grid Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

'''

'''
#SVM SVM 
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X_res, Y_res)
    grid_search.best_params_
    return print(grid_search.best_params_)
'''
# evaluate each model in turn
sum = 0
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=8)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
'''    
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
'''
# # Make predictions on validation dataset
# knn = SVC(gamma='auto')
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print("Matriz de confusión")
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))    


knn = RandomForestClassifier()
knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions)) 
# auxi=[3735.709224813508,28.296411222132157,626,3192.008743022178,87752.0,5,0,1,0]
auxi=np.array([14316.6139665898638,325.66810457648718,183,506.636292129062,42455.0,3,5,2,1])
predic=auxi.reshape(1,-1)
predictions = knn.predict(predic)
print(predictions)

# # Save the model as a pickle in a file
# joblib.dump(knn, 'C:/Users/crism/OneDrive/Desktop/Tesis Final/IA/modelo_knn_entrenado.pkl')