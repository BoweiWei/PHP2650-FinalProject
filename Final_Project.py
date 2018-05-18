import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from copy import deepcopy
from scipy.stats import gaussian_kde
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


my_data = pd.read_csv('default of credit card clients.csv', sep = ',', header = 1, index_col = 'ID')
column_names = list(my_data.columns.values)
AMT_idx = ['AMT' in x  for x in column_names] 
conti_idx = np.array(deepcopy(AMT_idx))
conti_idx[[0,4]] = True
cat_idx = [ not x  for x in conti_idx]

for i in range(len(column_names)):
    values = my_data.loc[:,column_names[i]]
    
    if conti_idx[i] == True:
        density = gaussian_kde(values)
        xs = np.linspace(0, max(values),200)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        plt.plot(xs,density(xs))
        plt.ylabel(column_names[i])
        
        plt.show()
        
    else:
        my_data.iloc[:, i] = my_data.iloc[:, i].astype('category')
        values.value_counts().plot(kind='bar')
        plt.ylabel(column_names[i])
        plt.show()
        

#feature scaling on numeric predictors
scaler = preprocessing.StandardScaler().fit(my_data.iloc[:, conti_idx])
my_data.loc[:, conti_idx] = scaler.transform(my_data.loc[:, conti_idx])

train = my_data.drop(labels ='default payment next month', axis = 1)
label = my_data.iloc[:, -1]

train_x = pd.get_dummies(train)


##### logistic regression ############
C_grid = {'C': [ 0.05, 0.1, 0.15, 0.2, 0.25] }
logistic_grid = GridSearchCV(LogisticRegression(solver = 'sag', max_iter= 500), 
                             C_grid, scoring= 'neg_log_loss',  n_jobs = -1)

logistic_grid.fit(train_x, label)

best_ls_grid= logistic_grid.best_estimator_
print(best_ls_grid)

best_lr = LogisticRegression(solver = 'sag', max_iter= 500, C = 0.1, n_jobs = -1, random_state=1)

print (np.mean(cross_val_score(best_lr, train_x, label, cv=5, scoring='neg_log_loss', n_jobs=-1)))
print (np.mean(cross_val_score(best_lr, train_x, label, cv=5, scoring='accuracy', n_jobs=-1)))


########## KNN ############
k_grid = {'n_neighbors': [25, 30, 35] }
knn_grid = GridSearchCV(KNeighborsClassifier(n_jobs = -1), 
                             k_grid, scoring= 'roc_auc',  n_jobs = -1)
knn_grid.fit(train_x, label)

best_knn_grid = knn_grid.best_estimator_
print(best_knn_grid)
best_knn = KNeighborsClassifier( n_neighbors = 30, n_jobs= -1, random_state=1)
print (np.mean(cross_val_score(best_knn, train_x, label, cv=5, scoring='accuracy', n_jobs=-1)))


########### LDA ####################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(solver ='lsqr')
param_grid = {
    'shrinkage' : [0.001, 0.003, 0.005, 0.008],
}
grid_search_lda = GridSearchCV(estimator = lda, param_grid = param_grid, 
                          cv = 3, n_jobs = -1)
grid_search_lda.fit(train_x, label)
best_grid_lda = grid_search_lda.best_estimator_
print(best_grid_lda)

best_lda = LinearDiscriminantAnalysis( shrinkage = 0.003 ,solver ='lsqr', random_state=1)
print (np.mean(cross_val_score(best_lda, train_x, label, cv=5, scoring='neg_log_loss', n_jobs=-1)))
print (np.mean(cross_val_score(best_lda, train_x, label, cv=5, scoring='accuracy', n_jobs=-1)))


########### QDA ###################
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis( )
param_grid = {
    'reg_param' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
}
grid_search_qda = GridSearchCV(estimator = qda, param_grid = param_grid, 
                          cv = 3, n_jobs = -1)
grid_search_qda.fit(train_x, label)
best_grid_qda = grid_search_qda.best_estimator_
print(best_grid_qda)

best_qda = QuadraticDiscriminantAnalysis( reg_param = 0.7, random_state=1)
print (np.mean(cross_val_score(best_qda, train_x, label, cv=5, scoring='neg_log_loss', n_jobs=-1)))
print (np.mean(cross_val_score(best_qda, train_x, label, cv=5, scoring='accuracy', n_jobs=-1)))



######### NN ####################
from sklearn.neural_network import MLPClassifier
param_grid = {
    'hidden_layer_sizes': [(6,), (7,), (8,), (9,), (10,), (11,)],
    'alpha': [0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25],
}

neural = MLPClassifier()
grid_search_nn = GridSearchCV(estimator = neural, param_grid = param_grid,
                              cv = 3, n_jobs = -1, verbose = 2, scoring= 'neg_log_loss')

# Fit the grid search to the data
grid_search_nn.fit(train_x, label)

best_grid_nn = grid_search_nn.best_estimator_
print(best_grid_nn)
best_nn = MLPClassifier(hidden_layer_sizes = (9,), alpha = 0.23, random_state=1)
print (np.mean(cross_val_score(best_nn, train_x, label, cv=5, scoring='neg_log_loss', n_jobs=-1)))
print (np.mean(cross_val_score(best_nn, train_x, label, cv=5, scoring='accuracy', n_jobs=-1)))



############ Random Forest ############
from sklearn.ensemble import RandomForestClassifier

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True, False],
    'max_depth': [ 35, 39, 43, 47],
    'max_features': [73, 77, 81, 85],
    'min_samples_leaf': [1],
    'min_samples_split': [10],
    'n_estimators': [200]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2, scoring= 'neg_log_loss')

# Fit the grid search to the data
grid_search_rf.fit(train_x, label)
best_grid_rf = grid_search_rf.best_estimator_

best_rf = RandomForestClassifier(bootstrap = True ,max_depth = 43, max_features = 81,
                                 min_samples_leaf = 1, min_samples_split= 10,
                                 n_estimators = 200, random_state=1)
print (np.mean(cross_val_score(best_rf, train_x, label, cv=5, scoring='neg_log_loss', n_jobs=-1)))
print (np.mean(cross_val_score(best_rf, train_x, label, cv=5, scoring='accuracy', n_jobs=-1)))


#############  LightGBM #################
from lightgbm import LGBMClassifier
lgb = LGBMClassifier()

# Create parameters to search
param_grid = {
    'learning_rate': [0.005],
    'n_estimators': [150, 200, 250, 300],
    'num_leaves': [45],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'random_state' : [501], # Updated from 'seed'
    'colsample_bytree' : [1],
    'subsample' : [0.85],
    'reg_alpha' : [0.6],
    'reg_lambda' : [0.4],
    }


# Create a based model
lgb = LGBMClassifier()
# Instantiate the grid search model
grid_search_lgb = GridSearchCV(estimator = lgb, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2, scoring= 'neg_log_loss')

# Fit the grid search to the data
grid_search_lgb.fit(train_x, label)
best_grid_lgb = grid_search_lgb.best_estimator_

best_lgb = LGBMClassifier(n_estimators = 300, num_leaves = 45, boosting_type = 'gbdt',
                          objective = 'binary', colsample_bytree = 1, subsample = 0.85,
                         reg_alpha = 0.6,  reg_lambda = 0.4, random_state=1)

print (np.mean(cross_val_score(lgb, train_x, label, cv=5, scoring='neg_log_loss', n_jobs=-1)))
print (np.mean(cross_val_score(lgb, train_x, label, cv=5, scoring='accuracy', n_jobs=-1)))


############ Voting ##############
from sklearn.ensemble import VotingClassifier
vot = VotingClassifier(estimators=[ ('lr', best_lr), ('nn', best_nn), ('lgb', lgb), 
                                   ('rf', best_rf), ('lda', best_lda)], voting='soft', random_state=1)

print (np.mean(cross_val_score(vot, train_x, label, cv=5, scoring='neg_log_loss', n_jobs=-1)))
print (np.mean(cross_val_score(vot, train_x, label, cv=5, scoring='accuracy', n_jobs=-1)))


########### Stacking #############

from vecstack import stacking
from sklearn import metrics

models = [lgb, best_nn, best_lr]
S_train, S_test = stacking(models,                     # list of models
                           train_x, label, train_x,   # data
                           regression=False,           # classification task (if you need 
                                                       #     regression - set to True)
                           mode='oof_pred_bag',        # mode: oof for train set, predict test 
                                                       #     set in each fold and vote
                           needs_proba=True,           # predict class labels (if you need 
                                                       #     probabilities - set to True) 
                           save_dir=None,              # do not save result and log (to save 
                                                       #     in current dir - set to '.')
                           metric= metrics.log_loss,           # metric: callable
                           n_folds=5,                  # number of folds
                           stratified=True,            # stratified split for folds
                           shuffle=True,               # shuffle the data
                           random_state=0,             # ensure reproducibility
                           verbose=2)                  # print all info


    

Stack_combined_features = np.concatenate((S_test[:, [0,2,4]], np.array(train_x)), axis = 1)


from sklearn.model_selection import GridSearchCV

C_grid = {'C': [0.05, 0.1, 0.15, 0.2] }
logistic_grid = GridSearchCV(LogisticRegression(solver = 'sag', max_iter= 500),  C_grid, scoring= 'neg_log_loss',  n_jobs = -1)
logistic_grid.fit(train_x, label)

best_random = logistic_grid.best_estimator_
print(logistic_grid.best_estimator_)

stk_estimator = LogisticRegression(solver = 'sag', C =0.1, n_jobs=-1, max_iter= 500, random_state = 1)

print (np.mean(cross_val_score(stk_estimator, train_x, label, cv=5, scoring='neg_log_loss', n_jobs=-1)))

C_combined_grid = {'C': [0.05, 0.1, 0.15, 0.2] }
logistic_combined_grid = GridSearchCV(LogisticRegression(solver = 'sag', max_iter= 1000),  C_grid, scoring= 'neg_log_loss',  n_jobs = -1)
logistic_combined_grid.fit(Stack_combined_features, label)

best_random = logistic_grid.best_estimator_
print(logistic_grid.best_estimator_)

stk_combined_estimator = LogisticRegression(solver = 'sag', C =0.1, n_jobs=-1,
                                            max_iter= 500, random_state = 1)

print (np.mean(cross_val_score(stk_combined_estimator, Stack_combined_features, 
                               label, cv=5, scoring='neg_log_loss', n_jobs=-1)))
print (np.mean(cross_val_score(stk_combined_estimator, Stack_combined_features, 
                               label, cv=5, scoring='accuracy', n_jobs=-1)))