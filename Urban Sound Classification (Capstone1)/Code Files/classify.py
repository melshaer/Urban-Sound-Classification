import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef

df = pd.read_csv('complete_features_dataset.csv')
del(df['Unnamed: 0'])
del(df['ID'])

# Prepare X and y from df
encoder = LabelEncoder()
y = encoder.fit_transform(df['Class'])
scalar = StandardScaler()
X = scalar.fit_transform(np.array(df.iloc[:, :-1], dtype=float))

# Split dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifier and fit training data
model1 = RandomForestClassifier(bootstrap=True)
model1.fit(X_train, y_train)

model2 = KNeighborsClassifier(n_neighbors=10)
model2.fit(X_train, y_train)

model_dummy = DummyClassifier(strategy='stratified', random_state=42)
model_dummy.fit(X_train, y_train)

# Classification Accuracy
sc1 = model1.score(X_test, y_test)
sc2 = model2.score(X_test, y_test)
sc_dummy = model_dummy.score(X_test, y_test)
print('Classification Accuracy of RF Trees is {0}'.format(sc1))
print('Classification Accuracy of K-NN is {0}'.format(sc2))
print('Classification Accuracy of the Dummy Stratified Classifier is {0}'.format(sc_dummy))

# Classification metrics and confusion matrix
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred_dummy = model_dummy.predict(X_test)

print(confusion_matrix(y_test, y_pred1))
print(classification_report(y_test, y_pred1))

print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))

print(confusion_matrix(y_test, y_pred_dummy))
print(classification_report(y_test, y_pred_dummy))

# Cross-validation
cv_results1 = cross_val_score(model1, X, y, cv=5)
cv_results2 = cross_val_score(model2, X, y, cv=5)
cv_results_dummy = cross_val_score(model_dummy, X, y, cv=5)

print(cv_results1)
print(cv_results2)
print(cv_results_dummy)
print('Mean accuracy score after cross-validation of RF Trees is {0}'.format(np.mean(cv_results1)))
print('Mean accuracy score after cross-validation of K-NN is {0}'.format(np.mean(cv_results2)))
print('Mean accuracy score after cross-validation of the Dummy Stratified Classifier is {0}'.format(np.mean(cv_results_dummy)))


# Random Forest Classifier parameters
params = model1.get_params()

# Decision Trees
indicator, n_nodes_ptr = model1.decision_path(X_train)
leaves_train = model1.apply(X_train)
leaves_test = model1.apply(X_test)

# Gini purity importance for each feature
gini = model1.feature_importances_

# Matthew's Correlation Coefficient (Phi Coefficient)
mcc1 = matthews_corrcoef(y_test, y_pred1)
mcc2 = matthews_corrcoef(y_test, y_pred2)
mcc_dummy = matthews_corrcoef(y_test, y_pred_dummy)
print('MCC for Random Forest Model is {0}'.format(mcc1))
print('MCC for K-NN is {0}'.format(mcc2))
print('MCC for the Dummy Stratified Classifier is {0}'.format(mcc_dummy))