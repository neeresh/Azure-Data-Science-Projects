import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import os

from azureml.core import Run

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Get the experiement run context
run = Run.get_context()

# Prepare the dataset
X_train = pd.read_csv('xtrain.csv')
run.log_list('Feature Names', list(X_train.columns))
run.log('X_train Shape', X_train.shape)

y_train = pd.read_csv('ytrain.csv')
run.log('y_train Shape', y_train.shape)

reg = 0.1

model = LogisticRegression(C = 1/reg, solver = 'liblinear')
model.fit(X_train, y_train)

fig = sns.countplot(x='Survived', data=X_train, hue='Pclass')
run.log_image('Target Feature', plot = fig)

# Calculate Accuracy
X_test = pd.read_csv('xtest.csv')
run.log('X_test Shape', X_test.shape)

y_test = pd.read_csv('ytest.csv')
run.log('y_test Shape', y_test.shape)

y_hat = model.predict(X_test)
acc = np.average(pd.DataFrame(y_hat, columns = ['Survived']) == y_test)
precision = precision_score(y_test, y_hat)
recall = recall_score(y_test, y_hat)

# Logging accuracy
run.log('Accuracy', np.float(acc))
run.log('Precision', precision)
run.log('Recall', recall)

# Plotting Precision-Recall Curve
fig = plot_precision_recall_curve(model, X_test, y_test)
run.log_image('Precision-Recall Curve', fig)

# Save the trained model
os.makedirs('outputs', exist_ok = True)
joblib.dump(value = model, filename = 'outputs/model.pkl')

run.complete()