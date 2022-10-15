
import argparse

# Getting the scripts
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type = float, dest = 'reg_rate', default = 0.01, help = 'Enter regularization rate')
parser.add_argument('--input-data', type = str, dest = 'training_dataset_id', help = 'training dataset')
args = parser.parse_args()

# Set regularization rate
reg = args.reg_rate

# Get the experiment run context
from azureml.core import Run 

run = Run.get_context()

# Load the diabetes dataset
diabetes = run.input_datasets['training_data'].to_pandas_dataframe()

# Seperate features and labels
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

# Split data into training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Train a logistic regression model
from sklearn.linear_model import LogisticRegression
import numpy as np

print('Training a logistic regression model with regularization rate of', reg)
run.log('Regularization Rate',  np.float(reg))
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
from sklearn.metrics import roc_auc_score

y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

# plot ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
fig = plt.figure(figsize=(6, 4))

# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')

# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

run.log_image(name = "ROC", plot = fig)
plt.show()

# Creating output folder
os.makedirs('practice-arena/outputs', exist_ok = True)

# Save the model
import joblib

joblib.dump(value = model, filename = 'practice-arena/outputs/diabetes_model.pkl')

run.complete()
