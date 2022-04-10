import pandas as pd
# Load libraries
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa,
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}


# import the dataset
df = pd.read_csv('prad_msk_stopsack_2021_clinical_data.tsv', sep='\t')
# print(data)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# more options can be specified also
df.columns = df.columns.str.replace(' ', '_')
# print(df.columns)
df.drop(['Patient_ID','Sample_ID', 'Cancer_Type_Detailed', 'Disease_Extent_At_Time_IMPACT_Was_Sent',
         'Fraction_Genome_Altered', 'Gene_Panel', 'Oncotree_Code',
         'Prostate-specific_antigen','Sample_Class', 'Number_of_Samples_Per_Patient',
         'Sample_Type', 'Tumor_Sample_Histology', '8q_arm',
         'Study_ID', 'Cancer_Type', 'Somatic_Status', 'Gleason_Grade',
         'Age_At_Procurement'], axis=1, inplace=True)

df = df.dropna()  # clean data
# print(df.describe())
# print(df.columns)
# RACE
df['Race_Category'].replace(['Black', 'White', 'Asian'],
                        [0, 1, 2], inplace=True)
# Race:
# Black = 0
# White = 1
# Asian = 2
df['Smoking'].replace(['Never', 'Current', 'Former'],
                            [0, 1, 2], inplace=True)
# Smoking:
# Never = 0
# Current = 1
# Former = 2
df["Patient's_Vital_Status"].replace(['Dead', 'Alive'],
                      [0, 1], inplace=True)
# Status:
# Dead = 0
# Alive = 1
df["M_Stage"].replace(['M0', 'M1'], [0, 1], inplace=True)
# Status:
# M0 = 0
# M1 = 1
df["Tissue_Type"].replace(['Liver', 'Lymph Node', 'Bone', 'Prostate', 'Other soft tissue', 'Lung'],
                                     [0, 1, 2, 3, 4, 5], inplace=True)
# Tissue:
# Liver = 0
# Lymph_Node = 1
# Bone = 2
# Prostate = 3
# Other_soft_tissue = 4
# Lung = 5

# print(df)
print(df.describe())

# standard scaler
df_ready = df.copy()
scaler = StandardScaler()
# num_cols = ['Age_at_Diagnosis', 'Mutation_Count', 'Race_Category', 'Tissue_Type']
num_cols = ['Age_at_Diagnosis', 'Mutation_Count', 'TMB_(nonsynonymous)']
df_ready[num_cols] = scaler.fit_transform(df[num_cols])
# print(df_ready)

# print(df_ready.groupby("Patient's_Vital_Status").size())  # print number of dead or alive

Dead = df_ready[df_ready["Patient's_Vital_Status"] == 0]
Alive = df_ready[df_ready["Patient's_Vital_Status"] == 1]
# print(Dead.shape)
# print(Alive.shape)


# using upsampling
# dead_upsample = resample(Dead,
#                          replace=True,
#                          n_samples=len(Alive),
#                          random_state=42)
# # print(dead_upsample.shape)
# data_sampled = pd.concat([Alive, dead_upsample])
# print(data_sampled.shape)

# using Synthetic Minority Oversampling Technique to upsample
X_train_smote = df_ready.drop(["Patient's_Vital_Status"], axis=1)
Y_train_smote = df_ready["Patient's_Vital_Status"]
# print(X_train_smote.shape, Y_train_smote.shape)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
X_train_res, Y_train_res = sm.fit_resample(X_train_smote, Y_train_smote.ravel())
# print(X_train_res.shape, Y_train_res.shape)
# print(len(Y_train_res[Y_train_res == 0]), len(Y_train_res[Y_train_res==1]))
# print(X_train_res)  # dataset
# print(Y_train_res)  # dead or alive

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_res, Y_train_res,
                                                    shuffle=True,
                                                    test_size=0.2,
                                                    random_state=1)

# Show the Training and Testing Data
# print('Shape of training feature:', X_train.shape)
# print('Shape of testing feature:', X_test.shape)
# print('Shape of training label:', y_train.shape)
# print('Shape of training label:', y_test.shape)


from sklearn import tree

# Building Decision Tree model
dtc = tree.DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

# Building Random Forest model
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

from sklearn.naive_bayes import GaussianNB

# Building Naive Bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier

# Building KNN model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Evaluate Model
knn_eval = evaluate_model(knn, X_test, y_test)

# Print result
print('K neighbours')
print('Accuracy:', knn_eval['acc'])
print('Precision:', knn_eval['prec'])
print('Recall:', knn_eval['rec'])
print('F1 Score:', knn_eval['f1'])
print('Cohens Kappa Score:', knn_eval['kappa'])
print('Area Under Curve:', knn_eval['auc'])
print('Confusion Matrix:\n', knn_eval['cm'])

# Evaluate Model
nb_eval = evaluate_model(nb, X_test, y_test)

# Print result
print('naive bayes')
print('Accuracy:', nb_eval['acc'])
print('Precision:', nb_eval['prec'])
print('Recall:', nb_eval['rec'])
print('F1 Score:', nb_eval['f1'])
print('Cohens Kappa Score:', nb_eval['kappa'])
print('Area Under Curve:', nb_eval['auc'])
print('Confusion Matrix:\n', nb_eval['cm'])

# Evaluate Model
dtc_eval = evaluate_model(dtc, X_test, y_test)

# Print result
print('Decision Tree')
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print('Cohens Kappa Score:', dtc_eval['kappa'])
print('Area Under Curve:', dtc_eval['auc'])
print('Confusion Matrix:\n', dtc_eval['cm'])

# Evaluate Model
rf_eval = evaluate_model(rf, X_test, y_test)

# Print result
print('Random forest')
print('Accuracy:', rf_eval['acc'])
print('Precision:', rf_eval['prec'])
print('Recall:', rf_eval['rec'])
print('F1 Score:', rf_eval['f1'])
print('Cohens Kappa Score:', rf_eval['kappa'])
print('Area Under Curve:', rf_eval['auc'])
print('Confusion Matrix:\n', rf_eval['cm'])


# plotting graph to compare algorithms
# Intitialize figure with two plots
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor('white')

# First plot
## set bar size
barWidth = 0.2
dtc_score = [dtc_eval['acc'], dtc_eval['prec'], dtc_eval['rec'], dtc_eval['f1'], dtc_eval['kappa']]
rf_score = [rf_eval['acc'], rf_eval['prec'], rf_eval['rec'], rf_eval['f1'], rf_eval['kappa']]
nb_score = [nb_eval['acc'], nb_eval['prec'], nb_eval['rec'], nb_eval['f1'], nb_eval['kappa']]
knn_score = [knn_eval['acc'], knn_eval['prec'], knn_eval['rec'], knn_eval['f1'], knn_eval['kappa']]

## Set position of bar on X axis
r1 = np.arange(len(dtc_score))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

## Make the plot
ax1.bar(r1, dtc_score, width=barWidth, edgecolor='white', label='Decision Tree')
ax1.bar(r2, rf_score, width=barWidth, edgecolor='white', label='Random Forest')
ax1.bar(r3, nb_score, width=barWidth, edgecolor='white', label='Naive Bayes')
ax1.bar(r4, knn_score, width=barWidth, edgecolor='white', label='K-Nearest Neighbors')

## Configure x and y axis
ax1.set_xlabel('Metrics', fontweight='bold')
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
ax1.set_xticks([r + (barWidth * 1.5) for r in range(len(dtc_score))], )
ax1.set_xticklabels(labels)
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_ylim(0, 1)

## Create legend & title
ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
ax1.legend()

# Second plot
## Comparing ROC Curve
ax2.plot(dtc_eval['fpr'], dtc_eval['tpr'], label='Decision Tree, auc = {:0.5f}'.format(dtc_eval['auc']))
ax2.plot(rf_eval['fpr'], rf_eval['tpr'], label='Random Forest, auc = {:0.5f}'.format(rf_eval['auc']))
ax2.plot(nb_eval['fpr'], nb_eval['tpr'], label='Naive Bayes, auc = {:0.5f}'.format(nb_eval['auc']))
ax2.plot(knn_eval['fpr'], knn_eval['tpr'], label='K-Nearest Nieghbor, auc = {:0.5f}'.format(knn_eval['auc']))

## Configure x and y axis
ax2.set_xlabel('False Positive Rate', fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontweight='bold')

## Create legend & title
ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend(loc=4)

# plt.show()

# model optimization using cross validation
from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search
param_grid = {
    'max_depth': [50, 80, 100],
    'max_features': [2, 3, 4],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 300, 500, 750, 1000]
}

# Create a base model
rf_grids = RandomForestClassifier(random_state=0)

# Initiate the grid search model
grid_search = GridSearchCV(estimator=rf_grids, param_grid=param_grid, scoring='recall',
                           cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)


# Select best model with best fit
best_grid = grid_search.best_estimator_

# Evaluate Model
best_grid_eval = evaluate_model(best_grid, X_test, y_test)

# Print result
print('Accuracy:', best_grid_eval['acc'])
print('Precision:', best_grid_eval['prec'])
print('Recall:', best_grid_eval['rec'])
print('F1 Score:', best_grid_eval['f1'])
print('Cohens Kappa Score:', best_grid_eval['kappa'])
print('Area Under Curve:', best_grid_eval['auc'])
print('Confusion Matrix:\n', best_grid_eval['cm'])


# plotting graph to compare Random fores and random forest after cross validation
# Intitialize figure with two plots
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor('white')

# First plot
## set bar size
barWidth = 0.2
rf_score = [rf_eval['acc'], rf_eval['prec'], rf_eval['rec'], rf_eval['f1'], rf_eval['kappa']]
best_grid_score = [best_grid_eval['acc'], best_grid_eval['prec'], best_grid_eval['rec'], best_grid_eval['f1'], best_grid_eval['kappa']]

## Set position of bar on X axis
r1 = np.arange(len(rf_score))
r2 = [x + barWidth for x in r1]

## Make the plot
ax1.bar(r1, rf_score, width=barWidth, edgecolor='white', label='Random Forest (Base Line)')
ax1.bar(r2, best_grid_score, width=barWidth, edgecolor='white', label='Random Forest (Optimized)')

## Add xticks on the middle of the group bars
ax1.set_xlabel('Metrics', fontweight='bold')
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
ax1.set_xticks([r + (barWidth * 0.5) for r in range(len(dtc_score))], )
ax1.set_xticklabels(labels)
ax1.set_ylabel('Score', fontweight='bold')
# ax1.set_ylim(0, 1)

## Create legend & Show graphic
ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
ax1.legend()

# Second plot
## Comparing ROC Curve
ax2.plot(rf_eval['fpr'], rf_eval['tpr'], label='Random Forest, auc = {:0.5f}'.format(rf_eval['auc']))
ax2.plot(best_grid_eval['fpr'], best_grid_eval['tpr'], label='Random Forest, auc = {:0.5f}'.format(best_grid_eval['auc']))

ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.set_xlabel('False Positive Rate', fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontweight='bold')
ax2.legend(loc=4)

plt.show()

print('Change of {:0.2f}% on accuracy.'.format(100 * ((best_grid_eval['acc'] - rf_eval['acc']) / rf_eval['acc'])))
print('Change of {:0.2f}% on precision.'.format(100 * ((best_grid_eval['prec'] - rf_eval['prec']) / rf_eval['prec'])))
print('Change of {:0.2f}% on recall.'.format(100 * ((best_grid_eval['rec'] - rf_eval['rec']) / rf_eval['rec'])))
print('Change of {:0.2f}% on F1 score.'.format(100 * ((best_grid_eval['f1'] - rf_eval['f1']) / rf_eval['f1'])))
print('Change of {:0.2f}% on Kappa score.'.format(100 * ((best_grid_eval['kappa'] - rf_eval['kappa']) / rf_eval['kappa'])))
print('Change of {:0.2f}% on AUC.'.format(100 * ((best_grid_eval['auc'] - rf_eval['auc']) / rf_eval['auc'])))


from joblib import dump, load

# Saving model
dump(rf, 'Origin/cancer_race_classification.joblib')
dump(grid_search, 'Origin/cancer_race_gridsearch.joblib')

# saved_model = load('cancer_race_classification.joblib')
#
# print("After save:",saved_model.predict(X_test))