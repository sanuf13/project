# For data manipulation
import numpy as np
import pandas as pd
python
# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

# For data modeling
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

# For saving models
import pickle
# Load dataset into a dataframe
df0 = pd.read_csv("HR_capstone_dataset.csv")

# Display first few rows of the dataframe
df0.head()

# Gather basic information about the data
df0.info()

# Gather descriptive statistics about the data
df0.describe()# Display all column names
df0.columns# Rename columns as needed
df0 = df0.rename(columns={'Work_accident': 'work_accident',
'average_montly_hours': 'average_monthly_hours',
'time_spend_company': 'tenure',
'Department': 'department'})
# Display all column names after the update
df0.columns# Check for missing values
df0.isna().sum() 

# Check for missing values
df0.isna().sum()

# Inspect some rows containing duplicates as needed
df0[df0.duplicated()].head()

# Drop duplicates and save resulting dataframe in a new variable as needed
df1 = df0.drop_duplicates(keep='first')# Display first few rows of new dataframe as needed
df1.head()

# Create a boxplot to visualize distribution of `tenure` and detect any outliers
plt.figure(figsize=(6,6))
plt.title('Boxplot to detect outliers for tenure', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df1['tenure'])
plt.show()

# Determine the number of rows containing outliers
# Compute the 25th percentile value in `tenure`
percentile25 = df1['tenure'].quantile(0.25)
# Compute the 75th percentile value in `tenure`
percentile75 = df1['tenure'].quantile(0.75)
# Compute the interquartile range in `tenure`
iqr = percentile75 - percentile25# Define the upper limit and lower limit for non-outlier values in `tenure`
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)
# Identify subset of data containing outliers in `tenure`
outliers = df1[(df1['tenure'] > upper_limit) | (df1['tenure'] < lower_limit)]
# Count how many rows in the data contain outliers in `tenure`
print("Number of rows in the data containing outliers in `tenure`:",␣
,→len(outliers))

# Get numbers of people who left vs. stayed
print(df1['left'].value_counts())
print()
# Get percentages of people who left vs. stayed
print(df1['left'].value_counts(normalize=True))

# Create a plot as needed
# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))
# Create boxplot showing `average_monthly_hours` distributions for␣
number_project`, comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='average_monthly_hours', y='number_project',hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Monthly hours by number of projects', fontsize='14')
# Create histogram showing distribution of `number_project`, comparing␣
,→employees who stayed versus those who left
tenure_stay = df1[df1['left']==0]['number_project']
tenure_left = df1[df1['left']==1]['number_project']
sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge',shrink=2, ax=ax[1])
ax[1].set_title('Number of projects histogram', fontsize='14')
# Display the plots
plt.show()

# Get value counts of stayed/left for employees with 7 projects
df1[df1['number_project']==7]['left'].value_counts()

# Create a plot as needed
# Create scatterplot of `average_monthly_hours` versus `satisfaction_level`,␣comparing employees who stayed versus those who left
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='satisfaction_level',hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');

# Create a plot as needed
# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))
# Create boxplot showing distributions of `satisfaction_level` by tenure,comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='satisfaction_level', y='tenure', hue='left',orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Satisfaction by tenure', fontsize='14')
# Create histogram showing distribution of `tenure`, comparing employees who␣stayed versus those who left
tenure_stay = df1[df1['left']==0]['tenure']
tenure_left = df1[df1['left']==1]['tenure']
sns.histplot(data=df1, x='tenure', hue='left', multiple='dodge', shrink=5,ax=ax[1])
ax[1].set_title('Tenure histogram', fontsize='14')
plt.show();

# Calculate mean and median satisfaction scores of employees who left and those␣stayed
df1.groupby(['left'])['satisfaction_level'].agg([np.mean,np.median])

# Create a plot as needed
# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))
# Define short-tenured employees
tenure_short = df1[df1['tenure'] < 7]
# Define long-tenured employees
tenure_long = df1[df1['tenure'] > 6]
# Plot short-tenured histogram
sns.histplot(data=tenure_short, x='tenure', hue='salary', discrete=1,
hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.5,ax=ax[0])
ax[0].set_title('Salary histogram by tenure: short-tenured people',fontsize='14')
# Plot long-tenured histogram
sns.histplot(data=tenure_long, x='tenure', hue='salary', discrete=1,
hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.4,␣
,→ax=ax[1])
ax[1].set_title('Salary histogram by tenure: long-tenured people',␣
,→fontsize='14');

# Create a plot as needed
# Create scatterplot of `average_monthly_hours` versus `last_evaluation`
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='last_evaluation',hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');

 # Create a plot as needed
# Create plot to examine relationship between `average_monthly_hours` and␣`promotion_last_5years`
plt.figure(figsize=(16, 3))
sns.scatterplot(data=df1, x='average_monthly_hours', y='promotion_last_5years',hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by promotion last 5 years', fontsize='14');

# Display counts for each department
df1["department"].value_counts() 

# Create a plot as needed
# Create stacked histogram to compare department distribution of employees who␣left to that of employees who didn't
plt.figure(figsize=(11,8))
sns.histplot(data=df1, x='department', hue='left', discrete=1,hue_order=[0, 1], multiple='dodge', shrink=.5)
plt.xticks(rotation='45')
plt.title('Counts of stayed/left by department', fontsize=14); 

 # Create a plot as needed Plot a correlation heatmap
plt.figure(figsize=(16, 9))
heatmap = sns.heatmap(df0.corr(), vmin=-1, vmax=1, annot=True, cmap=sns,color_palette("vlag", as_cmap=True))
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12);

# Copy the dataframe
df_enc = df1.copy()
# Encode the `salary` column as an ordinal numeric category
df_enc['salary'] = (
df_enc['salary'].astype('category')
.cat.set_categories(['low', 'medium', 'high'])
.cat.codes
)
# Dummy encode the `department` column
df_enc = pd.get_dummies(df_enc, drop_first=False)
# Display the new dataframe
df_enc.head()

# Create a heatmap to visualize how correlated variables are
plt.figure(figsize=(8, 6))
sns.heatmap(df_enc[['satisfaction_level', 'last_evaluation', 'number_project','average_monthly_hours', 'tenure']].corr(), annot=True, cmap="crest")
plt.title('Heatmap of the dataset')
plt.show()

: # Create a stacked bart plot to visualize number of employees across␣department, comparing those who left with those who didn't
# In the legend, 0 (purple color) represents employees who did not leave, 1␣(red color) represents employees who left
pd.crosstab(df1['department'], df1['left']).plot(kind ='bar',color='mr')
plt.title('Counts of employees who left versus stayed across department'
plt.ylabel('Employee count')
plt.xlabel('Department')
plt.show()

: # Select rows without outliers in `tenure` and save resulting dataframe in a␣new variable
df_logreg = df_enc[(df_enc['tenure'] >= lower_limit) & (df_enc['tenure'] <=␣→upper_limit)]# Display first few rows of new dataframe
df_logreg.head()

# Isolate the outcome variable
y = df_logreg['left']
# Display first few rows of the outcome variable
y.head()

# Select the features you want to use in your model
X = df_logreg.drop('left', axis=1)
# Display the first few rows of the selected features
X.head()

# Split the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,stratify=y, random_state=42)
# Construct a logistic regression model and fit it to the training dataset
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train,y_train)

# Use the logistic regression model to get predictions on the test set
y_pred = log_clf.predict(X_test)
 # Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)
# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm,display_labels=log_clf.classes_)
# Plot confusion matrix
log_disp.plot(values_format='')
# Display plot
plt.show()
df_logreg['left'].value_counts(normalize=True)

# Create classification report for logistic regression model
target_names = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=target_names))

# Isolate the outcome variable
y = df_enc['left']
# Display the first few rows of `y`
y.head()

# Select the features
X = df_enc.drop('left', axis=1)
# Display the first few rows of `X`
X.head()

: # Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,→stratify=y, random_state=0)

# Instantiate model
tree = DecisionTreeClassifier(random_state=0)
# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
'min_samples_leaf': [2, 5, 1],
'min_samples_split': [2, 4, 6]}
# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
# Instantiate GridSearch
tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')

 #time
tree1.fit(X_train, y_train)

 # Check best parameters
tree1.best_params_

# Check best AUC score on CV
tree1.best_score_

def make_results(model_name:str, model_object, metric:str):

#create dictionary that maps input metric to actual metric name in␣GridSearchCV
metric_dict = {'auc': 'mean_test_roc_auc',
'precision': 'mean_test_precision',
'recall': 'mean_test_recall',
'f1': 'mean_test_f1',
'accuracy': 'mean_test_accuracy'}
# Get all the results from the CV and put them in a df
cv_results = pd.DataFrame(model_object.cv_results_)
# Isolate the row of the df with the max(metric) score
best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]
# Extract Accuracy, precision, recall, and f1 score from that row
auc = best_estimator_results.mean_test_roc_auc
f1 = best_estimator_results.mean_test_f1
recall = best_estimator_results.mean_test_recall
precision = best_estimator_results.mean_test_precision
accuracy = best_estimator_results.mean_test_accuracy
# Create table of results
table = pd.DataFrame()table = pd.DataFrame({'model': [model_name],
'precision': [precision],'recall': [recall],'F1': [f1],'accuracy': [accuracy],'auc': [auc]})
return table

 # Get all CV scores
tree1_cv_results = make_results('decision tree cv', tree1, 'auc')
tree1_cv_results

# Instantiate model
rf = RandomForestClassifier(random_state=0)
# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3,5, None],
'max_features': [1.0],
'max_samples': [0.7, 1.0],
'min_samples_leaf': [1,2,3],
'min_samples_split': [2,3,4],
'n_estimators': [300, 500],
}
# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
# Instantiate GridSearch
rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')

#time
rf1.fit(X_train, y_train) # --> Wall time: ~10min

# Define a path to the folder where you want to save the model
path = '/home/jovyan/work/'

def write_pickle(path, model_object, save_as:str):
with open(path + save_as + '.pickle', 'wb') as to_write:pickle.dump(model_object, to_write)
def read_pickle(path, saved_model_name:str):
with open(path + saved_model_name + '.pickle', 'rb') as to_read:
model = pickle.load(to_read)
return model
def read_pickle(path, saved_model_name:str):

with open(path + saved_model_name + '.pickle', 'rb') as to_read:
model = pickle.load(to_read)
return model

# Write pickle
write_pickle(path, rf1, 'hr_rf1')

# Read pickle
rf1 = read_pickle(path, 'hr_rf1')

# Check best AUC score on CV
rf1.best_score_

: # Check best params
rf1.best_params_

# Get all CV scores
rf1_cv_results = make_results('random forest cv', rf1, 'auc')
print(tree1_cv_results)print(rf1_cv_results)

def get_scores(model_name:str, model, X_test_data, y_test_data):

preds = model.best_estimator_.predict(X_test_data)
auc = roc_auc_score(y_test_data, preds)
accuracy = accuracy_score(y_test_data, preds)
precision = precision_score(y_test_data, preds)
recall = recall_score(y_test_data, preds)
f1 = f1_score(y_test_data, preds)
table = pd.DataFrame({'model': [model_name],
'precision': [precision],
'recall': [recall],'f1': [f1],'accuracy': [accuracy],'AUC': [auc]})
return table

 # Get predictions on test data
rf1_test_scores = get_scores('random forest1 test', rf1, X_test, y_test)
rf1_test_scores

: # Drop `satisfaction_level` and save resulting dataframe in new variable
df2 = df_enc.drop('satisfaction_level', axis=1)
# Display first few rows of new dataframe
df2.head(

# Create `overworked` column. For now, it's identical to average monthly hours.
df2['overworked'] = df2['average_monthly_hours']
# Inspect max and min average monthly hours values
print('Max hours:', df2['overworked'].max())
print('Min hours:', df2['overworked'].min())

 # Define `overworked` as working > 175 hrs/week
df2['overworked'] = (df2['overworked'] > 175).astype(int)
# Display first few rows of new column
df2['overworked'].head()

: # Drop the `average_monthly_hours` column
df2 = df2.drop('average_monthly_hours', axis=1)
# Display first few rows of resulting dataframe
df2.head()

: # Isolate the outcome variable
y = df2['left']
# Select the features
X = df2.drop('left', axis=1)

# Create test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,stratify=y, random_state=0)

# Instantiate model
tree = DecisionTreeClassifier(random_state=0)
# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
'min_samples_leaf': [2, 5, 1],
'min_samples_split': [2, 4, 6]}
# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
# Instantiate GridSearch
tree2 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')

%%time
tree2.fit(X_train, y_train)

# Check best params
tree2.best_params_

# Check best AUC score on CV
tree2.best_score_

 # Get all CV scores
tree2_cv_results = make_results('decision tree2 cv', tree2, 'auc')
print(tree1_cv_results)
print(tree2_cv_results)

# Instantiate model
rf = RandomForestClassifier(random_state=0)
# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3,5, None],
'max_features': [1.0],
'max_samples': [0.7, 1.0],
'min_samples_leaf': [1,2,3],'min_samples_split': [2,3,4],'n_estimators': [300, 500],}
# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
# Instantiate GridSearch
rf2 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')

#time
rf2.fit(X_train, y_train) # --> Wall time: 7min 5s

# Write pickle
write_pickle(path, rf2, 'hr_rf2')

 # Read in pickle
rf2 = read_pickle(path, 'hr_rf2')

 # Check best params
rf2.best_params_

 # Check best AUC score on CV
rf2.best_score_

 # Get all CV scores
rf2_cv_results = make_results('random forest2 cv', rf2, 'auc')
print(tree2_cv_results)
print(rf2_cv_results)

# Get predictions on test data
rf2_test_scores = get_scores('random forest2 test', rf2, X_test, y_test)
rf2_test_scores

 # Generate array of values for confusion matrix
preds = rf2.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=rf2.classes_)
# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=rf2.classes_)
disp.plot(values_format='');

 # Plot the tree
plt.figure(figsize=(85,20))
plot_tree(tree2.best_estimator_, max_depth=6, fontsize=14, feature_names=X.
,→columns,
class_names={0:'stayed', 1:'left'}, filled=True);
plt.show()

 #tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_,␣columns=X.columns)
tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_,
columns=['gini_importance'],
index=X.columns)
tree2_importances = tree2_importances.sort_values(by='gini_importance',ascending=False)
# Only extract the features with importances > 0
tree2_importances = tree2_importances[tree2_importances['gini_importance'] != 0]
tree2_importances

sns.barplot(data=tree2_importances, x="gini_importance", y=tree2_importances.index, orient='h')
plt.title("Decision Tree: Feature Importances for Employee Leaving",fontsize=12)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.show()

 # Get feature importances
feat_impt = rf2.best_estimator_.feature_importances_
# Get indices of top 10 features
ind = np.argpartition(rf2.best_estimator_.feature_importances_, -10)[-10:]
# Get column labels of top 10 features
feat = X.columns[ind]
# Filter `feat_impt` to consist of top 10 feature importances
feat_impt = feat_impt[ind]
y_df = pd.DataFrame({"Feature":feat,"Importance":feat_impt})
y_sort_df = y_df.sort_values("Importance")
fig = plt.figure()
ax1 = fig.add_subplot(111)
y_sort_df.plot(kind='barh',ax=ax1,x="Feature",y="Importance") # Get feature importances
feat_impt = rf2.best_estimator_.feature_importances_
# Get indices of top 10 features
ind = np.argpartition(rf2.best_estimator_.feature_importances_, -10)[-10:]
# Get column labels of top 10 features
feat = X.columns[ind]
# Filter `feat_impt` to consist of top 10 feature importances
feat_impt = feat_impt[ind]
y_df = pd.DataFrame({"Feature":feat,"Importance":feat_impt})
y_sort_df = y_df.sort_values("Importance")
fig = plt.figure()
ax1 = fig.add_subplot(111)
y_sort_df.plot(kind='barh',ax=ax1,x="Feature",y="Importance")ax1.set_title("Random Forest: Feature Importances for Employee Leaving",fontsize=12)
ax1.set_ylabel("Feature")
ax1.set_xlabel("Importance")
plt.show()

