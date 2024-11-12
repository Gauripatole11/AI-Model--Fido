```python

```


```python

```


```python
#Import Libraries

import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```


```python
#Connect to MongoDB and Extract Data

MONGODB_URI = "mongodb+srv://new:new@newcluster.qg6ht.mongodb.net/fidoKeyDB"

client = MongoClient(MONGODB_URI)
db = client['fidoKeyDB']

# Extract collections
users = db['users']
securitykeys = db['securitykeys']
keyassignments = db['keyassignments']

# Load data into DataFrames
users_df = pd.DataFrame(list(users.find()))
keys_df = pd.DataFrame(list(securitykeys.find()))
assignments_df = pd.DataFrame(list(keyassignments.find()))

```


```python
#Merge DataFrames(assignments, users, keys) on relevant columns to create a unified Df

assignments_df.columns = assignments_df.columns.str.strip().str.lower()
users_df.columns = users_df.columns.str.strip().str.lower()

# Ensure 'userId' in assignments_df matches '_id' in users_df
assignments_df['userid'] = assignments_df['userid'].astype(str)
users_df['_id'] = users_df['_id'].astype(str)

# Merge assignments with users on 'userid' and '_id'
merged_df = assignments_df.merge(users_df, left_on='userid', right_on='_id', how='left', suffixes=('_assign', '_user'))

# Merge with keys data
merged_df = merged_df.merge(keys_df, left_on='keyid', right_on='_id', how='left', suffixes=('_user', '_key'))

```


```python
#Handle missing values

print("Missing values in merged DataFrame:")
print(merged_df.isnull().sum())

# Fill missing values in 'lastUsed' with a placeholder date
merged_df['lastUsed'].fillna(pd.Timestamp('1900-01-01'), inplace=True)

```

    Missing values in merged DataFrame:
    _id_assign            0
    userid                0
    keyid                 0
    assignedby            0
    status_assign         0
    assignedat            0
    createdat_assign      0
    updatedat             0
    __v_assign            0
    _id_user              0
    email                 0
    password             13
    firstname             0
    lastname              0
    role                  0
    department            0
    employeeid            0
    status_user           0
    mfaenabled            0
    fidoregistered        0
    createdat_user        0
    mfabackupcodes        0
    __v_user              0
    lastlogin             0
    mfasecret            13
    _id                   0
    serialNumber          0
    credentialId          0
    publicKey             0
    aaguid                0
    status                0
    signCount             0
    revokedBy            10
    deviceName            0
    lastUsed              0
    createdAt             0
    updatedAt             0
    __v                   0
    revokedAt            10
    currentAssignment     0
    userHandle            3
    dtype: int64
    

    C:\Users\Gauri Patole\AppData\Local\Temp\ipykernel_28236\2176258993.py:7: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      merged_df['lastUsed'].fillna(pd.Timestamp('1900-01-01'), inplace=True)
    


```python
#Feature Engineering

#convert date columns to datetime obj,creates new feature days_since_assignment, 
#which calculates the time since the assignment was made.

# Convert date columns to datetime
merged_df['assignedat'] = pd.to_datetime(merged_df['assignedat'], errors='coerce')
merged_df['createdat_user'] = pd.to_datetime(merged_df['createdat_user'], errors='coerce')
merged_df['lastUsed'] = pd.to_datetime(merged_df['lastUsed'], errors='coerce')

# Calculate 'days_since_assignment'
merged_df['days_since_assignment'] = (datetime.now() - merged_df['assignedat']).dt.days

```


```python
#Define Target Var ('key_assigned')

#Setting to 1 if an assignment else set to 0

merged_df['key_assigned'] = merged_df['assignedat'].notnull().astype(int)

assigned_keys_count = merged_df[merged_df['key_assigned'] == 1].shape[0]
unassigned_keys_count = merged_df[merged_df['key_assigned'] == 0].shape[0]

print("Assigned keys count:", assigned_keys_count)
print("Unassigned keys count:", unassigned_keys_count)

```

    Assigned keys count: 13
    Unassigned keys count: 0
    


```python
#Encode Categorial Variables

# Initialize label encoders

le_role = LabelEncoder()
le_department = LabelEncoder()
le_status = LabelEncoder()

# Encode 'role', 'department', and 'status' columns

merged_df['role_encoded'] = le_role.fit_transform(merged_df['role'])
merged_df['department_encoded'] = le_department.fit_transform(merged_df['department'])
merged_df['status_encoded'] = le_status.fit_transform(merged_df['status_user'])

# Save label encoders for future use

joblib.dump(le_role, 'le_role.pkl')
joblib.dump(le_department, 'le_department.pkl')
joblib.dump(le_status, 'le_status.pkl')

```




    ['le_status.pkl']




```python
#Prepare Training Data

# Define feature columns
feature_cols = ['role_encoded', 'department_encoded', 'status_encoded', 'days_since_assignment']

# Prepare the feature matrix (X) and target vector (y)
X = merged_df[feature_cols]
y = merged_df['key_assigned']

# Display the feature matrix and target vector
print("Feature Matrix (X):")
print(X.head())
print("Target Variable (y):")
print(y.head())

```

    Feature Matrix (X):
       role_encoded  department_encoded  status_encoded  days_since_assignment
    0             0                   0               0                      2
    1             0                   0               0                      2
    2             0                   0               0                      1
    3             0                   2               0                      1
    4             0                   0               0                      1
    Target Variable (y):
    0    1
    1    1
    2    1
    3    1
    4    1
    Name: key_assigned, dtype: int64
    


```python
#Split teh Data -> Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the sizes of the training and testing sets

print(f"Training Set Size: {X_train.shape[0]}")
print(f"Testing Set Size: {X_test.shape[0]}")

```

    Training Set Size: 10
    Testing Set Size: 3
    


```python
#Train teh Model

from sklearn.ensemble import RandomForestClassifier

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;RandomForestClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(random_state=42)</pre></div> </div></div></div></div>




```python
#Evaluate teh model

# Predict target var on test set
y_pred = model.predict(X_test)

# Evaluate model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save evaluation metrics 
evaluation_metrics = {
    'accuracy': accuracy,
    'classification_report': classification_report(y_test, y_pred),
    'confusion_matrix': confusion_matrix(y_test, y_pred)
}
joblib.dump(evaluation_metrics, 'evaluation_metrics.pkl')

```

    Accuracy: 1.0000
    Classification Report:
                  precision    recall  f1-score   support
    
               1       1.00      1.00      1.00         3
    
        accuracy                           1.00         3
       macro avg       1.00      1.00      1.00         3
    weighted avg       1.00      1.00      1.00         3
    
    Confusion Matrix:
    [[3]]
    

    C:\Users\Gauri Patole\ai_env\Lib\site-packages\sklearn\metrics\_classification.py:409: UserWarning: A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.
      warnings.warn(
    C:\Users\Gauri Patole\ai_env\Lib\site-packages\sklearn\metrics\_classification.py:409: UserWarning: A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.
      warnings.warn(
    




    ['evaluation_metrics.pkl']




```python
#Save the model

import joblib

# Save the trained model to a file
joblib.dump(model, 'trained_model.pkl')

print("Model saved successfully.")
```

    Model saved successfully.
    


```python
# Data Visualization

from dash import dash_table, dcc, html
import dash
import pandas as pd

# Sample data
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Role': ['Manager', 'Developer', 'Designer'],
    'Assigned Keys': [5, 3, 8],
    'Unassigned Keys': [1, 2, 0]
}
df = pd.DataFrame(data)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout with the DataTable
app.layout = html.Div([
    html.H1('Key Assignment Table'),
    dash_table.DataTable(
        id='key-assignment-table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'height': '400px', 'overflowY': 'auto'},
        style_cell={'textAlign': 'center'},
    ),
])

# Ensure all the setup is done before running the server
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

```



<iframe
    width="100%"
    height="650"
    src="http://127.0.0.1:8050/"
    frameborder="0"
    allowfullscreen

></iframe>




```python
#Data Visualization

from dash import dash_table, dcc, html
import dash
import pandas as pd

# Sample data
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Role': ['Manager', 'Developer', 'Designer'],
    'Assigned Keys': [5, 3, 8],
    'Unassigned Keys': [1, 2, 0]
}
df = pd.DataFrame(data)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout with the DataTable
app.layout = html.Div([
    html.H1('Key Assignment Table'),
    dash_table.DataTable(
        id='key-assignment-table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'height': '400px', 'overflowY': 'auto'},
        style_cell={'textAlign': 'center'},
    ),
])

# Ensure all the setup is done before running the server
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

```



<iframe
    width="100%"
    height="650"
    src="http://127.0.0.1:8050/"
    frameborder="0"
    allowfullscreen

></iframe>




```python

```


```python

```
