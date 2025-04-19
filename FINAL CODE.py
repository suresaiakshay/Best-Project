import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler , RobustScaler, MaxAbsScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer


#READING THE DATA FROM THE EXCELFILE
df = pd.read_csv("C:\\Users\\SURE SAI AKSHAY\\Desktop\\Brest cancer data set.csv")
df.head()

#GETTTING THE DATA FROM EXEL FILE
rows, columns = df.shape
print(f"Number of Rows : {rows}")
print(f"Number of Columns : {columns}")

df.info()
df.describe()
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)
df.head()
df['diagnosis'].value_counts()
sns.countplot(x='diagnosis', data=df)

# B for benignant
# M for malignant

#SPLITING THE DATA INTO FEATURES AND THE TARGET VARIABLE
x = df.drop(columns='diagnosis')
y = df['diagnosis']

#STANDARDIZATION OF DATA SET
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

#CONVERSION OF DIAGONOSIS VALUES(Like B,M) INTO NUMERIC VALUES(Like 0,1)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

#Splitting Data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=42)

#SEQUENTIAL MODEL IN KERAS
model = Sequential([
    Dense(16, activation='relu', input_dim=30),
    
    Dense(8, activation='relu'),
    
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#MODEL TRAINING
hist = model.fit(x_train, y_train, epochs=50, validation_split=0.2)

#EXTRACTING TRAINING HISTORY
tr_acc = hist.history['accuracy']
tr_loss = hist.history['loss']
val_acc = hist.history['val_accuracy']
val_loss = hist.history['val_loss']

epochs = [i+1 for i in range(len(tr_acc))]

plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs, tr_loss, 'r', label='Train Loss')
plt.plot(epochs, val_loss, 'g', label='Valid Loss')
plt.title('Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs, tr_acc, 'r', label='Train Accuracy')
plt.plot(epochs, val_acc, 'g', label='Valid Accuracy')
plt.title('Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

#to check whether the tumor is maligant or Benign
# Example feature values for new data
input_data = (11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115, 0.1495, 0.05888,
              0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758, 0.01168, 0.007445, 0.02406,
              0.001769, 12.98, 25.72, 82.98, 516.5, 0.1085, 0.08615, 0.05523, 0.03715,
              0.2433, 0.06563)

# Convert to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape to match the model input
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Apply the same scaling used on training data
input_data_std = scaler.transform(input_data_reshaped)

#PREDICT WITH SEQUENTIAL MODEL
y_predict = model.predict(x_test)
y_predict = (y_predict > 0.5)

#CONFUSION MATRIX FOR SEQUENTIAL MODEL
print('confusion matrix for ANN')
cm = confusion_matrix(y_test, y_predict)

#GRAPH FOR OBSERVING PREDICTED VS ACTUAL FOR SEQUENTIAL MODEL
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'], 
            yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("        ")
print(" ANN OUTPUT ")
print("------------")

#Accuracy of the data for sequential model
accuracy_ann = accuracy_score(y_test, y_predict)
print(f" ANN Accuracy: {accuracy_ann:.2f}")

# Make prediction for the tumor type with sequential model
prediction = model.predict(input_data_std)
prediction_label = (prediction > 0.5).astype(int)

# Output prediction for sequential model 
if prediction_label[0] == 0:
    print('The tumor is Benign')
else:
    print('The tumor is Malignant')

#CALCULATING MSE AND R^2 FOR ANN
mse_ann = mean_squared_error(y_test, y_predict)
r2_ann = r2_score(y_test, y_predict)
print(f"ANN MSE: {mse_ann:.4f}")
print(f"ANN R^2: {r2_ann:.4f}")


# DECISION TREE CLASSIFIER
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(x_train, y_train)

# PREDICT WITH DECISION TREE
y_predict_dt = dt_classifier.predict(x_test)

# CONFUSION MATRIX FOR DECISION TREE
print('confusion matrix for DT')
cm_dt = confusion_matrix(y_test, y_predict_dt)

#GRAPH FOR OBSERVING PREDICTED VS ACTUAL FOR DECISION TREE
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'], 
            yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("           ")
print(" DECISION TREE OUTPUT ")
print("----------------------")

# ACCURACY OF DECISION TREE
accuracy_dt = accuracy_score(y_test, y_predict_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.2f}")

# MAKE PREDICTION WITH DECISION TREE FOR THE TUMOR
prediction_dt = dt_classifier.predict(input_data_std)

# OUTPUT PREDICTION FOR DECISION TREE
if prediction_dt[0] == 0:
    print('The tumor is Benign')
else:
    print('The tumor is Malignant')

#CALCULATING MSE AND R^2
mse_dt = mean_squared_error(y_test, y_predict_dt)
r2_dt = r2_score(y_test, y_predict_dt)
print(f"Decision Tree MSE: {mse_dt:.4f}")
print(f"Decision Tree R^2: {r2_dt:.4f}")

# LOGISTIC REGRESSION CLASSIFIER
log_reg = LogisticRegression(random_state=42)
log_reg.fit(x_train, y_train)

# PREDICT WITH LOGISTIC REGRESSION
y_predict_lr = log_reg.predict(x_test)

# CONFUSION MATRIX FOR LOGISTIC REGRESSION
print('confusion matrix for LR')
cm_lr = confusion_matrix(y_test, y_predict_lr)

#GRAPH FOR OBSERVING PREDICTED VS ACTUAL FOR LOGISTIC REGRESSION
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'], 
            yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("           ")
print(" LOGISTIC REGRESSION OUTPUT ")
print("----------------------")

# ACCURACY OF LOGISTIC REGRESSION
accuracy_lr = accuracy_score(y_test, y_predict_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr:.2f}")

# MAKE PREDICTION WITH LOGISTIC REGRESSION FOR TUMOR TYPE
prediction_lr = log_reg.predict(input_data_std)

# OUTPUT PREDICTION FOR LOGISTIC REGRESSION
if prediction_lr[0] == 0:
    print('The tumor is Benign ')
else:
    print('The tumor is Malignant ')
  
#CALCULATING MSE AND R^2
mse_lr = mean_squared_error(y_test, y_predict_lr)
r2_lr = r2_score(y_test, y_predict_lr)
print(f"Logistic Regression MSE: {mse_lr:.4f}")
print(f"Logistic Regression R^2: {r2_lr:.4f}") 


# RANDOM FOREST CLASSIFIER
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(x_train, y_train)

# PREDICT WITH RANDOM FOREST
y_predict_rf = rf_classifier.predict(x_test)

# CONFUSION MATRIX FOR RANDOM FOREST
print('confusion matrix for RF')
cm_rf = confusion_matrix(y_test, y_predict_rf)

# GRAPH FOR OBSERVING PREDICTED VS ACTUAL FOR RANDOM FOREST
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'], 
            yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("           ")
print(" RANDOM FOREST OUTPUT ")
print("----------------------")

# ACCURACY OF RANDOM FOREST
accuracy_rf = accuracy_score(y_test, y_predict_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")

# MAKE PREDICTION WITH RANDOM FOREST FOR TUMOR TYPE
prediction_rf = rf_classifier.predict(input_data_std)

# OUTPUT PREDICTION FOR RANDOM FOREST
if prediction_rf[0] == 0:
    print('The tumor is Benign')
else:
    print('The tumor is Malignant')

#CALCULATING MSE AND R^2
mse_rf = mean_squared_error(y_test, y_predict_rf)
r2_rf = r2_score(y_test, y_predict_rf)
print(f"Random Forest MSE: {mse_rf:.4f}")
print(f"Random Forest R^2: {r2_rf:.4f}")

# GRADIENT BOOSTING CLASSIFIER
gb_classifier = GradientBoostingClassifier(random_state=42)
gb_classifier.fit(x_train, y_train)

# PREDICT WITH GRADIENT BOOSTING
y_predict_gb = gb_classifier.predict(x_test)

# CONFUSION MATRIX FOR GRADIENT BOOSTING
print('confusion matrix for GB')
cm_gb = confusion_matrix(y_test, y_predict_gb)

# GRAPH FOR OBSERVING PREDICTED VS ACTUAL FOR GRADIENT BOOSTING
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'], 
            yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("           ")
print(" GRADIENT BOOSTING OUTPUT ")
print("--------------------------")

# ACCURACY OF GRADIENT BOOSTING
accuracy_gb = accuracy_score(y_test, y_predict_gb)
print(f"Gradient Boosting Accuracy: {accuracy_gb:.2f}")

# MAKE PREDICTION WITH GRADIENT BOOSTING FOR TUMOR TYPE
prediction_gb = gb_classifier.predict(input_data_std)

# OUTPUT PREDICTION FOR GRADIENT BOOSTING
if prediction_gb[0] == 0:
    print('The tumor is Benign')
else:
    print('The tumor is Malignant')

#CALCULATING MSE AND R^2
mse_gb = mean_squared_error(y_test, y_predict_gb)
r2_gb = r2_score(y_test, y_predict_gb)
print(f"Gradient Boosting MSE: {mse_gb:.4f}")
print(f"Gradient Boosting R^2: {r2_gb:.4f}")



# XGBoost Classifier
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, random_state=42, eval_metric='logloss')
xgb_classifier.fit(x_train, y_train)

# Predict with XGBoost
y_predict_xgb = xgb_classifier.predict(x_test)

# Confusion Matrix for XGBoost
print('Confusion Matrix for XGBoost')
cm_xgb = confusion_matrix(y_test, y_predict_xgb)

# Graph for Observing Predicted vs Actual for XGBoost
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'], 
            yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("           ")
print(" XGBOOST OUTPUT ")
print("-----------------")

# Accuracy of XGBoost
accuracy_xgb = accuracy_score(y_test, y_predict_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")

# Make Prediction with XGBoost for the Tumor
prediction_xgb = xgb_classifier.predict(input_data_std)

# Output Prediction for XGBoost
if prediction_xgb[0] == 0:
    print('The tumor is Benign')
else:
    print('The tumor is Malignant')

# Calculating MSE and R^2 for XGBoost
mse_xgb = mean_squared_error(y_test, y_predict_xgb)
r2_xgb = r2_score(y_test, y_predict_xgb)
print(f"XGBoost MSE: {mse_xgb:.4f}")
print(f"XGBoost R^2: {r2_xgb:.4f}")

#VOTING CLASSIFIER
# Voting Classifier: Soft Voting (for probabilities)
voting_clf = VotingClassifier(
    estimators=[('decision_tree', dt_classifier), 
                ('logistic_regression', log_reg), 
                ('random_forest', rf_classifier),
                ('gradient_boosting', gb_classifier)],
    voting='soft'  # use 'hard' for majority voting without probabilities
)
# Fit ensemble model
voting_clf.fit(x_train, y_train)

# Predict with ensemble model
y_predict_voting = voting_clf.predict(x_test)

# Confusion Matrix for Ensemble Model
print('confusion matrix for VC')
cm_voting = confusion_matrix(y_test, y_predict_voting)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_voting, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'], 
            yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("           ")
print(" VOTING CLASSIFIER ")
print("--------------------------")

# Accuracy of ensemble model
accuracy_voting = accuracy_score(y_test, y_predict_voting)
print(f"Voting Classifier Accuracy: {accuracy_voting:.2f}")

# Make prediction for the tumor type with the voting ensemble
prediction_voting = voting_clf.predict(input_data_std)

# Output prediction for voting ensemble
if prediction_voting[0] == 0:
    print('The tumor is Benign')
else:
    print('The tumor is Malignant')


mse_voting = mean_squared_error(y_test, y_predict_voting)
r2_voting = r2_score(y_test, y_predict_voting)
print(f"voting classifier MSE: {mse_voting:.4f}")
print(f"voting classifier R^2: {r2_voting:.4f}")



data = pd.read_csv("C:\\Users\\SURE SAI AKSHAY\\Desktop\\Brest cancer data set.csv")

# Preprocessing (Adjust based on your data)
y = np.array(data['diagnosis'].tolist())  # Assuming 'diagnosis' is the target column
X = data.drop('diagnosis', axis=1).values

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
base_model1 = LogisticRegression(random_state=42)
base_model2 = DecisionTreeClassifier(random_state=42)

# Meta-model
meta_model = DecisionTreeClassifier()

stacking_clf = StackingClassifier(
    estimators=[
        ('lr', base_model1),
        ('dt', base_model2)
    ],
    final_estimator=meta_model
)

# Train the stacking classifier
stacking_clf.fit(X_train, y_train)

# Evaluate the stacking model
y_pred_stacking = stacking_clf.predict(X_test)
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)

# Confusion Matrix
cm_stacking = confusion_matrix(y_test, y_pred_stacking)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_stacking, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'], 
            yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("           ")
print(" STACKING CLASSIFIER ")
print("--------------------------")

# Accuracy of ensemble model
print(f'Stacking Classifier Accuracy: {accuracy_stacking:.2f}')

def predict_tumor_type(input_data):
    input_data_std = scaler.transform([input_data])  # Standardize input data
    prediction = stacking_clf.predict(input_data_std)
    
    # Map prediction to "Benign" or "Malignant"
    if prediction[0] == 0:
        return 'Malignant'
    else:
        return 'Bengin'

# Example usage
input_data = [0,11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115, 0.1495, 0.05888,
              0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758, 0.01168, 0.007445, 0.02406,
              0.001769, 12.98, 25.72, 82.98, 516.5, 0.1085, 0.08615, 0.05523, 0.03715,
              0.2433, 0.06563]

tumor_type = predict_tumor_type(input_data)
print(f"The tumor is {tumor_type}")

print(f"Stacking Classifier MSE: {mse_lr:.4f}")
print(f"Stacking Classifier R^2: {r2_lr:.4f}") 


# BOX PLOT FOR EACH FEATURE
for column in df.columns[1:]:
    plt.figure(figsize=(20, 10))
    sns.boxplot(x='diagnosis', y=column, data=df)
    plt.title(f'Box Plot of {column} Grouped by Diagnosis')
    plt.show()

#CORRELATION MATRIX
# DROPPING THE NON-NUMERIC COLUMNS
df_numeric = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# ENCODING THE DIAGNOSIS COLUMN
df['diagnosis_encoded'] = LabelEncoder().fit_transform(df['diagnosis'])

# UPDATING THE df_numeric 
df_numeric = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# CONVERTING ALL COLUMNS INTO NUMERIC
df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

# FILLING THE MISSING VALUES
df_numeric = df_numeric.fillna(0)  # or df_numeric.dropna()

# CALCULATING CORRELATION MATRIX
corr_matrix = df_numeric.corr()

# PLOTTING THE HEAT MAP FOR CORRELATION MATRIX
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, linecolor='black')
plt.title('Correlation Matrix of Features')
plt.show()

#ACCURACY FOR THE DIFFERENT MODELS
# Storing the accuracy values in a list
models = ['ANN', 'Decision Tree', 'Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Voting Classifier', 'Stacking Classifier', 'XGBoost']
accuracies = [accuracy_ann, accuracy_dt, accuracy_lr, accuracy_rf, accuracy_gb, accuracy_voting, accuracy_stacking, accuracy_xgb]  # Corresponding accuracies

# Creating a bar plot for accuracies
plt.figure(figsize=(10, 6))
plt.barh(models, accuracies, color=['skyblue', 'lightgreen', 'orange', 'purple', 'yellow', 'red', 'blue', 'pink'])

# Adding labels and title
plt.xlabel('Accuracy')
plt.ylabel('Models')
plt.title('Accuracy Comparison of Different Models')

# Adding accuracy values on the bars
for index, value in enumerate(accuracies):
    plt.text(value, index, f"{value:.2f}")

# Display the plot
plt.show()


#GETTING THE R^2 VALUES FOR THE INDIVIDUAL FEATURES
# Initialize a dictionary to store R² values
r2_values = {}

# Iterate over each feature (you can modify the feature selection as needed)
for column in df.columns:
    if column not in ['id', 'Unnamed: 32', 'diagnosis', 'diagnosis_encoded']:
        # Define the feature and the target variable
        X = df[[column]]
        y = df['diagnosis_encoded']
        
        # Add a constant to the model (for the intercept)
        X = sm.add_constant(X)
        
        # Fit the regression model
        model = sm.OLS(y, X).fit()
        
        # Get the R² value
        r2 = model.rsquared
        r2_values[column] = r2

# Convert the R² results to a DataFrame for easier visualization
r2_df = pd.DataFrame(list(r2_values.items()), columns=['Feature', 'R^2'])
print(r2_df)

# Plotting the R² values
plt.figure(figsize=(12, 6))
sns.barplot(x='R^2', y='Feature', data=r2_df, palette='viridis')
plt.title('R² Values for Individual Features')
plt.xlabel('R² Value')
plt.ylabel('Features')
plt.show()

#RADAR PLOT

features = df.columns.drop('diagnosis')  # Exclude the target 'TenYearCHD' column

# Normalize data for radar plot visualization
df_norm = df.copy()
df_norm[features] = (df[features] - df[features].min()) / (df[features].max() - df[features].min())

# Compute average feature values for each 'TenYearCHD' category
grouped_data = df_norm.groupby('diagnosis')[features].mean().reset_index()

# Prepare data for radar plot
categories = features
num_vars = len(categories)

# Function to create a radar plot
def create_radar_plot(values, title, color):
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]  # Repeat the first value to close the circle
    angles += angles[:1]

    plt.polar(angles, values, marker='o', color=color)
    plt.fill(angles, values, alpha=0.25, color=color)
    plt.title(title, size=15, y=1.1)
    plt.xticks(angles[:-1], categories, color='grey', size=12)
    plt.yticks(color='grey', size=10)
    plt.show()

# Plot radar chart for each 'DIAGNOSIS' category
colors = ['blue', 'green']
plt.figure(figsize=(8, 8))
for i, row in grouped_data.iterrows():
    create_radar_plot(row[features].values.tolist(), f'Radar Plot for DIAGNOSIS={row["diagnosis"]}', colors[i])


#SCATTER PLOT

plt.figure(figsize=(8, 6))
plt.scatter(df['radius_mean'], df['texture_mean'], c=df['diagnosis'].map({'M': 'red', 'B': 'green'}))
plt.xlabel('radius_mean')
plt.ylabel('diagnosis')
plt.title('Scatter Plot of radius_mean vs texture_mean')
plt.show()

#STACKED BAR GRAPH

diagnosis_counts = df['diagnosis'].value_counts()
diagnosis_counts.plot(kind='bar', stacked=True, color=['red', 'green'])
plt.title('Stacked Bar Chart for Diagnoses')
plt.ylabel('Count')
plt.show()

#BAR-GRAPH

df['diagnosis'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Bar Graph of Diagnosis Count')
plt.ylabel('Count')
plt.show()

#VIOLIN PLOT

sns.violinplot(x='diagnosis', y='radius_mean', data=df)  # Replace 'feature_name' with the actual feature column
plt.title('Violin Plot of radius_mean by Diagnosis')
plt.show()

sns.violinplot(x='diagnosis', y='texture_mean', data=df)  # Replace 'feature_name' with the actual feature column
plt.title('Violin Plot of radius_mean by Diagnosis')
plt.show()

sns.violinplot(x='diagnosis', y='perimeter_mean', data=df)  # Replace 'feature_name' with the actual feature column
plt.title('Violin Plot of radius_mean by Diagnosis')
plt.show()

#PIE CHART

diagnosis_counts = df['diagnosis'].value_counts()
diagnosis_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
plt.title('Pie Chart of Diagnosis Distribution')
plt.ylabel('')  # Remove ylabel
plt.show()


feature_contributions = df.drop(columns='diagnosis').abs().sum()  # Drop 'diagnosis' to consider only features

# Create a pie chart of feature contributions
plt.figure(figsize=(10, 7))
plt.pie(feature_contributions, labels=feature_contributions.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
plt.title('Contribution of Features to the Output')
plt.show()








