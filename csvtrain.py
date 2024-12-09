import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = "E:\ANIMAL_DISEASES_CLASSIFICATION\data.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Encode the 'Dangerous' column as binary (0 for 'No', 1 for 'Yes')
data['Dangerous'] = data['Dangerous'].map({'Yes': 1, 'No': 0})

# Check and handle missing values in the target column
data = data.dropna(subset=['Dangerous'])  # Remove rows where the target is NaN

# Combine the symptoms columns into a single feature set
symptoms_columns = ['symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5']
symptoms_data = data[symptoms_columns].apply(
    lambda row: pd.Series({symptom: 1 if symptom in row.values else 0 for symptom in row}),
    axis=1
)

# Prepare features and target variable
X = symptoms_data
y = data['Dangerous']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],           # Regularization parameter
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    'kernel': ['linear', 'rbf', 'poly']  # SVM kernel types
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Animal_dis_{best_params}")

# Train the SVM model with the best parameters
svm_model = SVC(**best_params, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Output the results
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)
