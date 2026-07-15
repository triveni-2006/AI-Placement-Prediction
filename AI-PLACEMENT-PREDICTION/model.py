import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Remove unnecessary columns
data = data.drop(['student_id', 'company_type', 'job_role', 'salary_lpa','resume_score','skill_score'], axis=1)

# Remove rows where target is missing
data = data.dropna(subset=['placed'])

# Convert categorical columns
data = pd.get_dummies(data)

# Split input and output
X = data.drop('placed', axis=1)
y = data['placed']

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save model
pickle.dump(model, open("placement_model.pkl", "wb"))

print("Model trained successfully!")