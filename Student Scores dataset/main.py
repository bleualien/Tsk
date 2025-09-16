# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 2: Load dataset
df = pd.read_csv('student-scores.csv')

# Step 3: List of subjects to predict
subjects = ['math_score', 'history_score', 'physics_score', 
            'chemistry_score', 'biology_score', 'english_score', 'geography_score']

# Step 4: Ask user for info
first_name = input("Enter your first name: ")
last_name = input("Enter your last name: ")
gender = input("Enter your gender: ")
weekly_hours = float(input("Enter weekly self study hours: "))

# Step 4a: Ensure minimum 7 hours
if weekly_hours < 7:
    print("You should study at least 7 hours. Prediction will assume 7 hours.")
    weekly_hours = 7

student_info = {
    'first_name': first_name,
    'last_name': last_name,
    'gender': gender,
    'weekly_self_study_hours': weekly_hours
}

# Step 5: Function to train model and predict a subject
def predict_subject(subject, hours):
    X = df[['weekly_self_study_hours']]
    y = df[subject]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict marks (pass DataFrame with column name)
    pred = model.predict(pd.DataFrame({'weekly_self_study_hours':[hours]}))[0]
    
    # Evaluate MSE
    mse = mean_squared_error(y_test, model.predict(X_test))
    
    return round(pred, 2), round(mse, 2)

# Step 6: Predict all subjects and store results
predicted_scores = {
    'First Name': first_name,
    'Last Name': last_name,
    'Gender': gender,
    'Weekly Self Study Hours': weekly_hours
}

mse_dict = {}

for subject in subjects:
    pred, mse = predict_subject(subject, weekly_hours)
    predicted_scores[subject.replace('_score','').title()] = pred
    mse_dict[subject.replace('_score','').title()] = mse

# Step 7: Convert to DataFrame for display
result_table = pd.DataFrame([predicted_scores])

# Step 8: Display results
print("\n===== Predicted Marks for the Student =====")
print(result_table.to_string(index=False))

# Optional: Display MSE for each subject
print("\n===== Model Mean Squared Error (MSE) for Each Subject =====")
for subj, mse_val in mse_dict.items():
    print(f"{subj}: {mse_val}")

# Step 9: Save results to CSV
result_table.to_csv('predicted_marks_user.csv', index=False)

# Step 10: Visualize predicted scores with enhanced bar chart
subjects_names = [s.replace('_score','').title() for s in subjects]
scores_values = [predicted_scores[s] for s in subjects_names]

plt.figure(figsize=(10,6))
bars = plt.bar(subjects_names, scores_values, 
               color=plt.cm.Blues([i/len(subjects_names) for i in range(len(subjects_names))]))
plt.ylim(0, 100)
plt.title(f"Predicted Marks for {first_name} {last_name}", fontsize=14, fontweight='bold')
plt.ylabel("Predicted Marks", fontsize=12)
plt.xlabel("Subjects", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display score value on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
