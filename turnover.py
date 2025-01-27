import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('fau_clinic_turnover_data.csv')

# Display basic dataset information
print("Dataset Info:")
print(df.info())
print("\nFirst 5 Rows:\n", df.head())


# Question 1: Job Satisfaction by Job Role
job_satisfaction = df.groupby('job_role')['satisfaction_level'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(job_satisfaction)))
job_satisfaction.plot(kind='bar', color=colors)
plt.title('Average Job Satisfaction by Job Role')
plt.xlabel('Job Role')
plt.ylabel('Average Satisfaction Level')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nJob Satisfaction by Job Role:")
print(job_satisfaction)




# Question 2: Salary by Job Role
# Group by job role and count salary levels
salary_distribution = df.groupby('job_role')['salary'].value_counts(normalize=True).unstack()

# Plot the distribution as a stacked bar chart
salary_distribution.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')
plt.title('Salary Level Distribution by Job Role')
plt.xlabel('Job Role')
plt.ylabel('Proportion of Salary Levels')
plt.legend(title='Salary Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Data Preprocessing: Convert categorical columns to numeric
label_encoder = LabelEncoder()
for col in ['job_role', 'salary']:
    df[col] = label_encoder.fit_transform(df[col])


# Question 3: Correlation with 'left'
correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Print correlations with 'left'
print("\nCorrelations with 'left':")
print(correlation_matrix['left'].sort_values(ascending=False))

# Question 4: Average Job Satisfaction of Employees Who Left
avg_satisfaction_left = df[df['left'] == 1]['satisfaction_level'].mean()
print(f"\nAverage Job Satisfaction of Employees Who Left: {avg_satisfaction_left:.2f}")

# Question 5: Duration at FAU Clinic for Employees Who Left
avg_time_spend_left = df[df['left'] == 1]['time_spend_clinic'].mean()
print(f"\nAverage Time Spent at FAU Clinic by Employees Who Left: {avg_time_spend_left:.2f} years")

#----------------
# Data for the bar plot
metrics = ['Job Satisfaction', 'Duration of Stay']
values = [avg_satisfaction_left, avg_time_spend_left]
colors = ['blue', 'red']

# Create the bar plot
plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color=colors)

# Add value annotations on top of bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.1,
             f'{value:.2f}', ha='center', va='bottom', color='white', fontsize=10)

# Add titles and labels
plt.title('Average Metrics of Employees Who Left', fontsize=14)
plt.ylabel('Average Values', fontsize=12)
plt.tight_layout()
plt.show()

#------------





# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())


#Combining Features
df['patients_hours_combined'] = df['number_patients'] * df['average_montly_hours']

# Model Building: Turnover Prediction
X = df.drop('left', axis=1)
y = df['left']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")

# Feature Importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(feature_importances)

# Add a constant term to x
x = sm.add_constant(X.astype(float))
model = sm.OLS(y, x).fit()
print(model.summary2())
