import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

df = pd.read_csv("fau_clinic_employee_wellbeing.csv")


#check missing value
print (df.isnull().values.any())

print (df.dtypes)

df["AGE"].unique()

#age mapping
age_mapping = {
    'Less than 20': 1,
    '21 to 35': 2,
    '36 to 50': 3,
    '51 or more': 4
}
df.loc[:,'AGE'] = df['AGE'].map(age_mapping)

#Gender mapping
df["GENDER"].unique()

#Gender mapping
gender_mapping = {
    "Male": 1,
    "Female": 0
}
df.loc[:,'GENDER'] = df['GENDER'].map(gender_mapping)


print (df)

# Convert DAILY_STRESS to numeric, forcing invalid values to NaN
df['DAILY_STRESS'] = pd.to_numeric(df['DAILY_STRESS'], errors='coerce')

# Drop rows with NaN in DAILY_STRESS
df = df.dropna(subset=['DAILY_STRESS'])

print (df.head())
print (df.describe())


#visualization
#daily stress by gender
stress_by_gender = df.groupby('GENDER')['DAILY_STRESS'].mean()
print(stress_by_gender)

sns.barplot(x='GENDER', y='DAILY_STRESS', data=df, errorbar=None, palette=['pink', 'blue'])
plt.xlabel("Gender (0=Female, 1=Male)")
plt.ylabel('DAILY_STRESS')
plt.title('Daily stress by gender')
plt.show()

# Group by AGE and calculate mean DAILY_STRESS
stress_by_age = df.groupby('AGE')['DAILY_STRESS'].mean()
print(stress_by_age)


# Visualization: Daily Stress by Age
sns.barplot(x='AGE', y='DAILY_STRESS', data=df, errorbar=None, palette='viridis')
plt.xlabel("Age Group: 1 = Less than 20, 2 = 21 to 35, 3 = 36 to 50, 4 = 51 or more")
plt.ylabel('Average Daily Stress')
plt.title('Daily Stress by Age Group')
plt.show()

#daily stress by gender
hobby = df.groupby('GENDER')['TIME_FOR_HOBBY'].mean()
print(hobby)

#hobbies by Gender
sns.barplot(x='GENDER', y='TIME_FOR_HOBBY', data=df, errorbar=None, palette=['yellow', 'blue'])
plt.xlabel("Gender (0=Female, 1=Male)")
plt.ylabel('TIME_FOR_HOBBY')
plt.title('Spend more time by hobbie')
plt.show()

# Step 6: Correlation Matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.show()

# Step 7: Correlations with WORK_LIFE_BALANCE_SCORE
wlb_correlation = correlation_matrix['WORK_LIFE_BALANCE_SCORE'].sort_values(ascending=False)
print("\nFactors Correlated with WORK_LIFE_BALANCE_SCORE:")
print(wlb_correlation)

x = df.drop(['WORK_LIFE_BALANCE_SCORE'], axis=1)
y = df['WORK_LIFE_BALANCE_SCORE']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

ml = LinearRegression()
ml.fit(x_train, y_train)
y_pred = ml.predict(x_test)



r2 =r2_score(y_test, y_pred)
print("r2 score :", r2)


predicted_value = ml.predict([[3,10,5,2,5,5,8,2,3,2,6,3,1]])
actual_value = df.loc[1,'WORK_LIFE_BALANCE_SCORE']
print(predicted_value, actual_value)


plt.figure(figsize=(15,10))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
plt.show()

pred_y_df = pd.DataFrame({'Actual Value': y_test, 'Predicted value': y_pred, 'Difference': y_test - y_pred})
print (pred_y_df[0:20])

# Add a constant term to x
x = sm.add_constant(x.astype(float))
model = sm.OLS(y, x).fit()
print(model.summary2())