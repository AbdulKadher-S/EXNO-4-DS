# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'bmi.csv'
data = pd.read_csv(file_path)

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

def apply_scalers(data, features):
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'RobustScaler': RobustScaler()
    }
    scaled_data = {}
    for key, scaler in scalers.items():
        scaled_features = scaler.fit_transform(data[features])
        scaled_data[key] = pd.DataFrame(scaled_features, columns=features)
        scaled_data[key]['Gender'] = data['Gender'].values
    return scaled_data

features = ['Height', 'Weight', 'Index']
scaled_data = apply_scalers(data, features)

X = data.drop('Gender', axis=1)
y = data['Gender']
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support(indices=True)]
selected_data = pd.DataFrame(X_selected, columns=selected_features)
selected_data['Gender'] = y.values

selected_data.to_csv('selected_scaled_data.csv', index=False)

plt.figure(figsize=(15, 8))
plt.suptitle('Feature Scaling Comparison')
for i, (scaler, df) in enumerate(scaled_data.items(), 1):
    plt.subplot(2, 2, i)
    for feature in features:
        sns.kdeplot(df[feature], label=feature)
    plt.title(scaler)
    plt.legend()
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('feature_scaling_comparison.png')

plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')

<img width="1546" height="808" alt="Screenshot 2025-10-15 135349" src="https://github.com/user-attachments/assets/456ecc0e-29df-4b91-a6ef-5930c84c8628" />
<img width="991" height="816" alt="Screenshot 2025-10-15 135338" src="https://github.com/user-attachments/assets/f2b6c959-db61-495d-8152-969b859338a0" />

# RESULT:
  The above program was executed successfully.
