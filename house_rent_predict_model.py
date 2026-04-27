import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("housing.csv")
df

missssing_values = df.isnull().sum()
missssing_values

df["ocean_proximity"].unique()

cleaned_data_set = df.dropna()

missssing_values = cleaned_data_set.isnull().sum()
missssing_values

plt.figure(figsize=(10,6))
sns.histplot(cleaned_data_set["median_house_value"],color="Gold",kde= True)
plt.title("Distribution of Median House Values")
plt.xlabel("Median House Value")
plt.ylabel("Frequency")
plt.show()

cleaned_data_set["median_house_value"].max()

plt.figure(figsize=(20,6))
sns.boxplot(x= cleaned_data_set["median_house_value"],color="skyblue")
plt.title("Box Plot of Median House Values")
plt.show()

Q1 = cleaned_data_set["median_house_value"].quantile(0.25)
Q3 = cleaned_data_set["median_house_value"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_no_outliers_1 = cleaned_data_set[(cleaned_data_set['median_house_value'] >= lower_bound) & (cleaned_data_set['median_house_value'] <= upper_bound)]
print("original Data Shape",cleaned_data_set.shape)
print("New Data Shape without outliers",data_no_outliers_1.shape)

sns.set(style = "whitegrid")
plt.figure(figsize=(20,6))
sns.boxplot(x = data_no_outliers_1["median_house_value"],color="orange" )
plt.title("Box Plot Median House Value")
plt.show()

sns.set(style = "whitegrid")
plt.figure(figsize=(20,3))
sns.boxplot(x = data_no_outliers_1["median_income"],color = "blue")
plt.title("Box plot of Median income")
plt.show()

data_no_outliers_1["median_income"].max()

Q1 = data_no_outliers_1["median_income"].quantile(0.25)
Q3 = data_no_outliers_1["median_income"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_no_outliers_2 = data_no_outliers_1[(data_no_outliers_1['median_income'] >= lower_bound) & (data_no_outliers_1['median_income'] <= upper_bound)]
print("Original data shape",data_no_outliers_1.shape)
print("New data shape without outliers",data_no_outliers_2.shape)

sns.set(style = "whitegrid")
plt.figure(figsize=(20,3))
sns.boxplot(x = data_no_outliers_2["median_income"],color = "blue")
plt.title("Box plot of Median income")
plt.show()

df = data_no_outliers_2

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only = True),annot = True, cmap = "Reds")
plt.title("Correlation Heat Map Of Housing")
plt.show()

df["ocean_proximity"].unique()

ocean_proximity_dummies = pd.get_dummies(df["ocean_proximity"],prefix = "ocean_proximity",dtype = int)
df = pd.concat([df.drop("ocean_proximity", axis = 1), ocean_proximity_dummies], axis= 1)
ocean_proximity_dummies

df.head()

df.columns

# Define your features (independent variables) and target (dependent variable)
Features =[
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
     'ocean_proximity_<1H OCEAN',
       'ocean_proximity_INLAND', 'ocean_proximity_ISLAND',
       'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']
Target = "median_house_value"

x = df[Features]
y = df[Target]

# Split the data into a training set and a testing set
# test_size specifies the proportion of the data to be included in the test split
# random_state ensures reproducibility of your split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1111)
print(f"Training set size : {x_train.shape[0]} Sample")
print(f'Test set size: {x_test.shape[0]} samples')

"""###Model Training"""

Model = LinearRegression()

Model.fit(x_train, y_train)

print("Training Complete ! Model Model has Learn the Pattern" )

"""###Test/Predict"""

# The model takes the clues (X_test) and gives us its guesses
y_pred = Model.predict(x_test)

print("Predictions complete.")

# Calculate the accuracy score (R-squared)

accuracy = Model.score(x_test, y_test)
print(f"Model Accuracy : {accuracy : .2f}")