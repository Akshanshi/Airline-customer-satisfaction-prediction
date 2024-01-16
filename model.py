import pickle
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(r"E:\summer AI training IIT kanpur\Invistico_Airline.csv")

# Drop the 'Arrival Delay in Minutes' column
df.drop('Arrival Delay in Minutes', axis=1, inplace=True)

# Perform label encoding on categorical variables
label_encoder = preprocessing.LabelEncoder()
df['Class'] = label_encoder.fit_transform(df['Class'])
df['Customer Type'] = label_encoder.fit_transform(df['Customer Type'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Type of Travel'] = label_encoder.fit_transform(df['Type of Travel'])

# Separate the target variable 'satisfaction' from the features
y = df['satisfaction']


# Convert the target variable to numpy array
y = y.to_numpy()

# Select the features
X = df.iloc[:, 7:21]


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Normalize the features using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit the RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Save the model to a file
pickle.dump(classifier, open('model.pkl', 'wb'))

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
import os
os.getcwd()
!zip -r -qq "app.py.zip" "app.py"                     # make zip model
# make auto download model weights
from google.colab import files                                                  # load file class
files.download('app.py.zip')                                       # download model zip file



