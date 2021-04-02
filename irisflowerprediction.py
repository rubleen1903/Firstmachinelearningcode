  
#This program is about iris flower classification
#Using K-NN to classuify the different types of iris flower.
#Made using online resources by Rubleen Kaur
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline 

np.random.seed(0)  # For reproducibility


iris = pd.read_csv("../input/Iris.csv")  # Load the data
iris.head()  # Peek at the data

iris.shape

iris["Species"].value_counts()
iris.Species.value_counts().plot(kind="pie", autopct='%.1f%%', figsize=(8,8))
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)


y = iris.Species  # Set target variable
X = iris.drop(["Species", "Id"], axis=1)  # Select feature variable 
y.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()  # Load the label encoder
y = le.fit_transform(y)  # Encode the string target features into integers

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
encoder.fit_transform(y.reshape(-1,1))

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  # Load the standard scaler
sc.fit(X)  # Compute the mean and standard deviation of the feature data
X_scaled = sc.transform(X)  # Scale the feature data to be of mean 0 and variance 1


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.3, random_state=1)  # Split the dataset into 30% testing, and 70% training 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

model = KNeighborsClassifier(n_neighbors=3)  # Load our classifier
model.fit(X_train, y_train)  # Fit our model on the training data
prediction = model.predict(X_test)  # Make predictions with our trained model on the test data 
accuracy = accuracy_score(y_test, prediction) * 100  # Compare accuracy of predicted classes with test data
print('k-Nearest Neighbours accuracy | ' + str(round(accuracy, 2)) + ' %.')  