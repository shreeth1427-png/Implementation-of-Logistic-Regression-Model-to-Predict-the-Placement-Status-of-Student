# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Load the student placement dataset and preprocess the data by encoding categorical variables and converting the target variable into numerical form.

2. Select the required features and split the dataset into training and testing data, then apply feature scaling.

3. Train the Logistic Regression model using the training dataset.

4. Test the model, evaluate its performance, and predict the placement status of a new student.2. 

 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: TEJASHREE M 
RegisterNumber: 252225220115
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
data = pd.read_csv("C:/Users/acer/Downloads/Placement_Data (2).csv")
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['workex'] = le.fit_transform(data['workex'])
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})
X = data[['gender', 'ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'workex']]
y = data['status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))
```
## Output:
<img width="605" height="226" alt="Screenshot 2026-02-06 085148" src="https://github.com/user-attachments/assets/dd9353a8-9673-48ab-967d-8d96eb3cf44e" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
