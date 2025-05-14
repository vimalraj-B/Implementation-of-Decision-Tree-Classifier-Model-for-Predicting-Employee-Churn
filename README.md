# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Load and preprocess data: Read CSV data, handle nulls, encode categorical features like "salary".
2. Feature-target split: Select relevant features for x and set y as the "left" column.
3. Train-test split & modeling: Split the data and train a DecisionTreeClassifier using the "entropy" criterion.
4. Evaluate & predict: Measure accuracy on the test set and make predictions on new data.
```

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VIMALRAJ B
RegisterNumber:  212224230304
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data

data.head()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier (criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

![Screenshot 2025-05-14 110056](https://github.com/user-attachments/assets/835e0a97-670c-4c81-8e17-74a5e22f86f2)

![Screenshot 2025-05-14 110103](https://github.com/user-attachments/assets/8770099b-5a83-4704-9a47-ec3c8d2bc2ed)

![Screenshot 2025-05-14 110112](https://github.com/user-attachments/assets/8e6b67d5-26e7-4cd4-9bc7-725982eba502)

![Screenshot 2025-05-14 110116](https://github.com/user-attachments/assets/5900c87e-8e3f-474f-a735-9e7a7950a3a8)

![Screenshot 2025-05-14 110123](https://github.com/user-attachments/assets/5d3aad8c-438c-44c4-b27e-70d627ef01eb)

![Screenshot 2025-05-14 110129](https://github.com/user-attachments/assets/886c8101-9c60-415d-95ec-c2155df9ded1)

![Screenshot 2025-05-14 110136](https://github.com/user-attachments/assets/c76ab6d2-102f-4c18-8304-1385d9440daf)

![Screenshot 2025-05-14 110140](https://github.com/user-attachments/assets/98b8df74-aea2-49f4-ba95-16d290ef0d73)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
