import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'Hours_Studied': [1,2,3,4,5,6,7,8,9,10],
    'Marks': [20,25,30,40,45,55,65,70,85,95]
}

df = pd.DataFrame(data)

X = df[['Hours_Studied']]
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Predicted Marks:", prediction)
