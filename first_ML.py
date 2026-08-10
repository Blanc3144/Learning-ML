import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

model = LinearRegression()

data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Education": [12, 12, 14, 14, 16, 16, 16, 18, 18, 18],
    "Age": [21, 22, 24, 25, 27, 28, 29, 31, 33, 35],
    "Salary": [25000, 28000, 35000, 40000, 50000,
               58000, 62000, 72000, 80000, 90000]
}

df = pd.DataFrame(data)

print(df)
x = df[["Experience", "Education", "Age"]]
y = df["Salary"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y, 
    test_size=0.2, 
    random_state=42
    )

model.fit(x_train, y_train)

new_person = pd.DataFrame([[6, 16, 29]], columns=["Experience", "Education", "Age"])
predicted_salary = model.predict(x_test)
print("predicted:", predicted_salary)
print("Actual:", y_test.values)
#print("Model Coefficients:", model.coef_)
#print("Model Intercept:", model.intercept_)

errors  = abs(y_test.values - predicted_salary)
mae = errors.mean()
print("errors:", errors)
print("MAE:", mae)