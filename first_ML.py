import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()

data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Education": [12, 12, 14, 14, 16, 16, 16, 18, 18, 18],
    "Age": [21, 22, 24, 25, 27, 28, 29, 31, 33, 35],
    "Salary": [25000, 28000, 35000, 40000, 50000,
               58000, 62000, 72000, 80000, 90000]
}

#df = pd.DataFrame(data)
df = pd.read_csv(r"C:\Users\Noir\Downloads\BostonHousing.csv")

print(df)
x = df[["crim", "nox","rm", "age","dis"]]
y = df["tax"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y, 
    test_size=0.2, 
    random_state=42
    )

model.fit(x_train, y_train)

#new_person = pd.DataFrame([[6, 16, 29]], columns=["Experience", "Education", "Age"])
predicted_tax = model.predict(x_test)
print("predicted:", predicted_tax)
print("Actual:", y_test.values)
#print("Model Coefficients:", model.coef_)
#print("Model Intercept:", model.intercept_)

#errors  = abs(y_test.values - predicted_salary)
#mae = errors.mean()
#print("errors:", errors)
#print("MAE:", mae)
errors = y_test.values - predicted_tax
squared_errors = errors ** 2
mse = squared_errors.mean()
print("errors:", errors)
print("MSE:", mse)

#Using scikit-learn's mean_squared_error function
mse_sklearn = mean_squared_error(y_test, predicted_tax)
print("MSE (scikit-learn):", mse_sklearn)