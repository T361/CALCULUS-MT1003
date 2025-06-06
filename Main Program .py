import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr

# Welcome message
def display_welcome_message():
    print("==========================================")
    print(" Data Prediction using Linear Regression.")
    print(" By:")
    print(" 1. Muhammad Wasay Tariq - BSSE")
    print(" 2. Taimoor Shaukat - BSSE")
    print(" 3. Daniyal Ali - BSSE")
    print("==========================================")

def choose_file():
    print("Select the dataset to operate on:")
    print("1. sales_data1.csv")
    print("2. sales_data2.csv")
    print("3. sales_data3.csv")
    while True:
        try:
            choice = int(input("Enter your choice (1/2/3): "))
            if choice == 1:
                return "sales_data1.csv"
            elif choice == 2:
                return "sales_data2.csv"
            elif choice == 3:
                return "sales_data3.csv"
            else:
                print("Invalid choice! Please select 1, 2, or 3.")
        except ValueError:
            print("Invalid input! Please enter a number (1/2/3).")

def load_data(file_name):
    try:
        selfpath = os.path.abspath(__file__)  # Path of this file
        script_directory = os.path.dirname(selfpath)  # Directory where this file is located
        file_path = os.path.join(script_directory, file_name)  # Path of the CSV dataset
        data = pd.read_csv(file_path)
        print(selfpath)

        print("Data loaded successfully!")
        print("Columns in the dataset:", data.columns.tolist())
        print(data.head())

        data.columns = data.columns.str.strip()
        return data
    except FileNotFoundError:
        print(f"File Not Found: {file_name}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train_model(data):
    if 'Time' not in data.columns or 'Sales' not in data.columns:
        raise ValueError("The dataset must contain 'Time' and 'Sales' columns.")

    X = data['Time'].values.reshape(-1, 1)
    y = data['Sales'].values
    model = lr()
    model.fit(X, y)
    return model

def predict_sales(model, year):
    future_year = np.array([[year]])
    predicted_sales = model.predict(future_year)
    return predicted_sales[0]

def calculate_equation(model):
    m = model.coef_[0]  # Gradient
    c = model.intercept_  # Y-intercept
    equation = f"y = {m:.2f}x + {c:.2f}"
    return m, c, equation

def integrate_equation(m, c):
    integral = f"y = ({m/2:.2f})x^2 + ({c:.2f})x + C"
    return integral

def visualize_trends(data, model):
    X = data['Time'].values.reshape(-1, 1)
    y = data['Sales'].values
    plt.scatter(X, y, color='blue', label='Original Data')
    plt.plot(X, model.predict(X), color='red', label='Regression Line')
    plt.title('Sales Trends and Predictions')
    plt.xlabel('Year')
    plt.ylabel('Sales (PKR)')
    plt.legend()
    plt.show()

def main():
    display_welcome_message()
    file_name = choose_file()
    data = load_data(file_name)
    
    if data is None:
        print("Exiting program due to data loading error.")
        return
    
    try:
        model = train_model(data)
    except ValueError as ve:
        print(f"Error: {ve}")
        return
    
    while True:
        print("\nMenu:")
        print("1. View Sales Trends")
        print("2. Predict Sales for a Specific Year")
        print("3. Calculate Equation (y = mx + c)")
        print("4. Integrate Equation")
        print("5. Exit")
        
        try:
            choice = int(input("Enter your choice: "))
        except ValueError:
            print("Invalid input! Please enter a number between 1 and 5.")
            continue
        
        if choice == 1:
            visualize_trends(data, model)
        elif choice == 2:
            try:
                year = int(input("Enter the year for prediction: "))
                predicted_sales = predict_sales(model, year)
                print(f"Predicted Sales for {year}: PKR {predicted_sales:,.2f}")
            except ValueError:
                print("Invalid year! Please enter a valid number.")
        elif choice == 3:
            m, c, equation = calculate_equation(model)
            print(f"Equation of the regression line: {equation}")
        elif choice == 4:
            m, c, _ = calculate_equation(model)
            integral = integrate_equation(m, c)
            print(f"Integral of the regression line: {integral}")
        elif choice == 5:
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please select a valid option.")

main()