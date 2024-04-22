import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

def regression(x,y1):
    # Assuming X contains ground truth distances covered and y contains errors
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x.reshape(-1, 1), y1, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test) 

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    # print("Mean Squared Error:", mse)

    # Predict expected error for new ground truth distances covered
    new_distances = x.reshape(-1, 1)  # New data points
    expected_errors = model.predict(new_distances)
    # print(expected_errors, y2[-5:-1], x[-5:-1])

    # Assuming model is your trained linear regression model
    slope = model.coef_[0]
    intercept = model.intercept_

    return slope, intercept, expected_errors

def main():
    # Generate some sample data
    base_dir = "Monocular-Visual-Odometry/data/Tello_dataset/line/consitancy/"
    slope = []
    intercept = []
    for i in range(1,11):
        # k = np.load(base_dir+f"dynamics_error_{i}.npy")
        k = np.load(base_dir + f"dynamics_error_{1}.npy")
        x = np.load(base_dir+"gt.npy")
        m, c = regression(x,k)
        slope.append(m)
        intercept.append(c)
    grad = np.mean(slope)
    interc = np.mean(intercept)
    gra_de = np.std(slope)
    intec_de = np.std(intercept)
    print(grad, interc, gra_de, intec_de)



    # y1 = np.load(base_dir+"500_features_statics_error_.npy")
    y1 = np.load(base_dir+"750dynamics_error_.npy")
    y2 = np.load(base_dir+"500dynamics_error_.npy")
    y3 = np.load(base_dir+"300dynamics_error_.npy")
    y4 = np.load(base_dir+"100dynamics_error_.npy")

    z1 = np.load(base_dir+"750dynamics_%error_.npy")
    z2 = np.load(base_dir+"500dynamics_%error_.npy")
    z3 = np.load(base_dir+"300dynamics_%error_.npy")
    z4 = np.load(base_dir+"100dynamics_%error_.npy")

    # print(np.mean(y2), np.std(y2))

    rmse1 = np.sqrt(np.mean(np.square(y1)))
    rmse2 = np.sqrt(np.mean(np.square(y2)))
    rmse3 = np.sqrt(np.mean(np.square(y3)))
    rmse4 = np.sqrt(np.mean(np.square(y4)))
    x = np.load(base_dir+"gt.npy")
    # Plot the three arrays on one graph
    plt.plot(x, y1, label='750 Features,' +' RMSE = ' + "{: .2f}".format(rmse1)+'m' + ', 1.58 frames/s')
    plt.plot(x, y2, label='500 Features,' +' RMSE = ' + "{: .2f}".format(rmse2)+'m' + ', 1.54 frames/s')
    plt.plot(x, y3, label='300 Features,' +' RMSE = ' + "{: .2f}".format(rmse3)+'m' + ', 1.64 frames/s')
    plt.plot(x, y4, label='100 Features,' +' RMSE = ' + "{: .2f}".format(rmse4)+'m' + ', 2.58 frames/s')

    # # Add labels and legend
    plt.xlabel('Distance covered in m')
    plt.ylabel('Error in m')
    plt.title('Error measurement of varying Optical flow quality')
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()




    y1 = y2
    # Assuming X contains ground truth distances covered and y contains errors
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x[:-5].reshape(-1, 1), y1[:-5], test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test) 

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    # print("Mean Squared Error:", mse)

    # Predict expected error for new ground truth distances covered
    new_distances = x.reshape(-1, 1)  # New data points
    expected_errors = model.predict(new_distances)
    # print(expected_errors, y2[-5:-1], x[-5:-1])

    # Assuming model is your trained linear regression model
    slope = model.coef_[0]
    intercept = model.intercept_

    print("Slope (m):", slope)
    print("Intercept (c):", intercept)
    print(slope*x[-1]+intercept)


    # Define the exponential function
    def exponential_func(x, a, b):
        return a * np.exp(b * x)

    # Sample data (replace with your actual data)
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2.5, 3.5, 6.5, 10.5, 20.5])
    x_data = x
    y_data = y1
    # Fit the data to the exponential function
    popt, pcov = curve_fit(exponential_func, x_data, y_data)

    # Extract the coefficients
    a = popt[0]  # Coefficient representing the initial value
    b = popt[1]  # Coefficient representing the rate of change

    # Predicted values using the exponential model
    y_pred = exponential_func(x_data, a, b)

    # Print the coefficients
    print(f'Coefficient a (initial value): {a}')
    print(f'Coefficient b (rate of change): {b}')

    # Plot the original data and the exponential fit
    plt.plot(x, y1, label='Visual Odometry error in a dynamic environment')
    plt.plot(x, y_pred, color='red', label='Linear model of the error')
    plt.xlabel('Distance covered in m')
    plt.ylabel('Error in m')
    # plt.title('Error measurement of varying number of features')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()