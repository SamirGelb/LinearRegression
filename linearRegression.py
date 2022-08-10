import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes

# Creating an object of the dataset
d = load_diabetes()

# Finding x values
d_X = d.data[:, np.newaxis, 2]

# Reserving the last 20 observations in the set as the training data
dx_train = d_X[:-20]
dy_train = d.target[:-20]

# Reserving the first 20 observations in the set as the testing data
dx_test = d_X[-20:]
dy_test = d.target[-20:]

# Reshaping the training dataset
x_train_intercepts = dx_train.squeeze()


# Defining a function to find the linear regression taking x and y as arguments
def linear_regression(x, y):

    # Calculating the gradient
    m = (np.mean(x) * np.mean(y) - np.mean(x*y))/((np.mean(x)**2) - np.mean(x ** 2))

    # Calculating the line of best fit
    b = np.mean(y) - m*np.mean(x)

    # Returning the gradient and line of best fit
    return [m, b]


# Declaring the variables outside the function and calling the linear regression function
m, b = linear_regression(x_train_intercepts, dy_train)

# Creating the straight line
y_intercepts = m*x_train_intercepts + b

# Plotting the line of best fit in blue
plt.plot(x_train_intercepts, y_intercepts, c='b', label='Line of Best Fit')

# Plotting the training data in red
plt.scatter(x_train_intercepts, dy_train, c='r', label='Training Data')

# Plotting the testing data in green
plt.scatter(dx_test, dy_test, c='g', label='Testing Data')

# Adding a legend to the graph
plt.legend(loc='upper left')

# Printing out the whole graph
plt.show()

# I learnt how to add labels and a legend to the graph here:
# https://stackoverflow.com/questions/19125722/adding-a-legend-to-pyplot-in-matplotlib-in-the-simplest-manner-possible


