x_old = 0
x_new = 6
gamma = 0.01 # Step size
precision = 0.00001


def func(x):
    y = 4 * x**3 - 9 * x**2
    return y

while abs(x_new - x_old) > precision:
    x_old = x_new
    x_new += -gamma * func(x_old)

print("Local minimum is %f" % x_new)