import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import time


dataset = pd.read_csv(r"~/convexOptim/Assignments/dataset.csv", index_col=0)
x = np.array(dataset['x'])
y = np.array(dataset['y'])


def value(theta0, theta1, x):
    y = theta0 + theta1 * x
    return y


def squareLoss(theta0, theta1, x, y):
    ypred = np.array([value(theta0, theta1, x) for x in x])
    subtracted = ypred - y
    sumSquareLoss = np.sum(np.power(subtracted, 2))
    return sumSquareLoss


def partialDeriv0(theta0, theta1, x, y):
    ypred = np.array([value(theta0, theta1, x) for x in x])

    subtracted = ypred - y
    update = np.sum(subtracted) * 2
    return update


def partialDeriv1(theta0, theta1, x, y):
    ypred = np.array([value(theta0, theta1, x) for x in x])

    subtracted = 2 * x * (ypred - y)
    update = np.sum(subtracted)
    return update


# try to fit a linear model
# y= mx+c type
theta0 = 0
theta1 = 1


iterations = 50000
lossHistory = np.empty(iterations)
count = [i for i in range(1, iterations+1)]  # x-axis for plotting lossHistory
beta1 = 0.9
beta2 = 0.999
eta = 0.002
m0 = v0 = mHat0 = 0
m1 = v1 = mHat1 = 0

startTime = time.time()
for iteration in range(1, iterations+1):

    loss = squareLoss(theta0, theta1, x, y)
    partialTheta0 = partialDeriv0(theta0, theta1, x, y)
    partialTheta1 = partialDeriv1(theta0, theta1, x, y)
    m0 = beta1 * m0 + (1-beta1)*(partialTheta0)
    v0 = max(beta2*v0, abs(partialTheta0))
    m1 = beta1 * m1 + (1-beta1)*(partialTheta1)
    v1 = max(beta2*v1, abs(partialTheta1))
    mHat0 = m0/(1-beta1**iteration)
    mHat1 = m1/(1-beta1**iteration)
    step0 = eta*mHat0/v0
    step1 = eta*mHat1/v1
    theta0 = theta0 - step0
    theta1 = theta1 - step1
    lossHistory[iteration-1] = loss

endTime = time.time()

print("Theta 0 : ", theta0, "Theta1 : ", theta1)
print("Time Taken : ", endTime-startTime)

plt.figure(0)
plt.scatter(x, y, c='red')
plt.plot(x, theta0 + theta1*x)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset')
plt.savefig('output.png')
plt.show()

plt.figure(1)
plt.plot(count, lossHistory)
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.title('Loss History')
plt.savefig('loss.png')
plt.show()
