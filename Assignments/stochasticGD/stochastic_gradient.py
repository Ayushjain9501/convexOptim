import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


dataset = pd.read_csv(r"~/convexOptim/Assignments/dataset.csv",index_col=0)
x=np.array(dataset['x'])
y=np.array(dataset['y'])


def value(theta0, theta1, x):
    y = theta0 + theta1 * x
    return y

def squareLoss(theta0,theta1,x,y):
    ypred = np.array([value(theta0,theta1,x) for x in x])
    subtracted = ypred - y
    sumSquareLoss = np.sum(np.power(subtracted,2))
    return sumSquareLoss



#try to fit a linear model
# y= mx+c type
theta0 = 0
theta1 = 1


iterations = 1000
lossHistory = np.empty(iterations)
count = [i for i in range(1,iterations+1)]  #x-axis for plotting lossHistory
lr = 0.001


for iteration in range(0,iterations):
    randInt = np.random.randint(0,100)
    step0 = lr * -2  * (y[randInt] - value(theta0,theta1,x[randInt]))
    step1 = lr * -2  * theta1 * (y[randInt] - value(theta0,theta1,x[randInt]))
    theta0 = theta0 - step0
    theta1 = theta1 - step1
    lossHistory[iteration] = squareLoss(theta0,theta1,x,y)



print("Theta 0 : ",theta0,"Theta1 : ",theta1)


plt.figure(0)
plt.scatter(x,y, c='red')
plt.plot(x,theta0 + theta1*x)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset')
plt.savefig('output.png')
plt.show()

plt.figure(1)
plt.plot(count,lossHistory)
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.title('Loss History')
plt.savefig('loss.png')
plt.show()


