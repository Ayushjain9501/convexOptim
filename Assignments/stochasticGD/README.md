# convexOptim

## Stochastic Gradient Descent
Gradient Descent works just fine, so why another variant?
On large datasets, Gradient descent computes redundant data...

Consider a Logisitic Regression Model that used 23,000 genes to predict if someone will have a disease.<br>
Let the Dataset include 10,00,000 samples.<br>
Then we compute 23 billion terms each iteration.<br>
And it's common to take atleast 1000 iterations<br>
We end up making 23 Trillion computations. And that is a very large number....
<br><br>
SGD does away with this redundancy by performing one update at a time. SGD takes a random point from the dataset and uses it to update the parameters.

In the Program
Iterations = 1000
Learning Rate = 0.001

We tried to fit a Straight line to a random sample.
![alt text](https://github.com/Ayushjain9501/convexOptim/blob/master/Assignments/stochasticGD/output.png)
![alt text](https://github.com/Ayushjain9501/convexOptim/blob/master/Assignments/stochasticGD/loss.png)

## Theta0 = 4.91479
## Theta1 = 93.0177


