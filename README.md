# convexOptim

## Assignment 1 Solution
We were supposed to maximize the likelihood function for the weibull distribution.


![alt text](https://www.weibull.com/hotwire/issue148/ht148-3.png)
<br><br>
Approach :
If Gamma approaches the minimum value of the sample and Beta is less than 1,then

![equation](https://latex.codecogs.com/gif.latex?%28%5Cbeta%20-1%29%5Clog%28min%28t%29-%5Cgamma%29%20%5Crightarrow%20infinity)

To achieve this, Gamma has been set to 0.9999 times the lowest sample value!

To find the maximum, the partial derivatives with respect to all the three parameters were set to zero and solved simultaneously. On solving,

Beta sastifies the following equation to be zero.

![alt_text](https://www.weibull.com/hotwire/issue148/ht148-6.png)

Since, we know 0 < Beta < 1,
A loop was run, which evaluated the value for the above function.
Beta was chosen such that value was closest to 0.

Eta Satisfies 

![alt_text](https://www.weibull.com/hotwire/issue148/ht148-5.png)

After getting value of Beta, Eta can be found easily.


Program written in Python
Library Used - Numpy

Reference(s) : 
https://www.weibull.com/hotwire/issue148/hottopics148.htm

