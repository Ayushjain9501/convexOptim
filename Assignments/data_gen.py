import numpy as np 
import pandas as pd 
np.random.seed(0)

def random_data():
    x = np.random.uniform(0,1,100)
    y = 100*(x+0.1*np.random.randn(100))
    return (x, y)

x , y= random_data()
#print(x,y)

data = {'x': x,"y":y}
dataf = pd.DataFrame(data, columns=['x','y'])
dataf.to_csv(r'~/convexOptim/Assignments/dataset.csv')