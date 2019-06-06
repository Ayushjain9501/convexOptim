import numpy as np 


def betaPartialDerivative(params, beta):
    n = np.size(params)
    logArray = np.log(params)
    powerBetaArray = np.power(params,beta)
    logPowerMult = np.multiply(powerBetaArray, logArray)
    logArray = np.sum(logArray)
    powerBetaArray = np.sum(powerBetaArray)
    logPowerMult = np.sum(logPowerMult)
    a=(logArray/n)  
    b=(1/beta)  
    c=(logPowerMult/powerBetaArray)
    value = a+b-c
    return value

def getBeta(params):
    beta = 1
    minValue = 1000
    prevBetaValue = 0
    for i in range(1000000,0,-1):
        a = i/1000000
        value = abs(betaPartialDerivative(params, a))
        if value < minValue :
            minValue = value
            beta = a
        if a==1 :
            prevBetaValue = value
        if value>prevBetaValue : 
            break
        prevBetaValue = value
    return beta

def getEta(params, beta) :
    n = np.size(params)
    powerBetaArray = np.power(params,beta)
    powerBetaArray = np.sum(powerBetaArray)
    eta = (powerBetaArray/n)**(1/beta)
    return eta


inputString = input('Enter the sample value separated by space : ')
inputList = inputString.split( )
inputList = [float(x) for x in inputList]


params = np.array(inputList)
params = np.sort(params)
gamma = params[0]*0.9999
params = np.subtract(params, gamma)

beta = getBeta(params)
eta= getEta(params,beta)

print("Beta : ", beta)
print("Eta : ", eta)
print("Gamma : ", gamma)





