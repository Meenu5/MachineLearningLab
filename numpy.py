import numpy as np
import pandas as pd
import numpy.testing as npt

# Q1) Simple calculator - Considered features are "add", "subtract", "multiply", "divide"
def calc(x,y,operation="") :
    operation = operation.lower()

    if(operation == "add" or len(operation) == 0) :
        return x+y
    elif(operation == "subtract") :
        return x-y
    elif(operation == "multiply") :
        return x*y
    elif(operation == "divide") :
        if(y == 0) :
            return "Error - Divide by zero"
        else :
            return x/y
    else :
        return "Error - Invalid Operation"

# Q2) Adjacent duplicate removal - Removes the adjacent duplicates present in a list x
def adj(x) :
    size = len(x)
    i=0
    while(i<len(x)-1) : 
        if(x[i+1] == x[i]) :
            del(x[i])
            i-=1
        i+=1
    return x

# Q3) Duplicate removal - Removes the duplicates and returns unique values in a list x
def dup(x) :
    hashmap = {}
    i = 0
    while i < len(x) :
        if x[i] in hashmap.keys() :
            del(x[i])
            i-=1
        else :
            hashmap[x[i]] = 1
        i+=1
    return x

# Q4a) Minimum value - Returns the row index of minimum value in a column of a matrix x
def minimum(x) :
    return np.argmin(x,axis=0)

# Q4b) Mean centering and scaling - Standardizing each column
def standardize(x) :
    mean = np.mean(x,axis=0)
    stddev = np.std(x,axis=0)
    return np.divide(np.subtract(x,mean),stddev)

# Q5) Fills the missing data of numerical columns with its rounded mean. Last 3 rows of the csv file are the data added manually with some missing values
def fillmissing_with_mean(df) :
    mean = df.mean(axis=0)
    print(mean)
    df = df.fillna(round(mean),axis=0)
    df.to_csv("salary_table_filled.csv")
    return df

# Q6a) Fuctions to compute mean, standard deviation and covariance
def compute(x,y) :
    n = len(x)
    meanx = np.sum(x) / n
    meany = np.sum(y) / n
    sigmax = (np.sum(x**2) / n - (np.sum(x)/n)**2)**0.5
    sigmaxy = np.sum((x - (np.sum(x)/n))*(y - (np.sum(y)/n))/(n-1))
    return meanx, sigmax, sigmaxy

# Q6b) Numpy functions to compute mean, standard deviation and covariance
def npcompute(x,y) :
    return np.mean(x), np.std(x), np.cov(x,y)[0][1]

def assert_check(x,y,num) :

    # Check if equal upto desired precision, current precision is set as 20
    try :
        npt.assert_almost_equal(x,y,num)
        print("Values are equal to the desired precision")
    except AssertionError:
        print("Values are not almost equal to the desired precision")


    # Check if equal upto desired significant digits, current number set to 20
    try :
        npt.assert_approx_equal(x,y,num)
        print("Values are equal to the desired significant digits")
    except AssertionError:
        print("Values are not almost equal to the desired significant digits")

    # Check if two values are equal
    try :
        npt.assert_approx_equal(x,y,num)
        print("Values are equal")
    except AssertionError:
        print("Values are not equal")



# SAMPLE FUNCTION CALLS
print("Q1")
print(calc(4,5,"multiply"))
print("Q2")
print(adj([1, 2, 2, 3, 2]))
print("Q3")
print(dup([1,2,2,3,2]))
print("Q4a")
print(minimum(np.random.randn(5,3)))
print("Q4b")
print(standardize(np.random.randn(5,3)))
print("Q5")
df = pd.read_csv("salary_table_missing_data.csv")
updated_data = fillmissing_with_mean(df)
print("Q6")
# mean and standard deviation
mux, sigmax = 1.78,0.1 
muy, sigmay = 1.66,0.1
x = np.random.normal(mux, sigmax, 10)
y = np.random.normal(muy, sigmay, 10)
meanx, sigmax, sigmaxy = compute(x,y)
npmeanx, npsigmax, npsigmaxy = npcompute(x,y)

# Numpy assert functions
# mean_x
assert_check(meanx,npmeanx,20)
# Sigma_x
assert_check(sigmax,npsigmax,20)
# Sigma_xy
assert_check(sigmaxy,npsigmaxy,20)

# Python assert functions
assert meanx==npmeanx
assert sigmax==npsigmax
assert sigmaxy==npsigmaxy


# Mean values are same when computed with functions or numpy built-in functions while the values do not match accurately for a higher precision and significant digits in case of standard deviation and covariance
