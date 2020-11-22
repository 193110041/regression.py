import requests
import pandas as pd
import scipy
import numpy as np
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    X= data["area"].values
    y= data["price"].values
    
    #Mean X and Y
    mean_x= np.mean(X)
    mean_y= np.mean(Y)
    
    m=len(X)
    numer= 0
    denom= 0
    for i in range(m):
        numer+= (X[i]- mean_x)*(Y[i]- mean_y)
        denom+=  (X[i]- mean_x)**2
    b1= numer/denom
    b0= mean_y-(b1*mean_x)
    
    print(b1,b0)
    
    # check the goodness of model using R-squared value
    ss_t=0
    ss_r=0
    for i in range(m):
        y_pred= b0+b1*X[i]
        ss_t+= (Y[i]-mean_y)**2
        ss_r+= (Y[i]-y_pred)**2
     r^2= 1-(ss_r/ss_t)
     print(r^2)
    
if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
