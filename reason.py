from pandas import read_csv,Series,DataFrame,to_datetime,TimeGrouper,concat,rolling_mean
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox,multivariate_normal
from numpy import ones,log
from pandas import to_numeric,options,tools, scatter_matrix, DataFrame
from matplotlib import pyplot
import numpy as np
from scipy import stats
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import csv
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')









def train_validate_test_split(df, train_percent=.9, validate_percent=.05, seed=None):
    '''

    read more about how these parameters are defined and made

    :param df: 
    :param train_percent: 
    :param validate_percent: 
    :param seed: 
    :return: 
    '''
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:validate_end]]
    test = df.ix[perm[validate_end:]]
    return train, validate, test


def featurePreparation(df):

    '''
    
    :param df: data frame that is t be analysed.
    :return: 
    
    Steps : 
    1. View the Data.
    2. Split into categorical and numeric data
    3. Summary of the features
    4. Find missing values
    5. Do something about missing values
    6. Correlations
    7. Distribution - CoxBox
    8. Create new features
    
    incorporate this : https://github.com/Mzkarim/Exploratory-Data-Analysis-in-Python/blob/master/Exploratory%20Data%20Anlysis%20in%20Python/Exploratory%20Analysis%20in%20Python%20Using%20Pandas.ipynb
    
    '''



    print(df.head())
    print(df.describe())
    '''
    print(features.describe())
    features.boxplot()
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)

    plt.show()

    scatter_matrix(features, alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show()
    '''


    '''
    transformedFeatures = DataFrame()
    for i in list(features.columns.values):
        if i not in ['Response']:
            transformedFeatures[i] = preprocessing.scale(boxcox(features[i] + 1)[0])
        else:
            transformedFeatures[i] = features[i]


            # scatter_matrix(transformedFeatures, alpha=0.2, figsize=(6, 6), diagonal='kde')
    # plt.show()
    '''

    '''

    some insights into feature preparration and engineering : https://www.kaggle.com/chechir/features-predictive-power/comments/notebook

    '''

    #return transformedFeatures

def outlierDetection():
    pass




def workflow(df):
    featurePreparation(df)


def main():
    dfTrain = read_csv("train.csv", header=0)
    dfTest = read_csv("test.csv", header=0)



    workflow(dfTrain)


if __name__ == '__main__':
    main()


    '''
    
    Variable	Description
    
TID	: Unique ID
Vendor_ID :	Technology service vendor associated with cab company
New_User :	If a new user is taking the ride
toll_price :	toll tax amount
tip_amount : 	tip given to driver (if any)
tax	: applicable tax
pickup_timestamp :	time at which the ride started
dropoff_timestamp :	time at which ride ended
passenger_count :	number of passenger during the ride
pickup_longitude :	pickup location longitude data
pickup_latitude :	pickup location latitude data
rate_category :	category assigned to different rates at which a customer is charged
store_and_fwd :	if driver stored the data offline and later forwarded
dropoff_longitude :	drop off longitude data
dropoff_latitude : 	drop off latitude data
payment_type : 	payment mode used by the customer (CRD = Credit Card, CSH - Cash, DIS - dispute, NOC - No Charge, UNK - Unknown)
surcharge	: surchage applicable on the trip
fare_amount :	trip fare (to be predicted)

    '''