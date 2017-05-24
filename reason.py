from pandas import read_csv,Series,DataFrame,to_datetime,TimeGrouper,concat,rolling_mean
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox,multivariate_normal
from numpy import ones,log
from pandas import to_numeric,options,tools, scatter_matrix, DataFrame
from matplotlib import pyplot
import numpy as np
from scipy import stats
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score
import csv
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


site = 'https://www.hackerearth.com/challenge/hiring/einsite-data-science-hiring-challenge/problems/9d09a02921e54cbdb0ed5ae27b7f7007/'

numerical_features = ['tolls_amount', 'tip_amount', 'mta_tax', 'passenger_count', 'surcharge']
numerical_features_withResposne = ['tolls_amount', 'tip_amount', 'mta_tax', 'passenger_count', 'surcharge', 'fare_amount']





def train_validate_test_split(df, train_percent=.9, validate_percent=.05, seed=None):
    '''

    read more about how these parameters are defined and made

    :param df: 
    :param train_percent: 
    :param validate_percent: 
    :param seed: 
    :return: 
    '''

    '''
    Use of this instead of the code below : 
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    - View the Data.
    - Split into categorical and numeric data
    - Summary of the features
    - Scaling and normalise
    - Find missing values
    - Binning (Numeric to Categorical)
    - Encoding
    - Do something about missing values
    - Correlations
    - Distribution - CoxBox
    - Create new features
    
    Read this : http://radimrehurek.com/data_science_python/
    
    http://shahramabyari.com/2015/12/30/my-first-attempt-with-local-outlier-factorlof-identifying-density-based-local-outliers/
    http://scikit-learn.org/dev/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
    http://scikit-learn.org/dev/auto_examples/neighbors/plot_lof.html
    
    
    read about this : https://mail.google.com/mail/u/0/?shva=1#inbox/15c38ef65090f7e7
    
    Read This :
    
    Three part series: 
    
    https://www.datacamp.com/community/tutorials/preprocessing-in-data-science-part-1-centering-scaling-and-knn#gs.Yi7mSxU
    https://www.datacamp.com/community/tutorials/preprocessing-in-data-science-part-2-centering-scaling-and-logistic-regression#gs.ZK02b2M
    https://www.datacamp.com/community/tutorials/preprocessing-in-data-science-part-3-scaling-synthesized-data#gs.6esUNyM
    
    
    https://www.datacamp.com/community/tutorials/exploratory-data-analysis-python#gs.OgCRcQ8
    
    http://datascienceguide.github.io/exploratory-data-analysis

    outlier detection mentioned here :

    http://blog.yhat.com/posts/detecting-outlier-car-prices-on-the-web.html
    http://napitupulu-jon.appspot.com/posts/outliers-ud120.html

    use regression model to fit and then comparing with orignial values.
    '''
    features = df[numerical_features]


    print(features.head())
    print(features.describe())


    #split data into numeric data.
    features = df[numerical_features]

    #Boxplot view of data
    features.boxplot()
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.show()


    #View the histogram to consider distribution transformation.
    features.hist()
    plt.show()



    '''
    check for anomalies in data before outlier detection, ex NaN infinity values, missing values etc
    fix that issue.


    '''
    response = df.drop(['fare_amount'], 1)
    outlierDetection(features,response)


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

def outlierDetection(df, response):
    LR = LinearRegression().fit(df, response)
    trainingErrs = abs(LR.predict(df) - response)

    outlierIdx = trainingErrs >= np.percentile(trainingErrs, 95)
    plt.scatter(df.tip_amount, response, c=(0, 0, 1), marker='s')
    plt.scatter(df.tip_amount[outlierIdx], response[outlierIdx], c=(1, 0, 0), marker='s')
    plt.show()





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