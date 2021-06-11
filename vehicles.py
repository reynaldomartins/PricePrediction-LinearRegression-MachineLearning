#Part 1: Read and prep the dataset
#-----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option('mode.chained_assignment', None)
np.random.seed(123)

dfVehiclesOriginal = pd.read_csv('vehicles.csv', sep=',', index_col=0, low_memory=False)
# print(dfVehiclesOriginal.head())

# Select data subset to be used in the analysis
keptCol = [ 'price','year','manufacturer','condition', 'cylinders','fuel','odometer',
                    'transmission', 'size', 'type', 'paint_color']
dfVehiclesOriginal =  dfVehiclesOriginal.loc[:,keptCol]

# ##################################################
# Convert numeric data
dfVehiclesOriginal[['price','odometer', 'year']] = dfVehiclesOriginal[['price','odometer','year']].apply(pd.to_numeric, errors='coerce')

# print(dfVehiclesOriginal.head())

# ##################################################
# 1st round cleasing
# Delete N/A rows to reduce data into analysis sample
# Verify number of NA values by column in the data
print("\nnumber of null cells before dropping rows with null")
print(dfVehiclesOriginal.isnull().sum())
print("\nshape of dataset before null cleasing")
print(dfVehiclesOriginal.shape)
dfVehicles = dfVehiclesOriginal.dropna()
print("\nnumber of null cells after dropping rows with null")
print(dfVehicles.isnull().sum())
print("\nshape of dataset after null cleasing")
print(dfVehicles.shape)

##################################################
# 2nd round cleasing
# Based on the analysis of the frequency distribution of each categorical variable
# generated after the first cleasing a new round of cleasing process is done

# a) Rows with variables informed by the seller as "other" will be deleted,
# since "other" could be hiding right data not informed by the seller

dfVehicles['cylinders'] = dfVehicles['cylinders'].apply(lambda x : np.nan if x == "other" else x)
dfVehicles['fuel'] = dfVehicles['fuel'].apply(lambda x : np.nan if x == "other" else x)
dfVehicles['transmission'] = dfVehicles['transmission'].apply(lambda x : np.nan if x == "other" else x)
dfVehicles['type'] = dfVehicles['type'].apply(lambda x : np.nan if x == "other" else x)

# print(dfVehicles.head())

# b) Car Manufacturers with low frequency will be taken out from the prediction model
# since they can have a high standard deviation. So the model will be limited to predict the
# prices just to a set of manufacturers, those with larger quantity of advertisements

dfVehicles['manufacturer'] = dfVehicles['manufacturer'].apply(lambda x : np.nan if x in [ "tesla","harley-davidson","alfa-romeo","datsun",
                               "ferrari","land rover","aston-martin","porche","morgan" ] else x)

# c) Some pairs of categorial classifications which are difficult to tell the difference will
# be grouped together since it was depending much on the seller interpretation
# It was the cases of
# Condition - Group together "fair" and "good" as just one category
# Condition - Group together "new" and "like new" as just one category

dfVehicles['condition'] = dfVehicles['condition'].apply(lambda x : "good" if x in [ "fair", "good" ] else x)
dfVehicles['condition'] = dfVehicles['condition'].apply(lambda x : "new" if x in [ "new", "like new" ] else x)

# d) For type, they will be eliminated all rows of utilitary vehicles (except pickups)
# So truck and bus will be eliminated

dfVehicles['type'] = dfVehicles['type'].apply(lambda x : np.nan if x in [ "truck", "bus" ] else x)

# e) For colors, all colors with low frequency in the distribution will be grouped together as
# "exotic" colors. They are orange, yellow and purple

dfVehicles['paint_color'] = dfVehicles['paint_color'].apply(lambda x : "exotic" if x in [ "orange","yellow","purple" ] else x)

# g) Eliminate all rows which prices were informed as below US$ 1000
# sometimes sellers put any low amount on the advertisement such as 0, 10, etc just to
# catch the attention of potential buyers to get a open offer for the vehicle

dfVehicles['price'] = dfVehicles['price'].apply(lambda x : np.nan if x < 1000 else x)

# h) Eliminate all rows were seller did not informed the odometer

dfVehicles['odometer'] = dfVehicles['odometer'].apply(lambda x : np.nan if x <= 0 else x)

# #Delete N/A rows AGAIN to reduce data into analysis sample
# #Verify number of NA values by column in the data

# print("\nnumber of null cells before dropping rows with null - 2nd round of cleasing")
# print(dfVehicles.isnull().sum())
# print("\nshape of dataset before 2nd round of cleasing")
# print(dfVehicles.shape)
# dfVehicles = dfVehicles.dropna()
print("\nnumber of null cells after dropping rows with null - 2nd round of cleasing")
print(dfVehicles.isnull().sum())
print("\nshape of dataset after 2nd round of cleasing")
print(dfVehicles.shape)

##################################################################################################
# Examining the correlation matrix in order to eliminate redundant variables highly correlated and
# to simplify the model
corr = dfVehicles.corr()
corr.to_csv("correlation.csv")

# import seaborn as sns
# sns.heatmap(corr, annot=True,cmap='coolwarm',fmt='.2g')
# plt.savefig('Correlation')
# plt.show()

# ##################################################
# transform categorical variables into dummy variables
categorical = ['manufacturer','condition', 'cylinders','fuel',
                    'transmission', 'size', 'type', 'paint_color']
dfVehiclesCat = pd.get_dummies(dfVehicles, columns=categorical, drop_first=False)

# Eliminate reference variables
referenceVars = ['manufacturer_ford','condition_good','cylinders_6 cylinders','fuel_gas','transmission_automatic',
                    'size_mid-size','type_sedan','paint_color_black']
dfVehiclesCat = dfVehiclesCat.drop(columns = referenceVars, axis = 1)

print("Columns after categorical columns are converted to dummy")
print(dfVehiclesCat.columns)

############################################################################
# Data normalization
# Convert large numerical numbers to logaritimic scale
# Convert to log those variables which presents values above 10 (exp(1))
############################################################################

dfVehiclesCat['price_log'] = dfVehiclesCat['price'].apply(lambda x : np.log(x))
dfVehiclesCat['odometer_log'] = dfVehiclesCat['odometer'].apply(lambda x : np.log(x))
dfVehiclesCat = dfVehiclesCat.drop(columns = ['price','odometer'], axis = 1)

# Calculate age of the vehicle and eliminate all vehicle older than 20 years
# Consider a car year model 2021 as 0 year old
# Older cars in such situation must be considered as rare and collectable

dfVehiclesCat['age'] = dfVehiclesCat['year'].apply(lambda x : 2021 - x)
dfVehiclesCat = dfVehiclesCat.drop(columns = ['year'], axis = 1)
dfVehiclesCat['age'] = dfVehiclesCat['age'].apply(lambda x : np.nan if x > 20 else x)
dfVehiclesCat = dfVehiclesCat.dropna()

print("\nnumber of null cells after dropping rows with null - last cleasing")
print(dfVehiclesCat.isnull().sum())
print("\nshape of dataset after last cleasing")
print(dfVehiclesCat.shape)

dfVehiclesCat.to_csv("cleaned.csv")

############################################################################
# Model summary Analysis
# Analyse the Explanatory Model using the full dataset
# Import modules for OLS - Ordinary Least Squares prediction model
############################################################################
import statsmodels.api as sm

outcomeVar = [ 'price_log' ]
explanVars = dfVehiclesCat.columns.values.tolist()
explanVars.remove('price_log')

# print(dfVehiclesCat[outcomeVar])
# print(dfVehiclesCat[explanVars])

# print(dfVehiclesCat.shape)

############################################################################
# Model summary Analysis
############################################################################

model_name = "OLS - Ordinary Least Squares"
y = dfVehiclesCat[outcomeVar]
X = sm.add_constant(dfVehiclesCat[explanVars])
model = sm.OLS(y, X)
results = model.fit()

print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Model {} Summary \n".format(model_name))
print(results.summary2())

# print('Parameters: ', results.params)
# print('Standard errors: ', results.bse)
# print('Predicted values: ', results.predict())

print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Odds Ratio using model {}\n".format(model_name))
# Odds ratios and 95% Confidence Interval
params = results.params
conf = results.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['2.5%','97.5%','Odds Ratio']
print("\nOdds Ratio (using OLM function) for the Full Dataset")
print(np.exp(conf).round(6).to_string())

############################################################################
# Features Reduction - Non significant features
############################################################################

print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Features reduction using model {}\n".format(model_name))

# List the variables that are and not are statistically significant
# False = insignificant feature; True = significant fearture
columns_significance = results.summary2().tables[1]['P>|t|']

# print(columns_significance)

data_columns = X.columns
features_table = pd.DataFrame(data=zip(data_columns,columns_significance),columns=['Feature',"Statistically Significant"])
# print("\n")
# print(features_table)
# print("\n")

non_significant_columns = []
significant_columns = []
for i in range(len(columns_significance)):
    if columns_significance[i] > 0.05:
        non_significant_columns.append(data_columns[i])
    else:
        significant_columns.append(data_columns[i])
print("\nColumns to be eliminated using {}".format(model_name))
print(non_significant_columns)

num_columns_kept = len(columns_significance)-len(non_significant_columns)
print("\nNumber of Columnns to be KEPT after eliminating features using {} : {}".format(model_name,num_columns_kept))
print("Number of Columnns to be REMOVED using {} : {}".format(model_name,len(non_significant_columns)))

print("Kept columns:")
print(significant_columns)

X = X.loc[:,significant_columns]

############################################################################
# Removing outliers from the dataset using Cooks Distance Model
############################################################################
# http://mpastell.com/2013/04/19/python_regression/

print("start influencials")

infl = results.get_influence()
# print(infl.cooks_distance)

percentage_outliers = 15
threshold = np.nanpercentile(infl.cooks_distance[0],100-percentage_outliers,interpolation='midpoint')

print("++++++++++++ Threshold ++++++++++++++")
print(threshold)

dfInfluencials = pd.DataFrame(zip(X.index.values.tolist(), infl.cooks_distance[0]), columns = ["index","cooks_d"])
dfInfluencials = dfInfluencials.set_index('index')

# dfInfluencials["cooks_d"].to_csv("cook.csv")

# print(dfInfluencials)
# print(dfInfluencials.shape)
# print(X.shape)

outliers = dfInfluencials.loc[dfInfluencials["cooks_d"] > threshold].index

print("Number of Rows before outliers elimination :")
print(X.shape[0])

print("Number of Outliers Found using Cooks Distance")
print(len(outliers))

X = X[~X.index.isin(outliers)]
y = y[~y.index.isin(outliers)]

print("Number of Rows after outliers elimination :")
print(X.shape[0])

print("finishig influencials")

############################################################################
# Model summary Analysis after Features Reduction and Outliers Elimination
############################################################################
model = sm.OLS(y, X)
results = model.fit()

# analysis
# https://data-flair.training/blogs/python-statistics/

print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Model {} Summary after Features Reduction and Outliers elimination\n".format(model_name))
print(results.summary())

############################################################################
# Prediction Models
############################################################################
# Importing modules for prediction models
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Import modules for OLS - Ordinary Least Squares prediction model
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Import modules for RIDGE MODEL - Ridge prediction model
from sklearn import linear_model

# Import modules for LASSO MODEL - Lasso prediction model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

# Thereafter split the dataset into 60% training set and 40% test set.
# When spliting the data, set the oprion random_state=0 so that the results
# tyou obtain can be duplicated during another run of the model.
dfTrain,dfTest=train_test_split(dfVehiclesCat, test_size=0.4, random_state=123)

X_train = dfTrain[explanVars]
y_train = dfTrain[outcomeVar]

X_test = dfTest[explanVars]
y_test = dfTest[outcomeVar]

# Estimate OLS, Ridge and Lasso models using the train sample and assess their predictive performance using the test
# sample and RMSE statistic. Note: for Ridge and Lasso regressions,
# you will first need to determine the optimal regularization parameter (i.e., alpha) value to use).
# Please use the following alpha values in cross-validation techniques to determine optimal alpha values for
# Ridge and Lasso regressions: [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 1,2,5,10].

# Create a dataset to store the results
dfRevModelResults = pd.DataFrame( { 'Model' : [] , 'RMSE' : [] } )

# +++ Linear Regression
lr = LinearRegression()
results_lr = lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)
lr_RMSE = np.sqrt(metrics.mean_squared_error(y_test,y_pred_lr))
dfRevModelResults = dfRevModelResults.append( { 'Model' : "LinearRegression" , 'RMSE' : lr_RMSE } ,ignore_index=True )

# +++ Ridge
alphas = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 1,2,5,10]
ridge_reg = linear_model.RidgeCV(alphas=alphas, store_cv_values=True)
ridge_reg.fit(X_train,y_train)
alpha = ridge_reg.alpha_

ridge_reg = linear_model.Ridge(alpha=alpha)
results_ridge = ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_test)
ridge_RMSE = np.sqrt(metrics.mean_squared_error(y_test,y_pred_ridge))
print("Alpha for Ridge = {}".format(alpha))
dfRevModelResults = dfRevModelResults.append( { 'Model' : "Ridge" , 'RMSE' : ridge_RMSE } ,ignore_index=True )

# +++ Lasso
lasso = Lasso()
alphas = {'alpha':[1e-8, 1e-4, 1e-3, 1e-2, 0.1, 1,2,5,10]}
lasso_regressor = GridSearchCV(lasso,alphas,scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(X_train,y_train)
alpha = lasso_regressor.best_params_['alpha']
print("Alpha for Lasso = {}".format(alpha))

lasso = Lasso(alpha=alpha)
results_lasso = lasso.fit(X_train,y_train)
y_pred_lasso = lasso.predict(X_test)
lasso_RMSE = np.sqrt(metrics.mean_squared_error(y_test,y_pred_lasso))
dfRevModelResults = dfRevModelResults.append( { 'Model' : "Lasso" , 'RMSE' : lasso_RMSE } ,ignore_index=True )

# Printing the results from the 3 models for sack of comparison
print("\nComparing the RMSE using the 3 methods")
print(dfRevModelResults)
bestModel =  dfRevModelResults.loc[ (dfRevModelResults['RMSE'] == dfRevModelResults['RMSE'].min()) ]
print("\nThe best model is :")
print(bestModel)
bestModelName = bestModel.iloc[0,0]

if (bestModelName == "LinearRegression"):
    predictor = lr
elif (bestModelName == "Ridge"):
    predictor = ridge_reg
else:
    predictor = lasso

if predictor == lasso:
    y_pred = y_pred_lasso
    results = results_lasso
else:
    y_pred = y_pred_lr[:,0]
    if predictor == lr:
        results = results_lr
    else:
        results = results_ridge

############################################################################
# Residuals Analysis
############################################################################
# https://www.statsmodels.org/stable/examples/notebooks/generated/regression_plots.html

residuals = np.subtract(np.exp(y_test[outcomeVar[0]].to_list()), np.exp(y_pred))
dfResiduals =  pd.DataFrame(zip(y_test.index.values.tolist() , np.exp(y_test[outcomeVar[0]].to_list()),np.exp(y_pred), residuals),
                    columns = [ "index" , "actual_price", "predicted_price", "residual"])
dfResiduals = dfResiduals.set_index('index')
dfResiduals = pd.merge(dfResiduals, dfVehicles, left_index=True, right_index=True, how="left")

# print(dfVehicles)
# print(dfResiduals)

dfResiduals.to_csv("residuals.csv")

srResiduals = pd.Series(residuals)
# print(residuals)
srResiduals.plot.hist(grid=True, bins=100, rwidth=0.9, range = [-20000,20000],
                   color='#607c8e')
plt.title('Residuals for Used Vehicles Price Prediction')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
# plt.show()
plt.savefig('Residual')
