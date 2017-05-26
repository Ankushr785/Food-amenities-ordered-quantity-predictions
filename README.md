# Food-amenities-ordered-quantity-predictions
A comprehensive repository containing the step by step approach to increasing the predictive accuracy of ordered quantities
***Business Problem***
                
We need to accurately forecast the quantity of different SKUs (food amenities) that will be ordered by customers in the future based on the past data.

***Approach 1***

>Data Definition


1. Data variables and definition

$ AvgSP - Average Selling price time series

$ Wholesale - Wholesale price time series

$ RetailPrice - Retail Price time series

$ FinalGRN - Aggregated cost price time series

$ TotalGTOrders - Time series of the total customers across all SKUs


2. Time period considered 

$ Train data - Mar 9th, 2017 - May 7th, 2017

$ Test data - May 8th, 2017 - May 19th, 2017


3. Derived variables considered - Ordered Quantity for the SKU Carrot (local)


>Data Understanding and Processing


1. Dealing with outliers

$ Heavy outliers were spotted in the Ordered Quantity of Carrot (local).

$ The values below 250 were converted to 250 and the values above 900 were converted to 900 for easing the model build up and testing


2. Summary statistics

       
3. Training and Test Dataset

$ Train - 9th Mar, 2017 - 7th May, 2017
$ Test - 8th May, 2017 - 19th May, 2017


4. Seasonal Effect 

$ Seasonal effect is very clear from the visualizations

$ It is scaled down performing 1st degree differencing on the data


5. Functions to create data input to model

$ Input required: AvgSP, Wholesale, RetailPrice, FinalGRN, TotalGTOrders

$ The training data is divided into 5 different time series for every input variable.

$ ARIMA is used to forecast the test values for all the inputs based on the training data time series.

$ Note: The forecasting is done on logarithmic scale

$ Accuracy metric used - rmse; Obtained rmse values (original scale) - AvgSP: 2.3, Wholesale: 4.3,  RetailPrice: 4.2, FinalGRN: 3.86, TotalGTOrders: 15.4


>Data Modelling

1. Model name

$ Gradient Boosting Regressor

$ It is an ensemble model which initially performs normal regression (using 'n_estimators' number of regression trees).

$ Then it improves the model by regressing over the errors and adding an extra variable (error term) to the initial regression equation.


2. Model Accuracy on training and test dataset

$ Accuracy metric - RMSE

$ Training data - 0.009

$ Test data - 235.6

$ The model clearly overfitted the training data. The reason being heavy multicollinearity. Principal Component Analysis or other feature decomposition techniques needed.


3. Comparison study of model

$ At this position, the model performs poorly as compared to the existing technique based on Seasonal Naiive Bayes method. The hidden patterns haven't been completely detected and processed.


4. How model will take care of customer addition input?

$ At this point, the model hasn't been calibrated to include customer addition input.
