# Food-amenities-ordered-quantity-predictions
A comprehensive repository containing the step by step approach to increasing the predictive accuracy of ordered quantities.


***Business Problem***
                
We need to accurately forecast the quantity of different SKUs (food amenities) that will be ordered by customers in the future based on the past data.

***Approach 1***

>Data Definition


1. Data variables and definition

$ AvgSP - Average Selling price time series
![avgsp vs time](https://cloud.githubusercontent.com/assets/26039458/26490509/357a97aa-4229-11e7-83f9-7f02c1ea1db5.png)


$ Wholesale - Wholesale price time series
![wholesale trend](https://cloud.githubusercontent.com/assets/26039458/26491780/aa4eb408-422e-11e7-9eaa-9816be117590.png)


$ RetailPrice - Retail Price time series
![retailprice trend](https://cloud.githubusercontent.com/assets/26039458/26491816/bff46820-422e-11e7-86bf-ba5ee5885b29.png)


$ FinalGRN - Aggregated cost price time series
![finalgrn trend](https://cloud.githubusercontent.com/assets/26039458/26491835/cf9a6158-422e-11e7-8166-8984c7554b3b.png)


$ TotalGTOrders - Time series of the total customers across all SKUs
![totalgtorders trend](https://cloud.githubusercontent.com/assets/26039458/26491845/d4db82d2-422e-11e7-9c83-f32d25b59159.png)



2. Time period considered 

$ Train data - Mar 9th, 2017 - May 7th, 2017

$ Test data - May 8th, 2017 - May 19th, 2017



3. Derived variables considered - Ordered Quantity for the SKU Carrot (local)


>Data Understanding and Processing


1. Dealing with outliers

$ Heavy outliers were spotted in the Ordered Quantity of Carrot (local).

$ The values below 250 were converted to 250 and the values above 900 were converted to 900 for easing the model build up and testing



2. Summary statistics
![summary stats](https://cloud.githubusercontent.com/assets/26039458/26491918/3bd268fc-422f-11e7-9202-f80c42b3782f.PNG)


       

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


![avgsp predictions](https://cloud.githubusercontent.com/assets/26039458/26491996/9e8e40c4-422f-11e7-8631-275e91504820.png)
![finalgrn predictions](https://cloud.githubusercontent.com/assets/26039458/26491997/9e9194f4-422f-11e7-8966-3aa105d4bdd0.png)
![retailprice predictions](https://cloud.githubusercontent.com/assets/26039458/26491998/9e969dbe-422f-11e7-9d2d-01e600ea2eac.png)
![totalgtorders prediction](https://cloud.githubusercontent.com/assets/26039458/26491999/9e9991ae-422f-11e7-85ec-c03f00845595.png)
![wholesale predictions](https://cloud.githubusercontent.com/assets/26039458/26492000/9ef22940-422f-11e7-9296-5be8c116fa57.png)



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

![orderedqty predictions](https://cloud.githubusercontent.com/assets/26039458/26492038/d326248c-422f-11e7-9451-3bfcbc9374a4.png)
![orderedqty trend](https://cloud.githubusercontent.com/assets/26039458/26492040/d360d866-422f-11e7-9058-882acd826d12.png)



3. Comparison study of model

$ At this position, the model performs poorly as compared to the existing technique based on Seasonal Naiive Bayes method. The hidden patterns haven't been completely detected and processed.



4. How model will take care of customer addition input?

$ At this point, the model hasn't been calibrated to include customer addition input.
