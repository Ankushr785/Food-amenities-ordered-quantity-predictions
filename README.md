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

         OrderedQty  CustomerCount    BilledQty  FulfilledQty  ReturnQty  \
count     72.000000,      72.000000,    72.000000,     72.000000,  72.000000   
mean     551.847222,     88.111111,   538.688889,    516.755556,  21.830556   
std      235.413909,      53.186770,   303.285303,    293.508888,  18.710964   
min      111.000000,       1.000000,     2.000000,      2.000000,   0.000000   
25%      322.500000,      48.000000,   316.650000,    307.475000,   7.500000   
50%      543.000000,      84.000000,   535.900000,    522.050000,  17.950000   
75%      753.000000,     132.000000,   748.500000,    711.025000,  30.700000   
max      975.000000,     196.000000,  1227.000000,   1196.300000,  88.100000   

             AvgSP  Wholesale  RetailPrice  StockOutAfterTime  OSCustomerCount  \
count    72.000000,  72.000000,    72.000000,          72.000000,        72.000000   
mean     44.828056,  35.694444,    42.513889,          83.320694,       290.763889   
std      11.599634,   8.332253,     9.186630,          17.674428,        46.412569   
min      21.930000,  16.000000,    24.000000,          42.470000,        61.000000   
25%      39.060000,  28.000000,    35.750000,          70.252500,       264.750000   
50%      43.930000,  37.500000,    44.000000,          89.535000,       292.000000   
75%      52.135000,  42.000000,    50.000000,         100.000000,       327.500000   
max      65.710000,  50.000000,    58.000000,         101.550000,       355.000000   

          FinalGRN  TotalGTOrders  Late1Hour  
count    72.000000,      72.000000,  72.000000  
mean     41.158750,     240.055556,  11.458333  
std      10.905969,      58.857883,   9.838037  
min      17.480000,     145.000000,   0.000000  
25%      34.990000,     196.750000,   4.000000  
50%      41.100000,     226.000000,   8.500000  
75%      48.125000,     283.500000,  15.250000  
max      60.460000,     352.000000,  46.000000

3. Training and Test Dataset

Train - 9th Mar, 2017 - 7th May, 2017
Test - 8th May, 2017 - 19th May, 2017

4. Seasonal Effect 

$Seasonal effect is very clear from the visualizations

