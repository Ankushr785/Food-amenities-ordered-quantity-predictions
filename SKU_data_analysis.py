import pandas as pd
import numpy as np
data = pd.read_csv("C:\\Users\\Ankush Raut\\Downloads\\SKU_DATA.csv")
data1 = pd.read_csv("C:\\Users\\Ankush Raut\\Downloads\\SKU_DATA.csv")

for i in range(len(data)):
    data.DeliveryDate[i] = data.DeliveryDate[i][3:6] + data.DeliveryDate[i][:3] + '2017'
data['DeliveryDate'] = pd.to_datetime(data['DeliveryDate'])
data.index = data['DeliveryDate']

#let's only analyze Carrot (local) first
data_carrot = data[data.SKUName == 'Carrot (local)']

#AvgSP time series

ts1 = data_carrot['AvgSP']
import matplotlib.pyplot as plt
plt.plot(ts1)

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=7)
    rolstd = pd.rolling_std(timeseries, window=7)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
   
#moving_avg = pd.rolling_mean(ts_log, 7)
#expwighted_avg = pd.ewma(ts_log, halflife=7)
ts1_diff = np.log(ts1) - np.log(ts1).shift()   #differencing
ts1_diff = ts1_diff.dropna()

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts1_diff, nlags=20)
lag_pacf = pacf(ts1_diff, nlags=20, method='ols')

#Plot ACF: 

plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts1_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts1_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts1_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts1_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


#now apply ARIMA forecast

from statsmodels.tsa.arima_model import ARIMA

#Combined model

X = np.log(ts1).values
train, test = X[0:60], X[60:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(3,1,2))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))

predict = []
for i in range(len(predictions)):
    predict.append(predictions[i][0])
    
tes = []
for i in range(len(test)):
    tes.append(test[i])

pre_al = []
for i in range(len(np.exp(predict))):
    pre_al.append(np.exp(predict)[i])
    
tes_al = []
for i in range(len(np.exp(tes))):
    tes_al.append(np.exp(tes)[i])


sse = 0
for i in range(len(tes_al)):
    sse+=(tes_al[i] - pre_al[i])**2
residuals = []
for i in range(len(tes_al)):
    residuals.append(tes_al[i] - pre_al[i])
    
rmse = (sse/len(tes_al))**0.5
print(rmse, np.mean(residuals))

# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

#Wholesale price time series

ts2 = data_carrot['Wholesale']
import matplotlib.pyplot as plt
plt.plot(ts2)

#moving_avg = pd.rolling_mean(ts_log, 7)
#expwighted_avg = pd.ewma(ts_log, halflife=7)
ts2_diff = np.log(ts2) - np.log(ts2).shift()   #differencing
ts2_diff = ts2_diff.dropna()

#ACF and PACF plots:

lag_acf_w = acf(ts2_diff, nlags=20)
lag_pacf_w = pacf(ts2_diff, nlags=20, method='ols')

#Plot ACF: 

plt.subplot(121) 
plt.plot(lag_acf_w)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts2_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts2_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf_w)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts2_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts2_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


#now apply ARIMA forecast

#Combined model

X_w = np.log(ts2).values
train_w, test_w = X_w[0:60], X_w[60:len(X_w)]
history_w = [x for x in train_w]
predictions_w = list()
for t in range(len(test_w)):
	model_w = ARIMA(history_w, order=(2,1,2))
	modelw_fit = model_w.fit(disp=0)
	output_w = modelw_fit.forecast()
	yhat_w = output_w[0]
	predictions_w.append(yhat_w)
	obs_w = test_w[t]
	history_w.append(obs_w)
	print('predicted=%f, expected=%f' % (yhat_w, obs_w))

predict_w = []
for i in range(len(predictions_w)):
    predict_w.append(predictions_w[i][0])
    
tes_w = []
for i in range(len(test_w)):
    tes_w.append(test_w[i])

pre_al_w = []
for i in range(len(np.exp(predict_w))):
    pre_al_w.append(np.exp(predict_w)[i])
    
tes_al_w = []
for i in range(len(np.exp(tes_w))):
    tes_al_w.append(np.exp(tes_w)[i])


sse_w = 0
for i in range(len(tes_al_w)):
    sse_w+=(tes_al_w[i] - pre_al_w[i])**2
    
residuals_w = []
for i in range(len(tes_al_w)):
    residuals_w.append(tes_al_w[i] - pre_al_w[i])
    
rmse_w = (sse_w/len(tes_al_w))**0.5
print(rmse_w, np.mean(residuals_w))

# plot
plt.plot(test_w)
plt.plot(predictions_w, color='red')
plt.show()


#Retail Price time series

ts3 = data_carrot['RetailPrice']
plt.plot(ts3)

#moving_avg = pd.rolling_mean(ts_log, 7)
#expwighted_avg = pd.ewma(ts_log, halflife=7)
ts3_diff = np.log(ts3) - np.log(ts3).shift()   #differencing
ts3_diff = ts3_diff.dropna()

#ACF and PACF plots:
lag_acf_rp = acf(ts3_diff, nlags=20)
lag_pacf_rp = pacf(ts3_diff, nlags=20, method='ols')

#Plot ACF: 

plt.subplot(121) 
plt.plot(lag_acf_rp)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts3_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts3_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf_rp)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts3_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts3_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


#now apply ARIMA forecast

#Combined model

X_rp = np.log(ts3).values
train_rp, test_rp = X_rp[0:60], X_rp[60:len(X_rp)]
history_rp = [x for x in train_rp]
predictions_rp = list()
for t in range(len(test_rp)):
	model_rp = ARIMA(history_rp, order=(2,1,2))
	modelrp_fit = model_rp.fit(disp=0)
	output_rp = modelrp_fit.forecast()
	yhat_rp = output_rp[0]
	predictions_rp.append(yhat_rp)
	obs_rp = test_rp[t]
	history_rp.append(obs_rp)
	print('predicted=%f, expected=%f' % (yhat_rp, obs_rp))

predict_rp = []
for i in range(len(predictions_rp)):
    predict_rp.append(predictions_rp[i][0])
    
tes_rp = []
for i in range(len(test_rp)):
    tes_rp.append(test_rp[i])

pre_al_rp = []
for i in range(len(np.exp(predict_rp))):
    pre_al_rp.append(np.exp(predict_rp)[i])
    
tes_al_rp = []
for i in range(len(np.exp(tes_rp))):
    tes_al_rp.append(np.exp(tes_rp)[i])


sse_rp = 0
for i in range(len(tes_al_rp)):
    sse_rp+=(tes_al_rp[i] - pre_al_rp[i])**2

residuals_rp = []
for i in range(len(tes_al_rp)):
    residuals_rp.append(tes_al_rp[i] - pre_al_rp[i])    

rmse_rp = (sse_rp/len(tes_al_rp))**0.5
print(rmse_rp, np.mean(residuals_rp))

# plot
plt.plot(test_rp)
plt.plot(predictions_rp, color='red')
plt.show()


#FinalGRN Time series



ts4 = data_carrot['FinalGRN']
import matplotlib.pyplot as plt
plt.plot(ts4)

#moving_avg = pd.rolling_mean(ts_log, 7)
#expwighted_avg = pd.ewma(ts_log, halflife=7)
ts4_diff = np.log(ts4) - np.log(ts4).shift()   #differencing
ts4_diff = ts4_diff.dropna()

#ACF and PACF plots:

lag_acf_fg = acf(ts4_diff, nlags=20)
lag_pacf_fg = pacf(ts4_diff, nlags=20, method='ols')

#Plot ACF: 

plt.subplot(121) 
plt.plot(lag_acf_fg)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts4_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts4_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf_fg)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts4_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts4_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


#now apply ARIMA forecast

#Combined model

X_fg = np.log(ts4).values
train_fg, test_fg = X_fg[0:60], X_fg[60:len(X_fg)]
history_fg = [x for x in train_fg]
predictions_fg = list()
for t in range(len(test_fg)):
	model_fg = ARIMA(history_fg, order=(2,1,1))
	modelfg_fit = model_fg.fit(disp=0)
	output_fg = modelfg_fit.forecast()
	yhat_fg = output_fg[0]
	predictions_fg.append(yhat_fg)
	obs_fg = test_fg[t]
	history_fg.append(obs_fg)
	print('predicted=%f, expected=%f' % (yhat_fg, obs_fg))

predict_fg = []
for i in range(len(predictions_fg)):
    predict_fg.append(predictions_fg[i][0])
    
tes_fg = []
for i in range(len(test_fg)):
    tes_fg.append(test_fg[i])

pre_al_fg = []
for i in range(len(np.exp(predict_fg))):
    pre_al_fg.append(np.exp(predict_fg)[i])
    
tes_al_fg = []
for i in range(len(np.exp(tes_fg))):
    tes_al_fg.append(np.exp(tes_fg)[i])


sse_fg = 0
for i in range(len(tes_al_fg)):
    sse_fg+=(tes_al_fg[i] - pre_al_fg[i])**2

residuals_fg = []
for i in range(len(tes_al_fg)):
    residuals_fg.append(tes_al_fg[i] - pre_al_fg[i])    

rmse_fg = (sse_fg/len(tes_al_fg))**0.5
print(rmse_fg, np.mean(residuals_fg))

# plot
plt.plot(test_fg)
plt.plot(predictions_fg, color='red')
plt.show()

#Total GTOrders time series



ts5 = data_carrot['TotalGTOrders']
import matplotlib.pyplot as plt
plt.plot(ts5)

#moving_avg = pd.rolling_mean(ts_log, 7)
#expwighted_avg = pd.ewma(ts_log, halflife=7)
ts5_diff = np.log(ts5) - np.log(ts5).shift()   #differencing
ts5_diff = ts5_diff.dropna()

#ACF and PACF plots:

lag_acf_gt = acf(ts5_diff, nlags=20)
lag_pacf_gt = pacf(ts5_diff, nlags=20, method='ols')

#Plot ACF: 

plt.subplot(121) 
plt.plot(lag_acf_gt)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts5_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts5_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf_gt)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts5_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts5_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


#now apply ARIMA forecast

#Combined model

X_gt = np.log(ts5).values
train_gt, test_gt = X_gt[0:60], X_gt[60:len(X_gt)]
history_gt = [x for x in train_gt]
predictions_gt = list()
for t in range(len(test_gt)):
	model_gt = ARIMA(history_gt, order=(15,1,1))
	modelgt_fit = model_gt.fit(disp=0)
	output_gt = modelgt_fit.forecast()
	yhat_gt = output_gt[0]
	predictions_gt.append(yhat_gt)
	obs_gt = test_gt[t]
	history_gt.append(obs_gt)
	print('predicted=%f, expected=%f' % (yhat_gt, obs_gt))

predict_gt = []
for i in range(len(predictions_gt)):
    predict_gt.append(predictions_gt[i][0])
    
tes_gt = []
for i in range(len(test_gt)):
    tes_gt.append(test_gt[i])

pre_al_gt = []
for i in range(len(np.exp(predict_gt))):
    pre_al_gt.append(np.exp(predict_gt)[i])
    
tes_al_gt = []
for i in range(len(np.exp(tes_gt))):
    tes_al_gt.append(np.exp(tes_gt)[i])


sse_gt = 0
for i in range(len(tes_al_gt)):
    sse_gt+=(tes_al_gt[i] - pre_al_gt[i])**2

residuals_gt = []
for i in range(len(tes_al_gt)):
    residuals_gt.append(tes_al_gt[i] - pre_al_gt[i])
    
rmse_gt = (sse_gt/len(tes_al_gt))**0.5
print(rmse_gt, np.mean(residuals_gt))

# plot
plt.plot(test_gt)
plt.plot(predictions_gt, color='red')
plt.show()

print(rmse, rmse_w, rmse_rp, rmse_fg, rmse_gt)

#OrderedQty time series
ts = data_carrot['OrderedQty']

plt.plot(ts)

#moving_avg = pd.rolling_mean(ts_log, 7)
#expwighted_avg = pd.ewma(ts_log, halflife=7)
ts_diff = np.log(ts) - np.log(ts).shift()   #differencing
ts_diff = ts_diff.dropna()

#ACF and PACF plots:

lag_acf_o = acf(ts_diff, nlags=20)
lag_pacf_o = pacf(ts_diff, nlags=20, method='ols')

#Plot ACF: 

plt.subplot(121) 
plt.plot(lag_acf_o)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf_o)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


#now apply ARIMA forecast

#Combined model

X_o = np.log(ts).values
train_o, test_o = X_o[0:60], X_o[60:len(X_o)]
history_o = [x for x in train_o]
predictions_o = list()
for t in range(len(test_o)):
	model_o = ARIMA(history_o, order=(2,1,1))
	modelo_fit = model_o.fit(disp=0)
	output_o = modelo_fit.forecast()
	yhat_o = output_o[0]
	predictions_o.append(yhat_o)
	obs_o = test_o[t]
	history_o.append(obs_o)
	print('predicted=%f, expected=%f' % (yhat_o, obs_o))

predict_o = []
for i in range(len(predictions_o)):
    predict_o.append(predictions_o[i][0])
    
tes_o = []
for i in range(len(test_o)):
    tes_o.append(test_o[i])

pre_al_o = []
for i in range(len(np.exp(predict_o))):
    pre_al_o.append(np.exp(predict_o)[i])
    
tes_al_o = []
for i in range(len(np.exp(tes_o))):
    tes_al_o.append(np.exp(tes_o)[i])


sse_o = 0
for i in range(len(tes_al_o)):
    sse_o+=(tes_al_o[i] - pre_al_o[i])**2
    
residuals_o = []
for i in range(len(tes_al_o)):
    residuals_o.append(tes_al_o[i] - pre_al_o[i])
    
rmse_o = (sse_o/len(tes_al_o))**0.5
print(rmse_o, np.mean(residuals_o))

# plot
plt.plot(test_o)
plt.plot(predictions_o, color='red')

#Final prediction model
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators = 100, learning_rate = .5, max_depth = 5)


splitter = data1[data1['SKUName'] == 'Carrot (local)']
gbr_train, gbr_test = splitter[:60], splitter[60:]
gbr_test = gbr_test.reset_index(drop=True)
gbr_train = gbr_train.reset_index(drop=True)


#gbr_train.AvgSP = (gbr_train.AvgSP - np.mean(gbr_train.AvgSP))/(np.max(gbr_train.AvgSP) - np.min(gbr_train.AvgSP))
#gbr_train.Wholesale = (gbr_train.Wholesale - np.mean(gbr_train.Wholesale))/(np.max(gbr_train.Wholesale) - np.min(gbr_train.Wholesale))
#gbr_train.RetailPrice = (gbr_train.RetailPrice - np.mean(gbr_train.RetailPrice))/(np.max(gbr_train.RetailPrice) - np.min(gbr_train.RetailPrice))
#gbr_train.FinalGRN = (gbr_train.FinalGRN - np.mean(gbr_train.FinalGRN))/(np.max(gbr_train.FinalGRN) - np.min(gbr_train.FinalGRN))
#gbr_train.TotalGTOrders = (gbr_train.TotalGTOrders - np.mean(gbr_train.TotalGTOrders))/(np.max(gbr_train.TotalGTOrders) - np.min(gbr_train.TotalGTOrders))


ser_pre_al = pd.Series(pre_al)
ser_pre_al_w = pd.Series(pre_al_w)
ser_pre_al_rp = pd.Series(pre_al_rp)
ser_pre_al_fg = pd.Series(pre_al_fg)
ser_pre_al_gt = pd.Series(pre_al_gt)

#ser_pre_al = (ser_pre_al - np.mean(ser_pre_al))/(np.max(ser_pre_al)-np.min(ser_pre_al))
#ser_pre_al_w = (ser_pre_al_w - np.mean(ser_pre_al_w))/(np.max(ser_pre_al_w)-np.min(ser_pre_al_w))
#ser_pre_al_rp = (ser_pre_al_rp - np.mean(ser_pre_al_rp))/(np.max(ser_pre_al_rp)-np.min(ser_pre_al_rp))
#ser_pre_al_fg = (ser_pre_al_fg - np.mean(ser_pre_al_fg))/(np.max(ser_pre_al_fg)-np.min(ser_pre_al_fg))
#ser_pre_al_gt = (ser_pre_al_gt - np.mean(ser_pre_al_gt))/(np.max(ser_pre_al_gt)-np.min(ser_pre_al_gt))



trainer = pd.DataFrame({'AvgSP':gbr_train.AvgSP, 'Wholesale':gbr_train.Wholesale, 'RetailPrice':gbr_train.RetailPrice, 'FinalGRN':gbr_train.FinalGRN, 'TotalGTOrders':gbr_train.TotalGTOrders})
trainer = trainer.reset_index(drop=True)
labels = pd.DataFrame({'OrderedQty':gbr_train.OrderedQty})
gbr_tester = pd.DataFrame({'AvgSP':ser_pre_al, 'Wholesale':ser_pre_al_w, 'RetailPrice':ser_pre_al_rp, 'FinalGRN':ser_pre_al_fg, 'TotalGTOrders':ser_pre_al_gt}, index=[0,1,2,3,4,5,6,7,8,9,10,11])
labels = labels.reset_index(drop=True)
#labels1 = (labels - np.mean(labels))/(np.max(labels)-np.min(labels))


gbr_fit = gbr.fit(trainer, labels)
gbr_predicted = gbr.predict(gbr_tester) 


sse_ = 0
for i in range(len(gbr_test['OrderedQty'])):
    sse_+=(gbr_test['OrderedQty'][i] - gbr_predicted[i])**2
rmse_ = (sse_/len(gbr_predicted))**0.5
print(rmse_)







