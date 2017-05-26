

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

sst = 0
for i in range(len(gbr_test.OrderedQty)):
    sst+=(gbr_test.OrderedQty[i] - np.mean(gbr_test.OrderedQty))**2

r_sq = 1 - (sse_/sst)
plt.plot(gbr_test.OrderedQty, color = 'blue')
plt.plot(gbr_predicted, color = 'red')
plt.title('OrderedQty predictions')
plt.show()

