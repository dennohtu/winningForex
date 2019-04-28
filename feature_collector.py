import pandas as pd
import numpy as np
from feature_functions import *

#Get csv data
data = pd.read_csv('Data/EURUSDHR.csv')
data.columns = ['date','open','high','low','close','volume']
data = data.set_index(pd.to_datetime(data.date))
data = data[['open','high','low','close','volume']]

prices = data.drop_duplicates(keep=False)

##Create lists of periods used as inputs to the functions
momentumkey = [3,4,5,8,9,10]
stochastkey = [3,4,5,8,9,10]
williamskey = [6,7,8,9,10]
prockey = [12,13,14,15]
wadlkey = [15]
adosckey = [2,3,4,5]
macdkey = [15,30]
ccikey = [15]
bollingerkey = [15]
heikenashikey = [15]
paveragekey = [2]
slopekey = [3,4,5,10,20,30]
fourierkey = [10,20,30]
sinekey = [5,6]

keyList = [momentumkey,stochastkey,williamskey,prockey,wadlkey,adosckey,macdkey,ccikey,
    bollingerkey,heikenashikey,paveragekey,slopekey,fourierkey,sinekey]

##calculate all features
momentumdict = momentum(prices,momentumkey)
print('1')
stochastdisct = stochastic(prices, stochastkey)
print('2')
williamsdict = williams(prices, williamskey)
print('3')
procdict = proc(prices, prockey)
print('4')
wadldict = wadl(prices, wadlkey)
print('5')
adoscdict = adosc(prices, adosckey)
print('6')
macdict = macd(prices, macdkey)
print('7')
ccidict = cci(prices, ccikey)
print('8')
bollingerdict = bollinger(prices, bollingerkey, 2)
print('9')

hkaprices = prices.copy()
hkaprices['Symbol'] = 'EUR/USD'
hka = OHLCresample(hkaprices, '15H')
heikenashidict = heikenashi(hka, heikenashikey)
print('10')
padict = price_averages(prices, paveragekey)
print('11')
slopdict = slopes(prices, slopekey)
print('12')
fourierdict = fourier(prices, fourierkey)
print('13')
sinedict = sine(prices, sinekey)
print('14')

#create list of dicts for results
resList = [momentumdict.close, stochastdisct.close, williamsdict.close, procdict.proc, wadldict.wadl,
    adoscdict.AD, macdict.line, ccidict.cci, bollingerdict.bands, heikenashidict.candles, 
    padict.avs, slopdict.slope, fourierdict.coeffs, sinedict.coeffs]

##List of base column names
colFeat = ['momentum', 'stoch', 'will', 'proc', 'wadl', 'adosc', 'macd', 'cci', 
    'bollinger', 'heiken', 'paverage', 'slope', 'fourier', 'sine']

#populate master frame
masterFrame = pd.DataFrame(index=prices.index)

for i in range(0, len(resList)):
    if colFeat[i] == 'macd':
        colId = colFeat[i] + str(keyList[6][0]) + str(keyList[6][0])
        masterFrame[colId] = resList[i]
    else:
        for j in keyList[i]:
            for k in (list(resList[i][j])):
                colId = colFeat[i] + str(j) + k

                masterFrame[colId] = resList[i][j][k]

threshold = round(0.7*len(masterFrame))

masterFrame[['open','high','low','close']] = prices[['open','high','low','close']]

##heikenashi is resampled ==> has empty data between
masterFrame.heiken15open = masterFrame.heiken15open.fillna(method='ffill')
masterFrame.heiken15close = masterFrame.heiken15close.fillna(method='ffill')
masterFrame.heiken15high = masterFrame.heiken15high.fillna(method='ffill')
masterFrame.heiken15low = masterFrame.heiken15low.fillna(method='ffill')

##Drop columns with > 30% null data
masterFrameCleaned = masterFrame.copy()

masterFrameCleaned = masterFrameCleaned.dropna(axis=1, thresh=threshold)
masterFrameCleaned = masterFrameCleaned.dropna(axis=0)

masterFrameCleaned.tocsv('Data/masterFrame.csv')
print('Feature calculations complete')