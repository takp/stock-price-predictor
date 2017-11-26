# Machine Learning Nanodegree

## Capstone Project
Takayoshi Nishida  
November 26, 2017

## I. Definition

Predicting the stock price has been researched for long. 
Now many people try to predict stock price with the machine learning algorithms, but there is not a single answer for this and it is still challenging problem.

It is also known that every country's stock market influences each other.
In this project, I am going to predict the stock price in Japan with the data of US stock price and USD/JPY exchange rates.

### Project Overview

The goal of this project is to predict the change rate of the close price of Nikkei 225 index compared to the previous day. 
Nikkei 225 is a stock market index for the Tokyo Stock Exchange in Japan. 

I have an hypothesis that the Nikkei 225 has a strong correlation with the close price of US stock price and JPY/USD currency exchange rate.
So, I am going to predict the change rate of the Nikkei 225 Based on its historical data, NASDAQ and USD/JPY exchange rates.

### Problem Statement

The problem I try to solve is predicting the change rate of the Nikkei 225.

The target variable is the Nikkei 225's relative change rate from the previous day.
For example, in case the Nikkei 225 index close price is "21450.04" and it was "21374.66" at the previous day, the relative change rate is ("21450.04" / "21374.66" ) ≒ 1.00352. 
Then, it is possible to know the error between the predicted rate and the actual rate.

### Metrics

I use MSE (Mean squared error) to evaluate the prediction.
Benchmark model is made by the [DummyClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html).
The MSE of the prediction should be less than the benchmark score.

## II. Analysis

### Data Exploration

The data I am going to use is Nikkei 225, NASDAQ and USD/JPY currency data.

##### 1. Nikkei 225 

The data starts from January 1950 to current date. This data can be obtained at Quandl.
- https://www.quandl.com/data/NIKKEI/INDEX-Nikkei-Index

The input feature data is the change rate from the previous day of Nikkei 225.

##### 2. NASDAQ Index

The data starts from January 2003 to current date. This data can be obtained at Quandl.
- https://www.quandl.com/data/NASDAQOMX/COMP-NASDAQ-Composite-COMP

The input feature data is the change rate from the previous day of the NASDAQ index.

##### 3. Currency Exchange - JPY/USD

The data starts from March 1991 to current date. This data can be obtained at Quandl.
- https://www.quandl.com/data/CURRFX/USDJPY-Currency-Exchange-Rates-USD-vs-JPY

The input feature data is the change rate from the previous day of the JPY/USD exchange rate.

### Exploratory Visualization

These are the example of the original data.

##### Nikkei data

```
         Date  Open Price  High Price  Low Price  Close Price
0  2017-11-24    22390.14    22567.20   22381.01     22550.85
1  2017-11-22    22601.55    22677.34   22513.44     22523.15
2  2017-11-21    22456.79    22563.25   22416.48     22416.48
3  2017-11-20    22279.98    22410.24   22215.07     22261.76
4  2017-11-17    22603.30    22757.40   22319.12     22396.80
```

##### Nasdaq data

```
   Trade Date  Index Value     High      Low  Total Market Value
0  2017-11-24      6889.16  6890.02  6873.74        1.032920e+13   
1  2017-11-22      6867.36  6874.52  6859.28        1.029596e+13   
2  2017-11-21      6862.48  6862.66  6820.02        1.029101e+13   
3  2017-11-20      6790.71  6795.83  6779.49        1.018278e+13   
4  2017-11-17      6782.79  6797.75  6777.43        1.017850e+13

   Dividend Market Value  
0             81506001.0  
1            536745500.0  
2            169687729.0  
3            151733686.0  
4            284443422.0  
```

##### Currency data

```
         Date        Rate  High (est)   Low (est)
0  2017-11-27  111.508003  111.540001  111.540001
1  2017-11-24  111.253998  111.571999  111.259003
2  2017-11-23  111.313004  111.376999  111.070999
3  2017-11-22  112.341003  112.343002  111.510002
4  2017-11-21  112.551003  112.699997  112.183998
```

All these 3 data has the close price.

Usually each market opens for the weekday, but the market is closed for the holidays and it's different between Japan and US.
So I remove the data if at lease one of the markets was closed.
