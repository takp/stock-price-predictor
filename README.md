# stock-price-predictor

Predicting the stock price has been researched for long. 
Now many people try to predict stock price with the machine learning algorithms, 
but there is not a single answer for this and it is still challenging problem.

It is also known that every country's stock market influences each other. 
In this project, I am going to predict the stock price in Japan with the data of US stock price and USD/JPY exchange rates.

## Versions

- Python 3.6.1 :: Anaconda 4.4.0 (64-bit)
- tensorflow-1.4.0
- keras-2.0.9

## Methodology

### LSTM Model

LSTM is a one kind of the RNN (Recurrent neural network), capable of learning long-term dependencies. 
Both RNN and LSTM has the repeating module of neural network. 
RNN repeats it very simple structure, but LSTM repeats it with special way. 
So that LSTM can remember with longer context, and it is the reason to use LSTM for this problem.

### Benchmark

Benchmark model is made by the DummyClassifier. The MSE (Mean Squared Error) of the prediction should be less than the benchmark score.

## Datasets

Please download the csv data from Quandl.

- Nikkei225: https://www.quandl.com/data/NIKKEI/INDEX-Nikkei-Index
- NASDAQ: https://www.quandl.com/data/NASDAQOMX/COMP-NASDAQ-Composite-COMP
- Currency Exchange(JPY/USD) https://www.quandl.com/data/CURRFX/USDJPY-Currency-Exchange-Rates-USD-vs-JPY

Add `/data` directory and locate the downloaded files under `/data` directory.

## Run

```bash
$ python main.py
```

## Proposal

You can find the proposal [here](https://github.com/takp/MLND-capstone-proposal/blob/master/proposal.pdf).

## Report

You can find the report [here](https://github.com/takp/MLND-capstone-project/blob/master/report.pdf).

## Author

Takayoshi Nishida <takayoshi.nishida@gmail.com>
