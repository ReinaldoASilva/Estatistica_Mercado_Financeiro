import pandas as pd 
import numpy as np
import yfinance as yf

mglu3 = yf.download('MGLU3.SA', period = 'max')['Adj Close'].plot()