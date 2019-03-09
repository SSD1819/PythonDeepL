
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


###le site qui a permis de faire la regression :
###https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9

#Reg
modeleReg=LinearRegression()
noms = trainSamp.columns.drop(["id","title","cast","crew"])

#nomsquanti = noms.drop(["original_language","original_title","production_companies","release_date"])
X = trainSamp[noms]
#suppression des nans
df2 = X.dropna(axis=0, how="any")

y =  df2.revenue
#suppression de revenue suite à la copie
newnoms=df2.columns.drop(["revenue"])
df2=df2[newnoms]

X = sm.add_constant(df2)
model = sm.OLS(y, X).fit()
model.summary()