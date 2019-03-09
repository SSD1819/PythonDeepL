
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Reg
modeleReg=LinearRegression()
noms = trainSamp.columns.drop(["id","title","cast","crew","revenue"])

nomsquanti = noms.drop(["original_language","original_title","production_companies","release_date"])
X = trainSamp[nomsquanti]
y =  trainSamp.revenue

tt = X.isna().sum()>0
tt


X.columns[]
modeleReg.fit(X,y)
