import pandas as pd

##### Importation de nos dataframes #####
test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")
sample = pd.read_csv("data/sample_submission.csv")


##### Mise en forme de nos colonnes qui sont des espèces de listes #####

#Fonction qui sert à
def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d

testo = test['genres']
testo = test['genres'].map(lambda x: sorted([d['name'] for d in eval(x)])).map(lambda x: ','.join(map(str, x)))

testo = test
testo['genres'] = testo['genres'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))
genres = testo.genres.str.get_dummies(sep=',')
testoo = pd.concat([train, genres], axis=1)