import pandas as pd

##### Importation de nos dataframes #####
test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")
sample = pd.read_csv("data/sample_submission.csv")

# Nb NAs
train.count()
train.isna().sum() #directement le nombre des NAs

#je pense il faut aussi supprimer "belongs_to_collection" colonne

##### Suppression de nos variables qui ne nous servent à rien #####
train = train.drop(['homepage','imdb_id','overview','poster_path','status','tagline','crew'],axis=1,errors='ignore')
test = test.drop(['homepage','imdb_id','overview','poster_path','status','tagline','crew'],axis=1,errors='ignore')

#j'ai des doutes qu'il faut supprimer 'crew'
train = train.drop(['homepage','imdb_id','overview','poster_path','status','tagline','crew']
##### Mise en forme de nos colonnes qui sont des espèces de listes #####

#Fonction qui sert à
def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d

testo = test
testo['genres'] = testo['genres'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))
genres = testo.genres.str.get_dummies(sep=',')
testoo = pd.concat([train, genres], axis=1)