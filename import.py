import pandas as pd

##### Importation de nos dataframes #####
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample = pd.read_csv("data/sample_submission.csv")

# Nb NAs
train.isna().sum() #directement le nombre des NAs

#je pense il faut aussi supprimer "belongs_to_collection" colonne

##### Suppression de nos variables qui ne nous servent à rien #####
train = train.drop(['belongs_to_collection','homepage','imdb_id','overview','poster_path','status','tagline'],axis=1,errors='ignore')
test = test.drop(['belongs_to_collection','homepage','imdb_id','overview','poster_path','status','tagline'],axis=1,errors='ignore')

##### Mise en forme de nos colonnes qui sont des espèces de listes #####

#Jeux de données intermédiaires (on s'en servira pour construire le final)
trainClean = train
testClean = test

###On ne garde que les 'name' dans les grandes listes

#Fonction qui sert à retourner d s'il correspond à quelquechose
def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d

#genre
trainClean['genres'] = trainClean['genres'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))
testClean['genres'] = testClean['genres'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

#compagnie de production
trainClean['production_companies'] = trainClean['production_companies'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))
testClean['production_companies'] = testClean['production_companies'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

#pays de production
trainClean['production_countries'] = trainClean['production_countries'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))
testClean['production_countries'] = testClean['production_countries'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

#langues parlées
trainClean['spoken_languages'] = trainClean['spoken_languages'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))
testClean['spoken_languages'] = testClean['spoken_languages'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

#Mots clefs
trainClean['Keywords'] = trainClean['Keywords'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))
testClean['Keywords'] = testClean['Keywords'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

###Tableau disjonctif complet pour ceux qui ont plusieur modalités

#genre
trainGenres = trainClean.genres.str.get_dummies(sep=',')
testGenres = testClean.genres.str.get_dummies(sep=',')

#compagnie de production
trainProdComp = trainClean.production_companies.str.get_dummies(sep=',')
testProdComp = testClean.production_companies.str.get_dummies(sep=',')

#pays de production
trainProdPays = trainClean.production_countries.str.get_dummies(sep=',')
testProdPays = testClean.production_countries.str.get_dummies(sep=',')

#langues parlées
trainLang = trainClean.spoken_languages.str.get_dummies(sep=',')
testLang = testClean.spoken_languages.str.get_dummies(sep=',')

#Mots clefs (trop gros)
#trainKeyW = trainClean.Keywords.str.get_dummies(sep=',')
#testKeyW = testClean.Keywords.str.get_dummies(sep=',')


###Mise en forme simple pour ceux qui n'ont qu'une seule modalité (genre le producteur, mais je m'en occupe plus tard)




###Concatenation de nos donnees (compagnie de production pas mise en tableau disjonctif pour l'instant)
trainSamp = pd.concat([trainClean, trainGenres, trainProdPays, trainLang], axis=1)
testSamp = pd.concat([testClean, testGenres, testProdPays, testLang], axis=1)

#Suppression des variables en trop (remplacées par des tableaux disjonctifs )
trainSamp = trainSamp.drop(['genres','production_countries','spoken_languages','Keywords'],axis=1,errors='ignore')
testSamp = testSamp.drop(['genres','production_countries','spoken_languages','Keywords'],axis=1,errors='ignore')