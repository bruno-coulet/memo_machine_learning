https://www.youtube.com/watch?v=0B5eIE_1vpU
https://calmcode.io/course/scikit-learn/introduction
https://github.com/koaning/calm-notebooks

```python
# il faut cette version pour ce tutoriel
%pip install --upgrade scikit-learn==0.23.0
```

## Schéma de base
data -> model -> prédiction

La data est coupée en 2  dataset Y et y

dataset X : variables explicatives (features) utilisées pour faire les prédictions
dataset y : variable cible (target) que l'on veut prédire

scikit-learn propose de nombreux datasets, comme `load_boston` par exemple

```python
from sklearn.datasets import load_boston

```

Si on appel le dataset tel quel, on obtient un dictionnaire :
```python
load_boston()
```

Le paramètre `return_X_y=True` permet de diviser le dataset en 2 arrays X et y :
```python
load_boston(return_X_y=True)
# On peut s'en servir pour créer les variable X et y
X, y = load_boston(return_X_y=True)
```

Autre méthode :
```python
# Chargement du dataset
data = load_boston()
# Récupérer X (features) et y (target)
X, y = data.data, data.target
```


1. créer le model <---  un objet python
2.  le modèle apprend à partir de la data <--- `.fit(X,y)` entrainement


```python
# Importation des modèles
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

# Régression linéaire
mod = LinearRegression()
# 2 entrainement du modèle avec les données
mod.fit(X, y)
# 3 prédiction
mod.predict(X)

# idem avec K plus proches voisins
mod = KNeighborsRegressor()
mod.fit(X, y)
mod.predict(X)
```



```python
# sélection d'un modèle et ajustment
mod = KNeighborsRegressor().fit(X, y)
# prédiction
pred = mod.predict(X)
# nuage de point
# valeurs des prédiction X en abcisses
# valeurs réelle (target) y en ordonnées
plt.scatter(pred, y)
```


## Preprocessing
data -> preprocessing + model -> prédiction

Les variable explicatives (features) du dataset X, utilisées pour faire les prédictions ne ont parfois des échelle différentes.

`preprocessing` : selon le modèle utilisé, il faut harmoniser les variables
`pipeline`  : pour que chaque étape soit effectuée dans l'ordre


```python
# classes Pipeline et Mise à l'échelle 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Dans l'objet pipeline : une liste de [tuples (nom, objet)]
pipe = Pipeline([
	# objet pour le prétraitement : mise à l'échelle des données
	("scale", StandardScaler()),
	# l'objet modèle de régression
	("model", KNeighborsRegressor())
	 ])

# Entraînement du pipeline sur les données X (features) et y (cible)
# le pipeline est appelé à la place du modèle
pipe.fit(X, y)
# Prédiction : idem
pred = pipe.predict(X)

# version raccourcie qui entraîne et prédit en une seule ligne
pred = pipe.fit(X, y).predict(X)

# Affichage du nuage de points
plt.scatter(pred, y)
```

