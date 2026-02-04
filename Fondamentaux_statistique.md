# Espérance, Variance et écart type 

#### Formules théoriques
<br>

<div class="section-title" style="background-color: #5c5cff; padding: 10px 14px; border-radius: 8px; margin: 12px 0;">
	<h2>Espérance</h2>
</div>
l’espérance est la valeur moyenne « attendue » si on répétait l’expérience un très grand nombre de fois.

$$
E(X) = \frac{\sum_{i=1}^{n} Y_i}{n}
$$

| Symbole           | Signification                                                                    |
| ----------------- | -------------------------------------------------------------------------------- |
| $E(X)$            | espérance mathématique de la variable aléatoire $X$<br>(valeur moyenne attendue) |
| $\sum_{i=1}^{n}$​ | somme des valeurs de $Y_i$​<br>de 1 à $n$                                        |
| $Yi$​             | les valeurs observées de la variable aléatoire $Y$                               |
| $n$               | nombre d'observations                                                            |
  

<br>

<div class="section-title" style="background-color: #5c5cff; padding: 10px 14px; border-radius: 8px; margin: 12px 0;">
	<h2>Variance</h2>
</div>
(dispersion des données)<br>
Mesure quantitative de la dispersion des valeurs autour de la moyenne :

1. calculer la moyenne des données
2. soustraire la moyenne à chaque valeur pour obtenir les écarts
3. élever chaque écarts au carré
4. calculer la moyenne des carrés

$$
Var(X) = E(Y^2) - [E(Y)]^2
$$

| Symbole   | Signification                                                                  |
| --------- | ------------------------------------------------------------------------------ |
| $Var(X)$  | Variance de la variable aléatoire $X$<br>(mesure de la dispersion des données) |
| $E(Y^2)$  | Espérance de la variable $Y$ au carré<br>(moyenne des carrés des valeurs)      |
| $E(Y)$    | Espérance de la variable $Y$<br>(moyenne attendue)                             |
| $E(Y)]^2$ | Carré de l'espérance de $Y$                                                    |


<br>

<div class="section-title" style="background-color: #5c5cff; padding: 10px 14px; border-radius: 8px; margin: 12px 0;">
	<h2>Écart type</h2>
</div>
La racine carrée de la variance<br>
s’exprime dans la même unité que les données (contrairement à la variance)<br>
moins sensible aux outliers 
$$
\sigma = \sqrt{Var(X)}
$$

| Symbole       | Signification                                                                              |
| ------------- | ------------------------------------------------------------------------------------------ |
| σ             | écart type (racine carrée de la variance)<br>mesure de la dispersion autour de la moyenne) |
| $Var(X)$      | variance de la variable aléatoire $X$                                                      |
| $\sqrt{...}$​ | racine carrée                                                                              |



<br>

<div class="section-title" style="background-color: #5c5cff; padding: 10px 14px; border-radius: 8px; margin: 12px 0;">
	<h2>variance corrigée (correction de Bessel)</h2>
</div>

Pour calculer la variance sur un échantillon (et non une population entière)
On divise par $n−1$ au lieu de $n$<br>
(avec $n$ = nombre d'élément dans l'échantillon)<br>
Cela donne une estimation moins biaisée pour les petits échantillons



#### Formules empiriques (calcul sur un jeu de données)

Les formules empiriques utilisent les valeurs observées d’un échantillon pour estimer les paramètres de la population.

 **Variance empirique  (pour la population)**
 Utilisée pour les calculs pratiques sur les données (machine learning).
 $$Var(X) = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \bar{Y})^2$$
 
| Symbole             | Signification                                                    |
| ------------------- | ---------------------------------------------------------------- |
| $Var(X)$            | Variance empirique de la variable aléatoire $X$                  |
| $n$                 | Nombre total d'observations                                      |
| $\sum_{i=1}^{n}$    | Somme des valeurs de $i$ allant de 1 à $n$                       |
| $Y_i$               | Valeur observée de la variable aléatoire $Y$                     |
| $\bar{Y}$           | Moyenne empirique des valeurs observées de $Y$                   |
| $(Y_i - \bar{Y})^2$ | Carré de l'écart entre chaque valeur observée et la moyenne      |
| $\frac{1}{n}$       | Facteur de normalisation (division par le nombre d'observations) |



<br>

 **Variance empirique (pour un échantillon - correction de Bessel)**       
Elle s’appuie sur la moyenne empirique $\bar{Y}$ et le $n-1$ corrige le biais introduit par l’estimation de la moyenne sur un échantillon.
$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (Y_i - \bar{Y})^2$$

| Symbole             | Signification                                               |
| ------------------- | ----------------------------------------------------------- |
| $s^2$               | Variance corrigée (estimation non biaisée de la variance)   |
| $n$                 | Nombre total d'observations                                 |
| $n-1$               | Degré de liberté (correction de Bessel)                     |
| $\sum_{i=1}^{n}$    | Somme des valeurs de $i$ allant de 1 à $n$                  |
| $Y_i$               | Valeur observée de la variable aléatoire $Y$                |
| $\bar{Y}$           | Moyenne empirique des valeurs observées de $Y$              |
| $(Y_i - \bar{Y})^2$ | Carré de l'écart entre chaque valeur observée et la moyenne |
| $\frac{1}{n-1}$     | Facteur de normalisation ajusté (division par $n-1$)        |


**Moyenne empirique**
C’est la moyenne calculée directement sur l’échantillon observé ; elle sert de base aux calculs de variance empirique et corrigée.
$$\bar{Y} = \frac{1}{n} \sum_{i=1}^{n} Y_i$$

| Symbole          | Signification                                              |
| ---------------- | ---------------------------------------------------------- |
| $\bar{Y}$        | Moyenne empirique des valeurs observées de $Y$             |
| $n$              | Nombre total d'observations                                |
| $\sum_{i=1}^{n}$ | Somme des valeurs de $Y_i$ de 1 à $n$                      |
| $Y_i$            | Valeur observée de la variable aléatoire $Y$               |
| $\frac{1}{n}$    | Facteur de normalisation (moyenne = somme divisée par $n$) |




