La régression logistique est un cas particulier de la famille des Generalized Linear Models (GLM), utilisée lorsque la variable cible est binaire.

#### odds (ou cotes)
ratio entre la **probabilité qu’un événement se produise** / **probabilité qu’il ne se produise pas** :
|odds/cote|calcul par comptage|
|-|-|
|$\frac{\text{mon équipe gagne}} {\text{mon équipe perd}}$|<img src="img/classification/logistic_regression/odds.png" width=150> 5/3 = 1,7|



#### probabilité
Mesure la chance qu’un événement se produise, sur une échelle **de 0 à 1**
ratio d'un **événement** / **ensemble des issues  possibles**
|probabilité|calcul par comptage||
|-|-|-|
|$\frac{\text{mon équipe gagne}} {\text{mon équipe gagne + mon équipe perd}}$|<img src="img/classification/logistic_regression/probability.png" width=200> 5/8 = 0.625|1 - probabilité (inverse)|
|$\frac{\text{mon équipe perd}} {\text{mon équipe gagne + mon équipe perd}}$|<img src="img/classification/logistic_regression/probability_losing.png" width=200> 3/8 = 0.375|1 - probabilité (inverse)|


**Si la probabilité d’un événement est $𝑝$**<br>
p = probabilité de gagner
alors :
|odds|calcul par probabilité|
|-|-|
|$$\text{odds}=\frac{𝑝}{1-p}$$|$$\frac{\frac{5}{8}}{1 - \frac{5}{8}}= \frac{\frac{5}{8}}{\frac{3}{8}}=\frac{5}{3}=1.7$$|

Et inversement, la probabilité :<br>
$$p=\frac{odds}{1+odds}$$<br>
<br>​
une probabilité de 0,75 correspond à des odds de 3 (car 0,75/0,25=3) :<br>

$$p = 0{,}75$$

$$1 - p = 0{,}25$$

$$\text{odds} = \frac{p}{1 - p}
= \frac{0{,}75}{0{,}25}
= 3$$


### logit : log (odds)
#### Probleme
Les **odds** sont asymétriques

|odds|| | |plage|
|-|-|-|-|-|
|très favorable|<img src="img/classification/logistic_regression/odds_very_high.png" width=150>|32/3|10.7|entre 1 et $+\infty$|
|favorable|<img src="img/classification/logistic_regression/odds_high.png" width=120>|8/3|2.66|entre 1 et $+\infty$|
|défavorable|<img src="img/classification/logistic_regression/odds_low.png" width=70>|1/4|0.25|entre 1 et 0|
|très défavorable|<img src="img/classification/logistic_regression/odds_very_low.png" width=150>|1/32|0.031|entre 1 et 0|


À valeur de probabilité égale mais opposée, la valeur des petits odds est compressée comparée aux grands odds :<br>
1/6 = **0.17**<br>
6/1 = **6**<br>

<img src="img/classification/logistic_regression/asymetry.png" width=400>

#### Solution

On transforme l’axe des probabilités pour pouvoir utiliser un modèle linéaire :<br>
Du domaine [0, 1] vers (−∞, +∞) grâce au logit :<br>
$\text{logit}(p) = \log\left(\frac{p}{1-p}\right)$


Puis, lors de la prédiction, on revient de (−∞, +∞) vers [0, 1]<br>
grâce à la fonction sigmoïde (inverse du logit) :<br>
$\sigma(z) = \frac{1}{1 + e^{-z}}$

Logarithme naturel = logarithme népérien, c’est-à-dire en base e (avec $e≈2.718$)

|logit (forme classique statistiques)|fonction sigmoïde (inverse)|
|-|-|
|transforme une sismoïde en droite|transforme une droite en sigmoïde<br>fonction d'activation de la regression logisitique|
|étire l'axe y de 0 à 1 vers ($-\infty$, $+\infty$)|restreint l'axe y à l'intervalle [0, 1]<br>selon la proximité avec la frontière de décision|
||prend une valeur réelle et retourne une probabilité|
|$$\text{logit(p)}=\log\left( \frac{p}{1 - p} \right)$$|$$\text{p}=\frac{e^\text{log(odds)}}{1 + e^\text{log(odds)}}=\frac{1}{1+e^{-z}}$$|




Cela revient à centrer sur 0 et normaliser:
|$\text{Probabilité p}$|$\text{logit(p)}$|Prediction|
|-|-|-|
|de 0.5 à 1|de 0 à $+\infty$|classe 1|
|de 0 à 0.5|de $-\infty$ à 0|classe 0|

<img src="img/classification/logistic_regression/log_function.png" width=400>



<img src="../memo_maths/img/log10.png" width=400>




<div align="center">
  <img src="img/classification/logistic_regression/log_reg_01.png" width="400" align="top">
  <img src="img/classification/logistic_regression/log_reg_11.png" width="400" align="top">
</div>

<br>

---
|||
|-|-|
|$log(1)=0$|$\log(0) = -\infty$|
||si on s'approche de 0 par des valeurs positives :<br>  $\lim_{x \to 0^+} \log(x) = -\infty$| 



Pour faire simple :
$log(\frac{1}{0^+}​)=log(1)−log(0)$

$\log\left(\frac{1}{0^+}\right) = 0 - (-\infty) = +\infty$  

ou plus exactement :  
$$\lim_{x \to 0^+} \log\left(\frac{1}{x}\right) = +\infty$$

---

| ![](img/classification/logistic_regression/log_reg_03.png) | ![](img/logistic_regression/log_reg_04.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](img/classification/logistic_regression/log_reg_05.png) | ![](img/logistic_regression/log_reg_06.png) |
| ![](img/classification/logistic_regression/log_reg_07.png) | ![](img/logistic_regression/log_reg_08.png) |
| ![](img/classification/logistic_regression/log_reg_09.png) | ![](img/logistic_regression/log_reg_10.png) |
