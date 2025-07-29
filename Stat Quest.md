[StatQuest](https://www.youtube.com/watch?v=Gv9_4yMHFhI&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=1&t=46s)

[Machine Learning avec sickit Learn, Aurélien Géron, ed. Dunod](https://github.com/ageron/handson-ml2)
## Intro

Prédictions et classification

1. Training set : permet d'imaginer des méthodes de prédiction
2. Choix d'une méthode de prédiction
3. Testing set : mesure l'écart entre les prédictions et les donnée de test

Utilise Testing Data pour évaluer les modèle de Machine Learning.  

L'important n'est pas comment un modèle colle au Training data mais de savoir si les prédictions sont justes.

![Testing data](testing-data.png)

![prediction](training-testing-prediction.png)

## Cross Validation

Les data sont découpées en n tranches
Certaines tranches sont destiné à l'entrainement, les tranches restante sont réservées pour le test.

**Comment choisir quelle tranche sert au test ?**

La <font color="orange">cross validation</font> consiste à réserver alternativement toutes les tranches de donnée aux test.
Au final, toutes les données ont servies à l'entrainement et au test :

 Soit 4 tranches de données :
 
 - les 3 premières tranches sont utilisé pour l'entrainement  ![cross_validation_4](cross_validation_train4.png)
   
 - la 4ème tranche est réservé aux tests
![cross_validation_4](cross_validation_test4.png)

- on note les résultats
![](cross_validation_track4.png)
- puis c'est la 3ème tranche qui est réservée aux tests, on note les résultats
![](cross_validation_track3.png)
- puis c'est la 2ème tranche qui est réservée aux tests, on note les résultats
![](cross_validation_test2.png)

- enfin c'est la 1ère tranche qui est réservée aux tests, on note les résultats

Reste à compiler les résultats.
Ainsi toutes les données ont servies à l'entrainement et au test



Sert à comparer différentes méthodes de Machine Learning :
- Logistic regression
- k-nearest neighbors
- support vector machines
![comparaison des modèles](img/cross_validation_comparaison.png)
  

## Confusion Matrix
tableau à n ligne et n colonne pour n paramètres à vérifier![confusion matrix](img/confusion_matrix_4_items.png)
Permet de déterminé ce que le modèle à prédit correctement (diagonale verte) et incorrectement (faux positifs et faux négatifs en rouge)
![[actual_predicted.png]]

## Sensitivity
Recall ou Sensitivity
(% of actual positive correctly predicted)
=   Vrais positifs / (vrais positifs + faux négatifs)

![sensitivity](img/sensitivity.png)

![sensitivity2](img/sensitivity2.png)
## Specificity
Specificity
% of actual negative correctly predicted

=   Vrais négatifs / (vrais négatifs + faux positifs)

![specificity.png](img/specificity.png)

![specificity2.png](img/specificity2.png)
## Precision
% of predicted positives correctly predicted

= vrais positifs / (vrais positifs + faux positifs)


## ROC
**Receivor Operator Characteristic**
Aide à choisir le meilleur seuil pour catégoriser les données

Essayons de savoir si une souris est obèse (1) ou pas (0) en sachant son poids

![](img/curve.png)

Les souris rouges ne sont pas obèses (0)
Les souris bleues sont  obèses (1)

![](img/mice.png)

Avec un seuil à 0,5

| Classification   | Seuil  |
| ---------------- | ------ |
| souris obèse     | >= 0,5 |
| souris pas obèse | <= 0,5 |

![](img/threshold_2.png)

**Comparaison des différent seuils**
Specificity = Vrais négatifs / (vrais négatifs + faux positifs)
Sensitivity =   Vrais positifs / (vrais positifs + faux négatifs)
![Comparaison des différent seuils](img/ROC.png)



## AUC
Area Under The Curve (ROC cuve)
![](img/AUC.png)

