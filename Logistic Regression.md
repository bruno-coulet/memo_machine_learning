
La regression logistique et la regression linéaire font partie des Generalized Linear Model GLM


prédit des valeurs de probabilité comprises entre 0 et 1

#### probabilité
Mesure la chance qu’un événement se produise
exprimée sur une échelle de 0 à 1
ratio d'un événement par l'ensemble des issues  possibles
$\frac{\text{mon équipe gagne}} {\text{mon équipe gagne + mon équipe perd}}$ <img src="img/machine_learning/logistic_regression/probability.png" width=200>

#### odds (ou cotes)
ratio entre la probabilité qu’un événement se produise et la probabilité qu’il ne se produise pas :
$\frac{\text{mon équipe gagne}} {\text{mon équipe perd}}$ <img src="img/machine_learning/logistic_regression/odd.png" width=200>


**Si la probabilité d’un événement est $𝑝$**<br>alors :$$\text{odd}=\frac{𝑝}{1-p}
$$

Et inversement 
$$p=\frac{odds}{1+odds}$$
​
 .
une probabilité de 0,75 correspond à des odds de 3 (car 0,75/0,25=3)

On **transforme l'axe y** de **probabilité de  0 à 1** en **log(odds)** : $$\log\left(\frac{p}{1 - p}\right)$$
(forme classique du _logit_ en statistiques)

![](../memo_maths/img/Log10.svg)



<!-- <table>
    <tr>
        <td><img src="img/machine_learning/logistic_regression/threshold.png" width="400"></td>
        <td><img src="img/machine_learning/logistic_regression/threshold_2.png" width="400"></td>
    </tr>
    <tr>
 <td><img src="img/machine_learning/logistic_regression/log_reg_01.png" width=400></td>
 <td><img src="img/machine_learning/logistic_regression/log_reg_11.png" width=400></td>
</tr>
</table> -->

<div align="center">
  <img src="img/machine_learning/logistic_regression/threshold.png" width="400" align="top">
  <img src="img/machine_learning/logistic_regression/threshold_2.png" width="400" align="top">
</div>

<br>

<div align="center">
  <img src="img/machine_learning/logistic_regression/log_reg_01.png" width="400" align="top">
  <img src="img/machine_learning/logistic_regression/log_reg_11.png" width="400" align="top">
</div>

<br>

---
$$log(1)=0$$

---  

$$\log(0) = -\infty$$      si on s'approche de 0 par des valeurs positives :  
$\lim_{x \to 0^+} \log(x) = -\infty$  

---

Pour faire simple :
$log(\frac{1}{0}​)=log(1)−log(0)$

$\log\left(\frac{1}{0}\right) = 0 - (-\infty) = +\infty$  

ou plus exactement :  
$$\lim_{x \to 0^+} \log\left(\frac{1}{x}\right) = +\infty$$

---

| ![](img/machine_learning/logistic_regression/log_reg_03.png) | ![](img/machine_learning/logistic_regression/log_reg_04.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](img/machine_learning/logistic_regression/log_reg_05.png) | ![](img/machine_learning/logistic_regression/log_reg_06.png) |
| ![](img/machine_learning/logistic_regression/log_reg_07.png) | ![](img/machine_learning/logistic_regression/log_reg_08.png) |
| ![](img/machine_learning/logistic_regression/log_reg_09.png) | ![](img/machine_learning/logistic_regression/log_reg_10.png) |
