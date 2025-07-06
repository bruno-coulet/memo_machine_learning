
La regression logistique et la regression linéaire font partie des Generalized Linear Model GLM


prédit des valeurs de probabilité comprises entre 0 et 1

On transforme l'axe y de probabilité de  0 à 1 en log(probabilite) : $\log\left(\frac{p}{1 - p}\right)$
(forme classique du _logit_ en statistiques)

| ![](img/machine_learning/logistic_regression/threshold.png)  | ![](img/machine_learning/logistic_regression/threshold_2.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| ![](img/machine_learning/logistic_regression/log_reg_01.png) | ![](img/machine_learning/logistic_regression/log_reg_11.png)  |



$log(1)=0$   
$\log(0) = -\infty$      si on s'approche de 0 par des valeurs positives :  $\lim_{x \to 0^+} \log(x) = -\infty$  

Pour faire simple :
$log(\frac{1}{0}​)=log(1)−log(0)$

$\log\left(\frac{1}{0}\right) = 0 - (-\infty) = +\infty$  

ou plus exactement :  
$$\lim_{x \to 0^+} \log\left(\frac{1}{x}\right) = +\infty$$

| ![](img/machine_learning/logistic_regression/log_reg_03.png) | ![](img/machine_learning/logistic_regression/log_reg_04.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](img/machine_learning/logistic_regression/log_reg_05.png) | ![](img/machine_learning/logistic_regression/log_reg_06.png) |
| ![](img/machine_learning/logistic_regression/log_reg_07.png) | ![](img/machine_learning/logistic_regression/log_reg_08.png) |
| ![](img/machine_learning/logistic_regression/log_reg_09.png) | ![](img/machine_learning/logistic_regression/log_reg_10.png) |
