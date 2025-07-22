**Calcul de la moyenne des valeurs pour chaque variables** (dimensions) du jeu de données
<br>
<img src="../img/machine_learning/acp/1_average_data_value.png" width="300"><br>
**Centrage des données**
soustrait la moyenne à chaque valeur, ce qui revient à recentrer les données autour de l'origine
<img src="../img/machine_learning/acp/2_data_centered.png" width="300"><br>
Recherche de la direction (droite ou vecteur) qui s’ajuste le mieux aux données pour y projeter les points<br>
<img src="../img/machine_learning/acp/3_search_best_line.png" width="300"><br>
Étant donné que **a** (distance de la projection à l'origine) est fixe, on peut :
- soit **minimiser b** la distance entre les points et leur projection sur la droite<br>
- soit **maximiser c** la distance entre l'origine et la projecion du point sur la droite (variance projetée sur la droite)
<p align="left">
<img src="../img/machine_learning/acp/5_minimize_b.png" width="300"style="vertical-align: top;">
<img src="../img/machine_learning/acp/6_maximize_c.png" width="300"style="vertical-align: top;">
</p>
La ligne qui s'ajuste le mieux avec les données est celle qui maximise le total du carré des distances projection/origine

La direction optimale est celle qui maximise la variance projetée (ou, équivalemment, qui minimise la somme des carrés des distances entre les points et leur projection).<br>
<img src="../img/machine_learning/acp/7_best_line_math.png" width="300"><br>
La direction (ou droite) qui capte le plus de variance possible dans les données est la **première composante principale** (PC1)
<img src="../img/machine_learning/acp/8_best_line_slope.png" width="300"><br>
Sa pente permet de calculer... 
<p align="left">
<img src="../img/machine_learning/acp/9_slope_ratio.png" width="300" height="200" style="vertical-align: top;">
<img src="../img/machine_learning/acp/10_slope_math.png" width="300" height="200" style="vertical-align: top;">
</p>
...les scores projetés (nouvelles coordonnées) de chaque individu sur cette composante
<p align="left">
  <img src="../img/machine_learning/acp/12_slope_to_1.png" width="300" style="vertical-align: top;">
  <img src="../img/machine_learning/acp/14_PC1_scores.png" width="300" style="vertical-align: center;">
</p>
Idem avec la pente de PC2
<p align="left">
<img src="../img/machine_learning/acp/15_PC2.png" width="300"style="vertical-align: top;">
<img src="../img/machine_learning/acp/16_PC2_scores.png" width="300"style="vertical-align: top;">
</p>
On pivote l'ensemble pour avoir PC1 à l'horizontal<br>
<img src="../img/machine_learning/acp/18_PC_graph.png" width="300"><br>

Pourcentage de variance expliquée par PC1 :
Il s’agit du rapport entre la variance projetée sur PC1 et la variance totale des données<br>
<img src="../img/machine_learning/acp/17_PC1_horizontal.png" width="300"><br>
Cela permet de mesurer l’importance de cette composante dans la représentation des données.

Variation sur PC1 = 15<br>
Variation sur PC2 = 3<br>
Variation Totale  = 15 + 3 = 18
Variation sur PC1   5 / 18 = 0,83 = 83 %
Variation sur PC2   3 / 18 = 0,17 = 17 %

<img src="../img/machine_learning/acp/19_PC_graph_math.png" width="800"><br>
Le **scree plot** est une représentation graphique des % de variation contenus par chaque PC
<img src="../img/machine_learning/acp/20_end.png" width="800"><br>
