# 🔍 Métriques de similarité en clustering non supervisé

Dans le clustering non supervisé, le choix d'une **métrique de similarité ou de distance** est essentiel pour déterminer la formation des clusters. Voici les principales métriques utilisées :

## 📐 Distances pour données numériques

| Nom         | Formule                                                                 | Remarques                                                  |
|--------------|------------------------------------------------------------------------|-------------------------------------------------------------|
| Euclidienne $∥x∥$ | $$ d(x, y) = \sqrt{ \sum (x_i - y_i)^2 } $$                           | Standard, utilisée dans K-Means                             |
| Manhattan    | $$ d(x, y) = \sum \left\| x_i - y_i \right\| $$                         | Moins sensible aux outliers                                 |
| Minkowski    | $$ d(x, y) = \left( \sum_{i=1}^n \left\| x_i - y_i \right\|^p \right)^{1/p} $$ | Paramètre $$ p $$ variable (1 = Manhattan, 2 = Euclidienne) |
| Mahalanobis  | Basée sur la matrice de covariance                                    | Prend en compte les corrélations                            |




## 🧾 Similarité pour données textuelles / binaires

| Nom         | Formule                                               | Utilisation typique                  |
|-------------|--------------------------------------------------------|--------------------------------------|
| Cosine      | $  \cos(\theta) = \frac{x \cdot y}{\|x\| \|y\|} $      | Texte, TF-IDF                        |
| Jaccard     | $  \frac{\|A \cap B\|}{\|A \cup B\|} $                     | Données binaires, ensembles          |
| Hamming     | $  \frac{\text{différences}}{n} $                    | Variables binaires ou catégorielles  |

## 🎯 Choix de la métrique selon le type de données

| Type de données            | Métriques adaptées                  |
|----------------------------|-------------------------------------|
| Numériques continues       | Euclidienne, Manhattan              |
| Texte / vecteurs creux     | Cosine                              |
| Binaires                   | Jaccard, Hamming                    |
| Mixtes (num. + catég.)     | Gower (ou combinaison personnalisée) |

---

## 🧠 Remarques

- Une **mauvaise métrique** peut conduire à un **clustering incohérent**
- Certaines méthodes (comme K-Means) supposent implicitement une distance euclidienne
- Des algorithmes comme **DBSCAN** ou **CAH** permettent d’utiliser d’autres métriques

