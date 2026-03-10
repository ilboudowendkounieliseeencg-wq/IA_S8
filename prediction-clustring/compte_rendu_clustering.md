# Compte Rendu — Algorithmes de Classification Non Supervisée (Clustering)

**Date :** Mars 2026
**Sujet :** Présentation, fonctionnement et comparaison des principaux algorithmes de clustering
**Outils :** Python, Scikit-learn, Datasets réels (Iris, Wine, Digits)

---

## Introduction

Le **clustering** (ou classification non supervisée) est une branche fondamentale de l'apprentissage automatique. Contrairement à la classification supervisée, aucune étiquette n'est fournie à l'algorithme : il doit **découvrir lui-même la structure cachée** dans les données en regroupant les observations similaires au sein de mêmes groupes (clusters) et en séparant les observations dissimilaires.

Les applications sont nombreuses : segmentation de clients, détection d'anomalies, compression d'images, analyse de gènes, recommandation de contenu, traitement du langage naturel, etc.

Ce compte rendu présente les principaux algorithmes de clustering, leur fonctionnement, leurs avantages, leurs limites et leurs cas d'usage.

---

## Métriques d'évaluation

Avant de décrire les algorithmes, il est important de connaître les métriques utilisées pour évaluer la qualité d'un clustering sans étiquettes de référence.

### Score de Silhouette
Pour chaque point i, le score de silhouette est :

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

- **a(i)** : distance moyenne de i à tous les autres points de son cluster (cohésion).
- **b(i)** : distance moyenne de i aux points du cluster voisin le plus proche (séparation).
- **s(i) ∈ [-1, 1]** : proche de 1 → bien classé ; proche de 0 → frontière ; négatif → mal classé.

### Inertie (WCSS — Within-Cluster Sum of Squares)
Somme des distances au carré entre chaque point et le centroïde de son cluster. Plus petite = meilleur regroupement intra-cluster. Utilisée principalement avec K-Means.

### BIC / AIC (pour les modèles probabilistes)
Critères de sélection du modèle pénalisant la complexité. Utiles pour GMM.

---

## 1. K-Means

### Principe
K-Means est l'algorithme de clustering le plus populaire. Il partitionne un ensemble de données en **K clusters** en minimisant la variance intra-cluster (inertie). Il repose sur la notion de **centroïde** : le centre de chaque cluster est la moyenne des points qui lui appartiennent.

### Algorithme (Lloyd's algorithm)
1. Initialiser K centroïdes aléatoirement (ou via K-Means++).
2. **Assignation** : affecter chaque point au centroïde le plus proche (distance euclidienne).
3. **Mise à jour** : recalculer chaque centroïde comme la moyenne des points assignés.
4. Répéter les étapes 2–3 jusqu'à convergence (les assignations ne changent plus).

### Paramètres clés
| Paramètre | Description |
|---|---|
| `n_clusters` (K) | Nombre de clusters à former |
| `init` | Méthode d'initialisation (`k-means++` recommandé) |
| `n_init` | Nombre de répétitions avec initialisations différentes |
| `max_iter` | Nombre maximum d'itérations |

### Choix de K : Méthode Elbow
On trace l'inertie en fonction de K. Le "coude" de la courbe indique le K optimal — au-delà, le gain marginal diminue fortement.

### Complexité
- **Temporelle** : O(n × K × d × i) où n = points, d = dimensions, i = itérations.
- **Spatiale** : O(n × K).

### Avantages
- Simple, rapide et scalable sur de grands datasets.
- Garantit la convergence.
- Fonctionne bien sur des clusters sphériques et bien séparés.

### Limites
- K doit être fixé à l'avance.
- Sensible aux outliers (les centroïdes sont tirés vers eux).
- Suppose des clusters convexes et de taille similaire.
- Minimum local possible (dépend de l'initialisation).

### Cas d'usage
Segmentation client, compression d'images, quantification vectorielle, pré-traitement pour d'autres algorithmes.

---

## 2. K-Medoids (PAM — Partitioning Around Medoids)

### Principe
K-Medoids est une variante de K-Means où les centres de clusters sont des **points réels du dataset** (appelés médoïdes), et non des moyennes fictives. Il minimise la somme des dissimilarités entre chaque point et son médoïde.

### Algorithme
1. Sélectionner K points aléatoires comme médoïdes initiaux.
2. **Assignation** : affecter chaque point au médoïde le plus proche.
3. **Optimisation** : pour chaque médoïde m et chaque non-médoïde o, calculer le coût d'échange. Si l'échange réduit le coût total, effectuer l'échange.
4. Répéter jusqu'à convergence.

### Différence avec K-Means
| | K-Means | K-Medoids |
|---|---|---|
| Centre | Moyenne (point fictif) | Point réel du dataset |
| Robustesse aux outliers | Faible | Élevée |
| Mesure de distance | Euclidienne uniquement | Toute métrique |
| Complexité | O(nKdi) | O(K(n-K)²) |

### Avantages
- Robuste aux outliers et au bruit.
- Fonctionne avec n'importe quelle métrique de distance.
- Les centres sont des points réels, donc interprétables.

### Limites
- Beaucoup plus lent que K-Means sur de grands datasets.
- K doit être fixé à l'avance.

### Cas d'usage
Analyse de données médicales, clustering de séquences biologiques, données avec beaucoup d'outliers.

---

## 3. K-Medians

### Principe
K-Medians est similaire à K-Means mais utilise la **médiane** au lieu de la **moyenne** pour calculer les centres de clusters. La fonction objectif minimise la somme des distances L1 (Manhattan) entre chaque point et le centre de son cluster.

### Algorithme
1. Initialiser K centres.
2. **Assignation** : affecter chaque point au centre le plus proche (distance Manhattan).
3. **Mise à jour** : recalculer chaque centre comme la **médiane** (composante par composante) des points assignés.
4. Répéter jusqu'à convergence.

### Distance L1 vs L2
| | K-Means (L2) | K-Medians (L1) |
|---|---|---|
| Fonction de coût | Somme des carrés | Somme des valeurs absolues |
| Centre | Moyenne | Médiane |
| Robustesse | Faible | Élevée |

### Avantages
- Plus robuste aux outliers que K-Means (la médiane est insensible aux valeurs extrêmes).
- Mieux adapté aux distributions non-gaussiennes.

### Limites
- Convergence plus lente que K-Means.
- K doit être fixé à l'avance.
- Moins de support dans les bibliothèques standard.

### Cas d'usage
Données financières avec valeurs aberrantes, analyse de signaux bruités.

---

## 4. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

### Principe
DBSCAN est un algorithme basé sur la **densité locale** des points. Il regroupe les points qui sont densément connectés et marque comme **bruit** les points isolés. Il peut détecter des clusters de **forme arbitraire**.

### Concepts fondamentaux
- **ε (epsilon)** : rayon du voisinage d'un point.
- **MinPts** : nombre minimum de points dans le voisinage ε pour qu'un point soit considéré comme point central.
- **Point central (core point)** : a au moins MinPts voisins dans son rayon ε.
- **Point frontière (border point)** : dans le rayon ε d'un point central, mais sans assez de voisins lui-même.
- **Point bruit (noise point)** : ni central ni frontière.

### Algorithme
1. Pour chaque point non visité, récupérer son voisinage ε.
2. Si le point a ≥ MinPts voisins → **point central** → créer un nouveau cluster.
3. Étendre le cluster en ajoutant tous les points densément atteignables.
4. Sinon → marquer comme **bruit** (peut devenir frontière plus tard).

### Complexité
- **Temporelle** : O(n log n) avec indexation spatiale (K-d tree), O(n²) sinon.

### Avantages
- Détecte automatiquement le nombre de clusters.
- Gère les clusters de forme arbitraire (non-convexes).
- Robuste au bruit et aux outliers (marqués comme bruit).
- Ne nécessite pas de spécifier K.

### Limites
- Sensible aux paramètres ε et MinPts (difficiles à choisir).
- Difficultés avec des clusters de densités très variables.
- Mauvaises performances en haute dimension (malédiction de la dimensionnalité).

### Cas d'usage
Détection d'anomalies, analyse de données géospatiales, segmentation d'images, clustering de trajectoires.

---

## 5. HDBSCAN (Hierarchical DBSCAN)

### Principe
HDBSCAN est une extension hiérarchique de DBSCAN. Il construit une **hiérarchie de clusters** à différentes densités et extrait les clusters les plus **stables** (persistants) à travers cette hiérarchie. Il est beaucoup plus robuste au choix des paramètres.

### Étapes de l'algorithme
1. **Distance de portée mutuelle** : transformer les distances pour tenir compte de la densité locale.
2. **Arbre couvrant minimum** : construire un MST sur le graphe des distances mutuelles.
3. **Hiérarchie de clusters** : transformer le MST en dendrogramme de clusters.
4. **Condensation** : simplifier la hiérarchie en supprimant les clusters de petite taille.
5. **Extraction** : sélectionner les clusters stables (maximisant la persistance).

### Différence avec DBSCAN
| | DBSCAN | HDBSCAN |
|---|---|---|
| Paramètres | ε et MinPts | Uniquement MinPts |
| Hiérarchie | Non | Oui |
| Densités variables | Non | Oui |
| Stabilité | Sensible | Robuste |

### Avantages
- Un seul paramètre principal (`min_cluster_size`).
- Gère des clusters de densités différentes.
- Fournit des probabilités d'appartenance à un cluster.
- Plus robuste que DBSCAN.

### Limites
- Plus lent que DBSCAN.
- Peut sur-segmenter avec des données très bruitées.

### Cas d'usage
Bioinformatique, analyse de texte, clustering de données complexes multi-densité.

---

## 6. OPTICS (Ordering Points To Identify the Clustering Structure)

### Principe
OPTICS est un algorithme basé sur la densité qui génère un **ordonnancement des points** révélant la structure de clustering à différents niveaux de densité. Contrairement à DBSCAN, il ne produit pas directement des clusters mais un **reachability plot** (graphe de portée) qui encode toute la structure hiérarchique.

### Concepts clés
- **Distance centrale (core-distance)** : distance au MinPts-ième voisin le plus proche.
- **Distance de portée (reachability-distance)** : max(core-distance(o), dist(o, p)).
- **Reachability plot** : visualisation de la structure de clustering — les vallées correspondent aux clusters.

### Algorithme
1. Sélectionner un point non traité.
2. Calculer sa distance centrale.
3. Insérer ses voisins dans une file de priorité ordonnée par distance de portée.
4. Traiter les points dans cet ordre, en mettant à jour la file.
5. L'ordre de traitement + les distances de portée constituent le résultat.

### Avantages
- Fonctionne sans fixer ε (seul MinPts est nécessaire).
- Détecte des clusters imbriqués et de densités variables.
- Le reachability plot permet une analyse visuelle riche.

### Limites
- Ne produit pas directement des étiquettes (nécessite une étape d'extraction).
- Plus lent que DBSCAN.
- Interprétation du reachability plot parfois subjective.

### Cas d'usage
Analyse exploratoire, données avec hiérarchies de densité, détection de sous-clusters.

---

## 7. Clustering Hiérarchique Agglomératif (HAC)

### Principe
Le clustering hiérarchique agglomératif (bottom-up) commence avec chaque point comme son propre cluster, puis **fusionne progressivement** les deux clusters les plus similaires jusqu'à obtenir un seul cluster racine. Le résultat est un **dendrogramme** représentant toute la hiérarchie.

### Algorithme
1. Initialiser n clusters (un par point).
2. Calculer la matrice de distances entre tous les clusters.
3. Fusionner les deux clusters les plus proches selon le critère de liaison.
4. Mettre à jour la matrice de distances.
5. Répéter jusqu'à n'avoir qu'un seul cluster.

### Critères de liaison (linkage)
| Méthode | Calcul de la distance entre clusters | Comportement |
|---|---|---|
| **Single** | Distance minimale entre paires de points | Clusters allongés, sensible au bruit |
| **Complete** | Distance maximale entre paires de points | Clusters compacts, sensible aux outliers |
| **Average** | Moyenne des distances entre toutes les paires | Compromis |
| **Ward** | Minimise l'augmentation de la variance totale | Clusters sphériques, le plus utilisé |

### Dendrogramme
Le dendrogramme visualise la hiérarchie des fusions. La hauteur de chaque fusion représente la distance entre les clusters fusionnés. On coupe le dendrogramme à la hauteur souhaitée pour obtenir K clusters.

### Complexité
- **Temporelle** : O(n³) en général, O(n² log n) avec certaines optimisations.
- **Spatiale** : O(n²).

### Avantages
- Pas besoin de spécifier K à l'avance.
- Fournit une hiérarchie complète des clusters.
- Déterministe (pas d'aléatoire).
- Résultat visualisable via le dendrogramme.

### Limites
- Pas scalable pour de très grands datasets (O(n²) en mémoire).
- Irréversible : une mauvaise fusion initiale ne peut être corrigée.
- Sensible aux outliers (surtout single et complete linkage).

### Cas d'usage
Phylogénétique, analyse de documents, segmentation d'images, analyse de gènes.

---

## 8. Gaussian Mixture Models (GMM)

### Principe
GMM est un modèle probabiliste qui suppose que les données sont générées par un **mélange de K distributions gaussiennes**. Contrairement aux méthodes déterministes, GMM effectue un **clustering probabiliste** : chaque point appartient à chaque cluster avec une certaine probabilité.

### Modèle mathématique
La densité de probabilité des données est :

```
p(x) = Σ_{k=1}^{K} π_k × N(x | μ_k, Σ_k)
```

- **π_k** : poids (probabilité a priori) du k-ième composant.
- **μ_k** : vecteur moyenne du k-ième gaussien.
- **Σ_k** : matrice de covariance du k-ième gaussien.

### Algorithme EM (Expectation-Maximization)
**Étape E (Expectation)** : calculer les responsabilités (probabilité a posteriori que chaque point appartienne à chaque composant).

**Étape M (Maximization)** : mettre à jour les paramètres (π_k, μ_k, Σ_k) pour maximiser la log-vraisemblance.

Répéter jusqu'à convergence.

### Types de covariance
| Type | Description | Flexibilité |
|---|---|---|
| `full` | Chaque composant a sa propre matrice de covariance | Maximale |
| `tied` | Tous partagent la même matrice | Moyenne |
| `diag` | Matrices diagonales (features indépendantes) | Réduite |
| `spherical` | Variance scalaire par composant | Minimale |

### Sélection de K : BIC et AIC
- **BIC** (Bayesian Information Criterion) : pénalise les modèles complexes. Préférer le minimum.
- **AIC** (Akaike Information Criterion) : similaire, moins sévère.

### Avantages
- Clustering probabiliste (soft assignment).
- Modélise des clusters elliptiques.
- Fournit des intervalles de confiance.
- Généralisation probabiliste de K-Means.

### Limites
- Suppose une distribution gaussienne des données.
- Sensible à l'initialisation (minimum local).
- K doit être spécifié.
- Peut diverger si une composante capture un seul point.

### Cas d'usage
Modélisation de la parole, vision par ordinateur, segmentation d'images, finance (modélisation des rendements).

---

## 9. Spectral Clustering

### Principe
Le Spectral Clustering exploite la théorie des **graphes spectraux** pour détecter des clusters de formes arbitraires, même non-convexes. L'idée est de transformer les données dans un espace où les clusters deviennent linéairement séparables, en utilisant les **vecteurs propres** du Laplacien du graphe de similarité.

### Algorithme
1. **Construire le graphe de similarité** : relier les points selon une fonction d'affinité (RBF, k-NN voisins).
2. **Calculer la matrice de similarité** W et le Laplacien L = D - W (D = matrice des degrés).
3. **Calculer les k plus petits vecteurs propres** de L (ou de la version normalisée).
4. **Projeter** les données dans l'espace des vecteurs propres.
5. **Appliquer K-Means** dans ce nouvel espace.

### Laplacien du graphe
```
L = D - W
```
Les vecteurs propres du Laplacien encodent la connectivité du graphe : les points du même cluster ont des valeurs propres similaires.

### Fonctions d'affinité
| Méthode | Description |
|---|---|
| `rbf` | Noyau gaussien : exp(-γ ‖x-y‖²) |
| `nearest_neighbors` | 1 si dans les k plus proches voisins, 0 sinon |
| `precomputed` | Matrice d'affinité fournie directement |

### Avantages
- Détecte des clusters de formes arbitraires (y compris concentriques).
- Pas d'hypothèse sur la forme des clusters.
- Efficace sur des données de basse à moyenne dimension.

### Limites
- Calcul des vecteurs propres coûteux pour de grands n (O(n³)).
- K doit être spécifié.
- Sensible au paramètre d'affinité.
- Pas scalable pour de très grands datasets.

### Cas d'usage
Segmentation d'images, analyse de réseaux sociaux, clustering de données non-convexes, traitement du signal.

---

## 10. Affinity Propagation

### Principe
Affinity Propagation est un algorithme basé sur l'**échange de messages** entre paires de points. Il ne nécessite pas de spécifier K : il découvre automatiquement le nombre de clusters en identifiant des **exemplaires** (points représentatifs de chaque cluster).

### Messages échangés
Deux types de messages sont échangés entre tous les paires (i, k) :

- **Responsabilité r(i, k)** : mesure combien le point k est adapté comme exemplaire pour i (compétition avec d'autres candidats).
- **Disponibilité a(i, k)** : mesure combien il est approprié pour i de choisir k comme exemplaire (support des autres points).

### Algorithme
1. Initialiser les disponibilités à 0.
2. **Mise à jour des responsabilités** : r(i,k) = s(i,k) - max_{k'≠k}[a(i,k') + s(i,k')]
3. **Mise à jour des disponibilités** : a(i,k) = min(0, r(k,k) + Σ_{i'≠i,k} max(0, r(i',k)))
4. **Amortissement** : λ × ancien + (1-λ) × nouveau (évite les oscillations).
5. Répéter jusqu'à convergence.

Les exemplaires sont les points k tels que a(k,k) + r(k,k) > 0.

### Paramètres clés
| Paramètre | Description |
|---|---|
| `damping` | Facteur d'amortissement λ ∈ [0.5, 1) |
| `preference` | Préférence initiale (contrôle le nombre de clusters) |

### Avantages
- Nombre de clusters déterminé automatiquement.
- Garantit que les centres sont des points réels (exemplaires).
- Pas d'initialisation aléatoire (déterministe).

### Limites
- Complexité O(n²) en temps et mémoire.
- Pas scalable pour de grands datasets.
- Sensible au paramètre `preference`.
- Peut produire trop ou trop peu de clusters.

### Cas d'usage
Analyse de données biologiques, sélection de représentants, résumé automatique de documents.

---

## 11. Self-Organizing Maps (SOM)

### Principe
Les Self-Organizing Maps (cartes auto-organisatrices), inventées par Teuvo Kohonen, sont un type de **réseau de neurones non supervisé** qui apprend à projeter des données de haute dimension sur une **grille 2D** (ou 1D) tout en préservant la topologie des données originales.

### Architecture
- **Couche d'entrée** : vecteurs de données.
- **Couche de sortie** : grille de neurones (m×n), chacun possédant un vecteur de poids de même dimension que les données.

### Algorithme d'apprentissage
1. Initialiser les poids aléatoirement.
2. Pour chaque échantillon x :
   a. **Trouver le BMU** (Best Matching Unit) : le neurone dont le poids est le plus proche de x.
   b. **Mettre à jour** les poids du BMU et de ses voisins (selon une fonction de voisinage gaussienne) : w(t+1) = w(t) + α(t) × h(BMU, j, t) × (x - w(t))
3. Décroître le taux d'apprentissage α et le rayon de voisinage au fil du temps.

### U-Matrix
La U-Matrix (Unified Distance Matrix) visualise les distances entre neurones voisins. Les zones sombres indiquent des frontières entre clusters ; les zones claires indiquent des régions homogènes.

### Avantages
- Préserve la topologie des données.
- Visualisation intuitive en 2D.
- Fonctionne bien en haute dimension.
- Pas besoin de spécifier K explicitement.

### Limites
- Plusieurs hyperparamètres à régler (taille de grille, σ, α).
- Convergence non garantie vers un optimum global.
- Frontières de clusters parfois floues.

### Cas d'usage
Visualisation de données de haute dimension, analyse de données génomiques, reconnaissance de formes, analyse de marché.

---

## 12. UMAP + K-Means (Réduction Dimensionnelle + Clustering)

### Principe
UMAP (Uniform Manifold Approximation and Projection) est une technique de **réduction de dimension non linéaire** qui préserve à la fois la structure locale et globale des données. Combiné avec K-Means, il permet de clustériser efficacement des données de **très haute dimension**.

### UMAP — Fonctionnement
1. **Construction du graphe local** : pour chaque point, trouver ses k plus proches voisins et construire un graphe pondéré représentant la structure locale.
2. **Optimisation de la représentation 2D** : minimiser la divergence entre la structure du graphe original et la représentation basse dimension via une descente de gradient.

### Pipeline UMAP + K-Means
```
Données HD (n × d) → UMAP → Données 2D (n × 2) → K-Means → Clusters
```

### Comparaison UMAP vs t-SNE
| | UMAP | t-SNE |
|---|---|---|
| Vitesse | Rapide | Lent |
| Structure globale | Préservée | Peu préservée |
| Déterminisme | Semi-déterministe | Non déterministe |
| Paramètres | `n_neighbors`, `min_dist` | `perplexity` |
| Applicable au clustering | Oui | Limité |

### Avantages
- Très efficace pour des données de haute dimension (images, texte, génomique).
- Préserve la structure locale ET globale.
- Rapide et scalable.
- La représentation 2D est directement visualisable.

### Limites
- La réduction dimensionnelle peut déformer les distances originales.
- Les résultats dépendent des hyperparamètres.
- Interprétation des distances absolues difficile.

### Cas d'usage
Analyse de données scRNA-seq (single-cell RNA), classification d'images, NLP (embeddings de mots), analyse de données multiomiques.

---

## Tableau Comparatif Global

| Algorithme | Type | K requis | Forme clusters | Outliers | Scalabilité | Complexité |
|---|---|---|---|---|---|---|
| K-Means | Centroïde | Oui | Convexe | Sensible | Très haute | O(nKdi) |
| K-Medoids | Centroïde | Oui | Convexe | Robuste | Moyenne | O(K(n-K)²) |
| K-Medians | Centroïde | Oui | Convexe | Robuste | Moyenne | O(nKdi) |
| DBSCAN | Densité | Non | Arbitraire | Très robuste | Haute | O(n log n) |
| HDBSCAN | Densité | Non | Arbitraire | Très robuste | Haute | O(n log n) |
| OPTICS | Densité | Non | Arbitraire | Robuste | Moyenne | O(n²) |
| HAC | Hiérarchique | Non* | Arbitraire | Sensible | Faible | O(n³) |
| GMM | Probabiliste | Oui | Elliptique | Sensible | Haute | O(nKdi) |
| Spectral | Graphe | Oui | Arbitraire | Modéré | Faible | O(n³) |
| Affinity Prop. | Message | Non | Arbitraire | Modéré | Faible | O(n²) |
| SOM | Neuronal | Non* | Topologique | Modéré | Moyenne | O(n×m²×i) |
| UMAP + K-Means | Hybrid | Oui | Arbitraire | Modéré | Très haute | O(n log n) |

*Non requis mais peut être déduit du dendrogramme / de la grille.

---

## Guide de Sélection de l'Algorithme

```
Mes données sont-elles de haute dimension (>50) ?
├── Oui → UMAP + K-Means, ou SOM
└── Non ↓

Connais-je le nombre de clusters K ?
├── Non → DBSCAN, HDBSCAN, Affinity Propagation, OPTICS, HAC
└── Oui ↓

Les clusters ont-ils des formes arbitraires (non-convexes) ?
├── Oui → DBSCAN, HDBSCAN, Spectral Clustering
└── Non ↓

Y a-t-il beaucoup de bruit / outliers ?
├── Oui → DBSCAN, HDBSCAN, K-Medoids, K-Medians
└── Non ↓

Ai-je besoin de probabilités d'appartenance ?
├── Oui → GMM, HDBSCAN
└── Non ↓

Ai-je besoin d'une hiérarchie ?
├── Oui → HAC, HDBSCAN, OPTICS
└── Non → K-Means (solution par défaut, rapide et efficace)
```

---

## Conclusion

Il n'existe pas d'algorithme universellement supérieur : le choix dépend de la **nature des données**, de la **connaissance a priori** du problème (nombre de clusters, présence de bruit, forme des clusters) et des **contraintes computationnelles**.

En pratique, il est recommandé de :
1. Commencer par une **analyse exploratoire** (visualisation PCA/UMAP).
2. Tester **plusieurs algorithmes** et comparer les scores de silhouette.
3. Valider les clusters avec une **expertise métier**.
4. Utiliser le **score de silhouette** et la **cohérence métier** comme critères de sélection finaux.

Le clustering est autant un art qu'une science : l'interprétabilité et la pertinence des clusters dans le contexte applicatif sont souvent plus importantes que les métriques brutes.

---

*Compte rendu généré avec Python, Scikit-learn et Matplotlib. Datasets : Iris (Fisher, 1936), Wine (UCI ML Repository), Digits (NIST).*
