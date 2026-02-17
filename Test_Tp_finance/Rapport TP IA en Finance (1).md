

Rapport TP IA en Finance
## Nom :
## Prénom :
## Filière :
## Date :
Titre du TP :
## Introduction
Ce TP vise à combiner des notions d’analyse statistique et de modélisation supervisée en finance. Les
objectifs sont (1) d’étudier des mesures de risque de portefeuille (moyenne, variance, VaR, Sharpe) afin
de conseiller un client investisseur, (2) d’appliquer le théorème de Bayes pour mettre à jour des
probabilités de défaut de crédit séquentiellement, et (3) de construire un modèle KNN de scoring crédit,
d’optimiser son hyperparamètre $K$ par validation croisée, puis d’évaluer sa performance (matrice de
confusion, AUC/ROC, seuil de décision).
Le contexte métier couvre la gestion de portefeuilles d’actions et le scoring crédit dans le secteur
bancaire. Par exemple, on étudie deux portefeuilles d’actions (l’un conservateur, l’autre agressif) pour
vérifier lequel respecte une contrainte de VaR à 10%. Ensuite, on examine comment l’information
additionnelle (retard de paiement puis découvert) modifie la probabilité de défaut d’un client via Bayes.
Enfin, on s’intéresse à l’impact d’un modèle KNN de credit scoring sur le ROI annuel en comparant vrais/
faux positifs (détections de défaut) et faux négatifs (défauts non détectés), selon différents seuils.
Ce rapport détaille chaque question du TP : explications pédagogiques, interprétations financières,
résultats chiffrés (tableaux formattés), formules en LaTeX, extraits de code Python, et graphiques
(histogrammes, boxplots, courbes) pour illustrer les analyses.
Partie 1 — Statistiques descriptives et VaR
Explication pédagogique
Dans un premier temps, on calcule des statistiques descriptives des rendements mensuels pour deux
portefeuilles   d’actions   historiques   (Portefeuille   A:   conservateur;   Portefeuille   B:   agressif).   Les
mesures clés sont la moyenne et l’écart-type mensuels, la médiane, ainsi que leur annualisation
(rendement et volatilité annuels) via capitalisation composée :
On calcule aussi la Value-at-Risk (VaR) paramétrique à 95%, suivant la formule
avec $1.65$ le quantile 5% de la loi normale. Cette VaR (mensuelle et annualisée) permet de mesurer la
perte maximale tolérable à 95% de confiance. On teste la normalité des rendements (test Shapiro-Wilk)
pour juger de la fiabilité de la VaR paramétrique (une absence de normalité peut invalider l’approche).
Enfin, on calcule le ratio de Sharpe $(R_{\text{annuel}} - r_f)/\sigma_{\text{annuel}}$ (avec taux sans
## R

## =
annuel
## (1+R

## )−
mensuel
## 12
## 1,σ

## =
annuel
σ

## .
mensuel
## 12
VaR

## =
## 95%
μ−1.65σ,
## 1

risque $r_f=3\%$) pour comparer rendement ajusté du risque. Tous ces calculs sont effectués dans le
code suivant.
Résultats et interprétations
Les  rendements   mensuels   moyens  sont   plus   élevés   pour   le   portefeuille   B   (agressif)   qu’A
(conservateur), mais sa volatilité l’est aussi. Les histogrammes superposés (vert pour A, rouge pour B)
illustrent   ces   distributions   (figure   ci-dessous)   :   on   voit   que   A   présente   surtout   des   rendements
légèrement positifs concentrés autour de 1%, tandis que B a des rendements plus extrêmes (positifs et
négatifs). Les boxplots comparatifs confirment cette observation (outliers plus nombreux en B).
Figure : Distributions des rendements mensuels des portefeuilles A (vert) et B (rouge) avec leurs moyennes
(lignes en pointillé).
Les résultats chiffrés (arrondis) sont :
-   Portefeuille   A   (conservateur)  :   moyenne   mensuelle   ≈  0,83%,   écart-type   mensuel   ≈  1,02%,
rendement annuel ≈ 10,31%, volatilité annuelle ≈ 3,53%. Médiane ≈ 0,95%.
-  Portefeuille B (agressif) : moyenne mensuelle ≈ 1,31%, écart-type mensuel ≈ 4,18%, rendement
annuel ≈ 17,05%, volatilité annuelle ≈ 14,42%. Médiane ≈ 4,05%.
La VaR 95% mensuelle est calculée avec $z_{0.05}=-1.645$ :
- A : $\mu_A + z\sigma_A \approx 0.83\% + (-1.645)(1.02\%) = -0.84\%$. En perte (capital €500k), VaR
mensuelle ≈ €4 210.
- B : $\mu_B + z\sigma_B \approx 1.31\% + (-1.645)(4.18\%) = -5.58\%$. En € : €27 900.
La VaR annuelle (en incorporant moyennes et volat annualisées) vaut environ :
- A : $R_{ann} + z\sigma_{ann} \approx 10.31\% + (-1.645)(3.53\%) = 4.57\%$ (soit perte €22 849).
## - B : $17.05\% + (-1.645)(14.42\%) = -6.65\%$ (perte €33 242).
La contrainte client était perte annuelle ≤ €50 000 (10% du capital). Les calculs montrent que la VaR
annuelle du portefeuille A (€−22k) respecte largement cette contrainte (✓), tandis que celle de B (€−33k)
est aussi respectée ici. Toutefois, la normalité des rendements n’est valide que pour A : le test de
Shapiro donne p>0.05 pour A (compatible normale) mais p<0.05 pour B (non normale). Cela signifie
que la VaR paramétrique de B (basée sur loi normale) est moins fiable.
Pour comparer l’efficience risque-rendement, on calcule le Sharpe (formule vue dans le code) :
Le portefeuille A a un Sharpe supérieur à 1 (excellent), B environ 1 (bon). En croisant ces critères (VaR
respectée et Sharpe élevé), le code recommande finalement le portefeuille A pour un client
conservateur. En synthèse (tableau ci-dessous extrait du code) :
CritèrePortefeuille APortefeuille B
Rendement annuel10.31%17.05%
Volatilité annuelle3.53%14.42%
## 12
## 34
## 34
## 2
## 5
## 6
## Sharpe

## =
## A

## ≈
## 3.53
## 10.31−3.00
2.07 ,Sharpe

## =
## B

## ≈
## 14.42
## 17.05−3.00
## 0.97.
## 78
## 2

CritèrePortefeuille APortefeuille B
VaR 95% annuelle (€)
## −22 849 €−33 242 €
Contrainte VaR respectée
## ✓ Oui✓ Oui
## Ratio Sharpe2.0680.969
Normalité (p-value)0.124 (normale)0.000 (non normale)
Interprétation métier : Le portefeuille B offre un rendement théorique plus élevé, mais à un risque
volatil   et   non-gaussien   élevé.   Le   client   qui   tolère   une   perte   max   de   10%   du   capital   est   plutôt
conservateur ; on privilégie donc A qui maximise le ratio Sharpe et garantit une VaR sous le seuil. B n’est
pas recommandé malgré son rendement supérieur, car son risque est plus grand (la forte volatilité et
non-normalité impliquent qu’on pourrait subir des pertes plus sévères en pratique).
Partie 2 — Bayes et scoring crédit
Explication pédagogique
Dans cette partie, on utilise le théorème de Bayes pour mettre à jour les probabilités de défaut d’un
client en fonction d’événements observés. On considère initialement un client standard avec probabilité
a priori de défaut $P(\text{Défaut})=5\%$.
Question 2.1 (manuel) : Supposons que ce client a eu un retard de paiement (événement $E_1$). Les
conditions initiales données sont $P(E_1|\text{Défaut})$ et $P(E_1|\neg\text{Défaut})$ (valeurs dans le
code du TP). On applique la formule de Bayes :
Le code imprime ce calcul détaillé. Le résultat est un posteriori autour de 29,63%. Ceci se compare
exactement à la notion de précision d’un classifieur : la précision = TP/(TP+FP) pour ce cas (voir Question
2.4). On observe que $P(\text{Défaut}|E_1)\approx 0.2963$.
Question 2.2 (sécuentiel) : Deux semaines plus tard, le même client présente un découvert > €500
(événement $E_2$). On prend le posterior $0.2963$ comme nouveau prior. On réapplique Bayes avec
$P(E_2|\cdot)$ (données du tableau du TP). Le calcul Bayésien donne une nouvelle probabilité a
posteriori d’environ 64.60% (29.63% → 64.60%). Le code du notebook affiche étape par étape les calculs
bayésiens  et trace la probabilité de défaut en fonction des étapes (graphe ci-dessous). On voit
l’évolution : - Étape 0 (prior initial 5%) - Étape 1 (après retard) ≈ 29.63% - Étape 2 (après découvert) ≈
## 64.60%.
Figure : Évolution séquentielle de $P(\text{Défaut})$ avec les événements observés (Bayes th. du risque crédit)
## .
Interprétation   métier   :  Chaque   information   négative   augmente   le   risque   client   de   manière
multiplicative.   Le   code   calcule   aussi   les   facteurs   d’augmentation   cumulatifs.   Après   ces   deux
événements, la probabilité de défaut est environ 13 fois supérieure à l’initial (64.6% vs 5%). Des seuils
métiers (15% pour surveillance, 30% pour restrictions) sont indiqués sur le graphique. À 64.6%, le client
P(Dfaut  ∣E

## )=eˊ
## 1

## .
## P(E

∣Dfaut  )P(Dfaut  )+P(E

∣¬ Dfaut  )(1−P(Dfaut  ))
## 1
eˊeˊ
## 1
eˊeˊ
## P(E

∣Dfaut  )P(Dfaut  )
## 1
eˊeˊ
## 9
## 10
## 9
## 11
## 1213
## 14
## 3

dépasse largement ces seuils, ce qui suggère un refus de crédit immédiat pour cause de risque trop
élevé.
Question 2.3 (fonctions) : Le notebook définit une fonction générique bayes_update(prior, P(E|
A), P(E|¬A)) (code non affiché ici) avec docstring (commentaires) et tests. Elle reprend la formule :
$P(A|E)=P(E|A)P(A)/P(E)$. Ceci permet de faire les mises à jour plus systématiquement pour d’autres
événements ou clients.
Question 2.4 (matrice confusion) : On considère maintenant un modèle de prédiction (par ex. naive
Bayes) dont la matrice de confusion sur 10 000 clients teste donne : 400 vrais positifs (TP), 950 faux
positifs (FP), 100 faux négatifs (FN) et 8550 vrais négatifs (TN). On calcule la  précision  (positive
predictive value) = TP/(TP+FP) = 400/1350 ≈ 0.2963. On note que c’est exactement la même valeur
que $P(\text{Défaut}|E_1)$ calculée Bayésiennement (29.63%). Le notebook souligne ce lien conceptuel
: en classification, la précision mesure $P(\text{vrai défaut}|\text{prédiction défaut})$, équivalant
bien à la probabilité bayésienne posterior calculée.
Par   conséquent,   l’exercice   met   en   évidence   que   l’apprentissage   probabiliste   (naive   Bayes)   et
l’interprétation statistique (Bayes manuel) sont cohérents. Il illustre que l’optimisation des probabilités a
posteriori revient à maximiser la précision du modèle de scoring.
Partie 3 — K-Nearest Neighbors et évaluation
Explication pédagogique
Enfin, on crée un  dataset synthétique  de clients avec variables socio-économiques (âge, salaire,
ancienneté, endettement, etc.) et un indicateur binaire de défaut. Le code a généré 2000 clients selon la
procédure suivante : on simule $P(\text{défaut})$ de base (5%) et on augmente la probabilité pour les
clients à risque élevé (par ex. ratio dette/revenu>0.5, retards >3, score crédit <600, etc.). Le code
Python (voir correction) résume la génération et sauvegarde dans credit_data.csv.
Q3.1 Exploration :  On charge ce dataset, on affiche les cinq premières lignes et les statistiques
descriptives (non montrées ici). Le taux de défaut global est d’environ 16.7%. On étudie les corrélations :
par exemple, le ratio dette/revenu et l’historique de retards sont fortement corrélés avec la cible «
défaut ». On trace une heatmap des corrélations entre toutes les variables (ci-dessous). On réalise
également des boxplots de, disons,  ratio_dette_revenu  et  historique_retards  par classe
défaut (non montrés ici), pour visualiser que les défaillants ont typiquement un ratio dette/revenu plus
élevé et plus de retards.
(extrait du code de génération de stats) montre comment ces statistiques sont calculées. Par
exemple, le code pour calculer et créer le DataFrame final est :
df  = pd .DataFrame({
'age': age  ,'salaire': salaire,'anciennete_emploi': anciennete,
'dette_totale': dette_totale,'ratio_dette_revenu': ratio,
'nb_credits_actifs': nb_credits,'historique_retards': historique,
'score_credit_bureau': score_credit,'defaut': defaut
## })
## .
## 15
## 16
## 17
## 18
## 18
## 19
## 20
## 1
## 2021
## 4

## Q3.2 Prétraitement
On sépare ensuite X/y   et on effectue un split train/test (70/30) stratifié pour préserver la proportion de
défaut (~16.7% dans chaque jeu). Le jeu d’entraînement compte 1400 clients, test 600, avec le même
taux   de   défaut.   On   applique   un  StandardScaler  pour   normaliser   toutes   les   caractéristiques
numériques (fit sur train, transform train et test). Ceci est fait afin que chaque variable ait moyenne 0 et
écart-type 1 sur le train. Le code confirme que les moyennes centrées sont proches de zéro après
scalage.
Q3.3 Optimisation de $K$
On cherche le meilleur nombre de voisins $K$ pour le KNN. Le critère principal est la performance AUC
en validation croisée 5-fold sur le train. On teste $K$ de 1 à 30. Pour chaque $K$, on obtient la moyenne
des scores AUC, Recall, et Précision sur les 5 folds. Par exemple, la procédure est :
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 )
forKinrange(1,31 ):
aucs, precs, recs= [],[],[]
for(train_idx,val_idx)incv .split(X_train_scaled, y_train):
model= KNeighborsClassifier(n_neighbors=K)
model.fit  (X_train_scaled[train_idx],  y_train.values[train_idx])
probs= model.predict_proba(X_train_scaled[val_idx])[:,1]
pred= model.predict(X_train_scaled[val_idx])
aucs.append(roc_auc_score(y_train.values[val_idx],  probs))
precs.append(precision_score(y_train.values[val_idx],  pred,
zero_division=0))
recs.append(recall_score(y_train.values[val_idx],  pred,
zero_division=0))
# enregistre moyennes dans un DataFrame...
Le code stocke pour chaque $K$ la moyenne et l’écart-type de l’AUC, Recall et Précision.
En pratique, on observe (voir graphe) que l’AUC croît jusqu’à environ $K=25$–30 où elle se stabilise
autour de 0.62. La meilleure AUC moyenne obtenue est pour $K\approx27$ (AUC${\max}\approx0.619$).
Nous choisissons donc $K=27$. Ce choix maximise la capacité discriminante (AUC). Le graphique
suivant (moyenne AUC vs K) illustre cette optimisation :}
Figure : Validation croisée 5-fold – AUC moyen vs $K$ pour KNN. Le $K$ optimal (ici 27) maximise l’AUC.
Q3.4 Évaluation du modèle final
On entraîne le KNN final avec $K=27$ sur tout le jeu train. Sur le jeu test, la prédiction (au seuil 0.5)
donne malheureusement 0 faux positifs et   0 vrais positifs : le modèle prédit systématiquement non-
défaillance (majoritaire). La matrice de confusion est donc :
TN = 500 (clients sans défaut bien classés)
## FP = 0
FN = 100 (tous les défauts ratés)
## TP = 0
## 22
## 23
## •
## •
## •
## •
## 5

Le modèle à $K=27$ est trop « conservatif » sur ce dataset. Les métriques sur test sont : précision (pour
la classe défaut) nulle (pas de vrais positifs), rappel = 0, F1 = 0. L’AUC sur test reste modeste (~0.59). Le
classification_report ne montre aucun score positif détecté.
On constate qu’avec $K=27$, le KNN est très parcimonieux. En pratique, on devrait repenser $K$ ou
considérer un seuil plus bas pour lever plus de défaillants.
Q3.5 Courbe ROC et seuil de décision
Pour analyser le compromis sensibilité/spécificité, on trace la courbe ROC (TPR vs FPR) du modèle
$K=27$ (ci-dessous). L’AUC-ROC test ≈ 0.59   (faible). L’indice de Youden ($\text{TPR}-\text{FPR}$) permet
de trouver un seuil optimal (non montré ici), mais on veut surtout garantir un Recall ≥ 80% (sécurité
bancaire).
En testant différents seuils de probabilité (0.3, 0.5, 0.7), on obtient par ex. (approx.) :
Seuil de décisionRecall (%)Précision (%)
0.5 (par défaut)0%–
## 0.38%29%
## 0.239%20%
## 0.183%19%
Au seuil 0.1, on atteint un rappel de ~83% (c.-à-d. on détecte 83 des 100 défauts réels) mais la précision
chute (~19%). Ce seuil serait le minimum pour atteindre l’objectif de 80%.
Q3.6 ROI et recommandations (Executive Summary)
En contexte bancaire, on compare les gains et coûts métier suivants : chaque défaut détecté (TP)
évite une perte moyenne de €15 000 ; chaque défaut manqué (FN) coûte €15 000 (perte du principal). De
plus, chaque cliente nécessitant analyse approfondie (FP) coûte €500, et chaque refus inutile (FP) coûte
€1 200 de marge perdue. Le ROI net annuel est donc calculé par :
En appliquant ces formules sur les seuils testés, on trouve (sur l’échantillon test) qu’un seuil trop élevé
(0.5) ne détecte aucun défaut (ROI net = 0). Un seuil moyen (0.3–0.2) génère encore un ROI négatif (plus
de coûts que de gains). Le seuil 0.1 donne, en revanche, un ROI positif : par exemple, TP=83 (gain
1 245k€), FP=348 (coûts 3481700=591.6k€), FN=17 (pertes 255k€), ce qui aboutit à un gain net ~+572k€* sur
base 600 clients test, soit ROI annuel ~~0.38 (38%) par rapport aux pertes initiales.
Executive Summary (Q3.6d) : Le $K$ optimal recommandé est 27 , car il maximise l’AUC en validation
(AUC≈0.62). À ce $K$, on observe en test une AUC-ROC ≈0.59, rappel initial de 0% (aucun défaut
capté par le seuil 0.5) et précision non définie (0 TP). Pour être utile, il faut abaisser le seuil de décision.
Au seuil ~0.1, on atteint un rappel de ~83% avec précision ~19%, satisfaisant la contrainte métier «
détecter ≥80% des défauts ». Les métriques clés de ce modèle (avec $K=27$) sont donc : AUC ≈0.59,
## Recall ≈0.83,  Précision ≈0.19.
## 24
## 22
ROI net=(TP×15000)−(FP×(500+1200))−(FN×15000).
## 22
## 6

Sur le plan financier, un tel modèle rapporte un ROI annuel significatif : en détectant 83% des défauts
réels, on évite ~€1,245k de pertes sur défauts, pour un coût d’environ €591k (analyses + opportunités
perdues), et on subit €255k de pertes des défauts manqués. Le gain net annuel simulé est de ~€572k
(en données test), soit un ROI ~38%. On recommande donc d’adopter  $K=27$  avec un seuil
décisionnel bas (~0.1) pour prioriser la détection des défauts. L’impact business attendu est une
réduction substantielle des pertes sur prêts (grâce à la détection des gros risques), au prix d’un certain
nombre de faux positifs (prêts refusés) tolérable, car le gain net demeure positif. Cette stratégie
permettra de diminuer le taux de défaut effectif de la banque tout en optimisant le retour sur capital.
## Conclusion
Ce TP a permis d’appliquer concrètement plusieurs concepts d’IA et de statistiques en finance : calcul de
VaR et Sharpe pour la gestion de portefeuille, usage du théorème de Bayes dans le scoring crédit, et
entraînement d’un modèle KNN pour classification. Les apprentissages clés sont :  l’importance de
mesurer   et   respecter   les   contraintes   clients   (VaR   ≤   perte   tolérée)  ;     l’interprétation   des
probabilités   Bayésiennes   comme   des   métriques   de   classification   (precision)  ;     la   nécessité
d’optimiser les hyperparamètres ML (ici $K$) par validation croisée  ; et  l’évaluation des coûts
métier (ROI) pour choisir le seuil décisionnel.
Les principales difficultés rencontrées concernent l’interprétation des métriques financières (par ex.
passage du taux mensuel annuel, définition de la VaR) et la prise en compte des déséquilibres (très peu
de défauts mène à beaucoup de faux négatifs avec KNN). En pratique bancaire, ces outils peuvent être
déployés pour affiner le scoring crédit et le conseil en investissement : un analyste pourrait utiliser les
VaR et Sharpe pour conseiller un client sur un portefeuille adapté à son profil de risque, et un
département risque pourrait calibrer un modèle KNN pour le scoring auto afin de minimiser les pertes
sur prêts (en optimisant AUC/recall vs coût des faux positifs).
## Références
Cours d’IA / statistiques financières – formules de VaR, Sharpe, Bayes.
Documentation Python 3 (NumPy, pandas, Matplotlib, SciPy) et scikit-learn (KNN, roc_auc_score,
train_test_split, StandardScaler, confusion_matrix, classification_report).
Docs de scikit-learn sur KNeighborsClassifier, roc_auc_score, etc..
Chapitre code du notebook fourni pour calculs VaR et Sharpe, Bayes, et KNN.
Tp1- Apprentissage Supervisé _ KNN, Évaluation et
## Surapprentissage.docx
file://file_00000000065c7243ac68c9b9c4845930
TP_IA_en_finance.ipynb
file://file-WSbvMnCYtHqcQ4k8xSdhL5
## 2422
## •
## •
## •
## •
## 2492522
## 13461920212224
## 257891011121314151617182325
## 7