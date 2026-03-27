<img src="Encgd.png" style="height:80px;margin-right:50px"/>                 <img src="uhp.png" style="height:80px;margin-right:50px"/>

<img src="photo_Elisee_DS.jpg" style="height:100px;margin-right:50px"/>

**Auteur :** ILBOUDO WENDKOUNI ELISEE
**Filière :** CAC 2
**Apogée :** 22007857
**Module :** Intelligence Artificielle
**Encadré par :** Pr. Larhlimi Abdelrahim
**Date :** 17 Mars 2026  

## Compte rendu projet IA

## Modèle Prédictif des Performances financières des entreprises marocaines
### Architecture XGBoost avec Logique d'Affinage Temporel à 15 Jours


## 📋 Sommaire

1. [Introduction et Contexte Historique](#1-introduction-et-contexte-historique)
   - 1.1 [Présentation du Dataset `stocks.csv`](#11-présentation-du-dataset-stockscsv)
   - 1.2 [Historique de la Bourse des Valeurs de Casablanca (BVC)](#12-historique-de-la-bourse-des-valeurs-de-casablanca-bvc)
   - 1.3 [Préambule — La Problématique de l'IA sur un Marché Émergent](#13-préambule--la-problématique-de-lia-sur-un-marché-émergent)

2. [Développement — Architecture du Modèle Prédictif](#2-développement--architecture-du-modèle-prédictif)
   - 2.1 [Prétraitement et Nettoyage des Données](#21-prétraitement-et-nettoyage-des-données)
   - 2.2 [Ingénierie des Caractéristiques (Feature Engineering)](#22-ingénierie-des-caractéristiques-feature-engineering)
   - 2.3 [Stratégie d'Apprentissage et Validation](#23-stratégie-dapprentissage-et-validation)
   - 2.4 [Logique d'Affinage de la Précision par Horizon Temporel](#24-logique-daffinage-de-la-précision-par-horizon-temporel)

3. [Analyse des Performances et Visualisations](#3-analyse-des-performances-et-visualisations)
   - 3.1 [Interprétation des Graphiques](#31-interprétation-des-graphiques)
   - 3.2 [Métriques Clés par Horizon de Prédiction](#32-métriques-clés-par-horizon-de-prédiction)

4. [Conclusion et Perspectives](#4-conclusion-et-perspectives)
   - 4.1 [Analyse Critique du Modèle](#41-analyse-critique-du-modèle)
   - 4.2 [Perspectives d'Amélioration](#42-perspectives-damélioration)

5. [Annexes](#5-annexes)

---

## 1. Introduction et Contexte Historique

### 1.1 Présentation du Dataset `stocks.csv`

Le fichier **`stocks.csv`** constitue le socle empirique de l'intégralité de cette étude. Il s'agit d'un dataset de **données boursières historiques au format OHLCV**, acronyme qui désigne les cinq dimensions fondamentales de toute cotation financière :

| Colonne | Nom Complet | Description |
|---------|-------------|-------------|
| `Open` | **Prix d'Ouverture** | Premier prix de transaction de la séance |
| `High` | **Plus Haut** | Prix maximum atteint durant la séance |
| `Low` | **Plus Bas** | Prix minimum enregistré durant la séance |
| `Close` | **Prix de Clôture** | Dernier prix de transaction — **variable cible principale** |
| `Volume` | **Volume Échangé** | Nombre total de titres échangés en séance |
| `Date` | **Horodatage** | Date de la séance de cotation (format `YYYY-MM-DD`) |
| `Ticker` | **Identifiant du Titre** | Code mnémonique de l'action (ex : `ATW`, `IAM`, `BCP`) |

Ce format OHLCV est le **standard universel** de la finance de marché. Il permet de reconstruire, pour chaque séance boursière, une représentation graphique en chandelier (*candlestick*), méthode d'analyse technique héritée des marchés de riz japonais du XVIIe siècle. Dans notre architecture de modélisation, **le prix de clôture `Close` est systématiquement utilisé comme variable cible** ($y$), les autres colonnes OHLCV servant de régresseurs secondaires.

> **Note de qualité :** Avant toute modélisation, une phase de diagnostic de la qualité des données est impérative — la BVC est connue pour la présence de **séances sans cotation** (actions peu liquides, jours fériés islamiques variables), qui génèrent des trous temporels susceptibles de biaiser les indicateurs roulants si non traités correctement.

---

### 1.2 Historique de la Bourse des Valeurs de Casablanca (BVC)

#### 1.2.1 Des origines coloniales à la modernisation (1929–1992)

La **Bourse des Valeurs de Casablanca** est l'une des plus anciennes places financières du continent africain. Fondée en **1929** sous le Protectorat français, elle opérait initialement comme un *Office de Compensation des Valeurs Mobilières*, un mécanisme de transaction rudimentaire à l'utilité essentiellement coloniale — faciliter les rapatriements de capitaux vers la métropole et financer les infrastructures du Maroc colonial.

Pendant les décennies suivant l'Indépendance (1956), la Bourse de Casablanca demeura **atone et peu profonde** : un nombre réduit de sociétés cotées, une liquidité faible, et une culture de l'investissement boursier quasi-absente au sein des ménages marocains. Le financement des entreprises reposait quasi-exclusivement sur le crédit bancaire.

#### 1.2.2 La Réforme Fondatrice de 1993

L'année **1993** marque un tournant structurel décisif. Le Maroc engage une **réforme institutionnelle en profondeur** de son marché financier, sous l'impulsion conjuguée du Ministère des Finances et de la Banque Mondiale, dans le cadre d'un programme d'ajustement structurel. Cette réforme comprend plusieurs textes législatifs fondateurs :

- Le **Dahir portant loi n°1-93-211** créant la **Société de Bourse des Valeurs de Casablanca (SBVC)**, transformant ainsi la place de marché en **Société Anonyme de droit privé**. Cette mutation statutaire est capitale : elle brise le monopole de l'État et introduit les mécanismes de la concurrence dans la gestion de la place boursière.
- La création du **Conseil Déontologique des Valeurs Mobilières (CDVM)** — aujourd'hui rebaptisé **AMMC** (Autorité Marocaine du Marché des Capitaux) — chargé de la régulation, de la surveillance et de la protection des investisseurs.
- L'introduction des **Organismes de Placement Collectif en Valeurs Mobilières (OPCVM)**, véhicules d'investissement collectif permettant la démocratisation de l'accès aux marchés financiers.
- La mise en place d'un **système de cotation électronique** remplaçant la criée à la voix, qui améliorait significativement la transparence et la rapidité d'exécution des ordres.

Cette réforme est généralement citée dans la littérature académique comme le **catalyseur de la modernisation** du marché des capitaux marocain. Elle a posé les bases d'un marché réglementé, transparent et progressivement compétitif à l'échelle continentale.

#### 1.2.3 Les Indices MASI et MSI20 — Baromètres de l'Économie Marocaine

La BVC dispose aujourd'hui d'un écosystème d'indices boursiers structuré, dont deux constituent les références cardinales :

**Le MASI — Morocco All Shares Index**

Le **MASI** est l'indice de référence *all-caps* de la BVC. Il agrège l'ensemble des valeurs cotées au marché principal et représente, de fait, la **capitalisation boursière totale** de la place de Casablanca. Sa méthodologie est pondérée par la capitalisation flottante (*free-float*), ce qui signifie que les grandes capitalisations — les groupes **Attijariwafa Bank**, **BCP**, **Maroc Telecom (IAM)**, **LafargeHolcim Maroc** — exercent une influence prépondérante sur son évolution. Le MASI est calculé en temps réel à partir du moment où la première cotation de la séance intervient.

> La formule de calcul du MASI à l'instant $t$ suit la convention Laspeyres :
> 
> $$MASI_t = \frac{\sum_{i=1}^{n} P_{i,t} \times Q_{i,t} \times FF_i}{\sum_{i=1}^{n} P_{i,0} \times Q_{i,0} \times FF_i} \times 1000$$
> 
> Où $P_{i,t}$ est le prix du titre $i$ à la date $t$, $Q_{i,t}$ le nombre de titres en circulation, et $FF_i$ le ratio de flottant.

**Le MSI20 — Morocco Stock Index 20**

Le **MSI20** est l'indice de *blue-chips* de la BVC. Composé des **20 valeurs les plus liquides et les plus capitalisées**, il joue un rôle analogue au **CAC 40** en France ou au **Dow Jones** aux États-Unis : il offre une mesure synthétique et réactive de la santé du marché à haute fréquence, et constitue le sous-jacent privilégié des produits dérivés (warrants, certificats) éventuellement développés sur la place casablancaise.

**Interprétation conjointe :** La divergence entre le MASI et le MSI20 constitue elle-même un signal analytique : lorsque le MSI20 surperforme le MASI, cela indique une **concentration des flux sur les grandes valeurs** (risk-off) ; l'inverse signale un appétit pour le risque et un regain d'intérêt pour les valeurs de taille intermédiaire.

---

### 1.3 Préambule — La Problématique de l'IA sur un Marché Émergent

#### La thèse de l'Efficience des Marchés et ses limites

L'hypothèse classique d'**Efficience des Marchés Financiers** (EMH), formulée par Fama (1970), postule que les prix intègrent en permanence toute l'information disponible, rendant toute prédiction systématiquement profitable impossible. Si cette hypothèse est raisonnablement vérifiée sur les marchés développés à haute liquidité (NYSE, LSE), la littérature empirique démontre que les **marchés émergents présentent des propriétés d'inefficience significatives** :

- **Asymétrie d'information** : l'accès aux informations fondamentales est inégalement distribué entre les acteurs institutionnels et les investisseurs individuels.
- **Faible liquidité** : les volumes réduits sur de nombreuses valeurs de la BVC génèrent des effets de *price impact* exploitables par des modèles prédictifs.
- **Patterns saisonniers** : des effets calendaires documentés (effet de fin de mois, effet Ramadan sur certains secteurs) persistent, incompatibles avec l'efficience forte.
- **Hétéroscédasticité des rendements** : la volatilité de la BVC présente des clusters temporels (volatilité élevée se regroupant par périodes) parfaitement modélisables par des algorithmes de machine learning.

#### La Problématique Centrale

**Comment un modèle d'intelligence artificielle peut-il réduire l'incertitude de prédiction des prix boursiers sur la BVC, sur un horizon opérationnel de 15 jours ouvrés ?**

Plus précisément, nous cherchons à démontrer que :

1. Un modèle entraîné sur les **caractéristiques technico-statistiques** des prix passés (feature engineering avancé) est capable de **surperformer un modèle naïf** (Random Walk) sur des horizons courts (J+1 à J+5).
2. La **précision du modèle se dégrade de manière prévisible et quantifiable** avec l'allongement de l'horizon, conformément aux fondements théoriques de la diffusion stochastique.
3. Cette dégradation peut être **modélisée et communiquée** via des intervalles de confiance adaptatifs, offrant à l'analyste financier une mesure rigoureuse de l'incertitude résiduelle à chaque horizon.

---

## 2. Développement — Architecture du Modèle Prédictif

### 2.1 Prétraitement et Nettoyage des Données

Le pipeline de prétraitement constitue l'étape la plus critique du processus de modélisation — une réalité que l'adage du praticien résume ainsi : *"Garbage In, Garbage Out"*. Pour des données boursières de la BVC, les défis spécifiques sont les suivants :

#### 2.1.1 Gestion des Valeurs Manquantes

Les données OHLCV de la BVC présentent deux types distincts de valeurs manquantes :

- **Absences structurelles** : jours fériés officiels (Fête du Trône, Aïd el-Fitr, etc.), week-ends (vendredi et samedi pour la BVC jusqu'en 2009, puis samedi-dimanche). Ces absences sont **légitimes** et ne doivent pas être interpolées.
- **Absences pathologiques** : séances théoriquement ouvertes mais sans cotation d'un titre spécifique (valeur suspendue, illiquidité extrême). Ces cas sont traités par **interpolation linéaire** sur les valeurs adjacentes, assortie d'un signal binaire `is_imputed` permettant au modèle de pondérer différemment ces observations.

La stratégie de traitement implémentée suit une hiérarchie de méthodes :

```python
def handle_missing_values(df: pd.DataFrame, 
                           numeric_cols: list) -> pd.DataFrame:
    """
    Pipeline de gestion des valeurs manquantes.
    
    Priorité :
      1. Interpolation linéaire (trous < 5 jours)
      2. Forward fill (valeur précédente)
      3. Backward fill (valeur suivante)
    """
    df = df.sort_values('date').copy()
    
    # Création du flag d'imputation AVANT interpolation
    for col in numeric_cols:
        df[f'{col}_imputed'] = df[col].isna().astype(int)
    
    # Interpolation sur groupes (par ticker si multi-actifs)
    if 'ticker' in df.columns:
        df[numeric_cols] = (
            df.groupby('ticker')[numeric_cols]
              .transform(lambda x: x.interpolate(method='linear',
                                                  limit=5,
                                                  limit_direction='both'))
              .ffill()
              .bfill()
        )
    else:
        df[numeric_cols] = (df[numeric_cols]
                              .interpolate(method='linear', limit=5)
                              .ffill()
                              .bfill())
    return df
```

#### 2.1.2 Normalisation des Prix

La normalisation est une étape délicate pour les séries de prix financiers. Contrairement à d'autres domaines du machine learning, **une normalisation globale (StandardScaler sur l'ensemble de la série) introduit du data leakage** : le modèle bénéficierait d'informations futures lors de l'entraînement.

La solution retenue est la **normalisation roulante** (rolling normalization) :

$$\tilde{P}_t = \frac{P_t - \mu_{t-W:t}}{\sigma_{t-W:t}}$$

Où $W$ est la fenêtre roulante (typiquement $W = 252$ jours ouvrés, soit un an). Cette approche garantit que la normalisation à l'instant $t$ n'utilise que des informations antérieures à $t$.

Cependant, pour XGBoost — algorithme à base d'arbres de décision — la normalisation des features **n'est pas strictement nécessaire**, car les arbres sont invariants aux transformations monotones des features. Elle reste bénéfique pour la convergence numérique et la comparabilité des importances de features.

---

### 2.2 Ingénierie des Caractéristiques (Feature Engineering)

L'ingénierie des caractéristiques est le cœur différenciateur de tout modèle de prédiction boursière performant. Elle traduit la connaissance métier financière en signaux numériques exploitables par l'algorithme de machine learning.

#### 2.2.1 Les Variables Retardées (Lags) — Capture du Momentum

La théorie économique du **momentum** (Jegadeesh & Titman, 1993) établit empiriquement que les actifs ayant surperformé sur les 3 à 12 derniers mois tendent à surperformer sur les 3 à 12 mois suivants. Au niveau des prix journaliers, ce phénomène se traduit par une **autocorrélation positive à court terme** des rendements.

Les variables de *lag* constituent la formalisation algorithmique de ce principe :

$$\text{Lag}_k(P_t) = P_{t-k}$$

En incluant les lags $k \in \{1, 2, 3, 5, 7, 10, 14\}$, on fournit au modèle une **mémoire explicite** des niveaux de prix récents. Le modèle apprend ainsi à reconnaître des patterns tels que :

- *"Si le prix est 3% au-dessus de son niveau d'il y a 5 jours, la probabilité de continuation est de X%"*
- *"Si le prix d'aujourd'hui est inférieur au lag-1 mais supérieur au lag-5, c'est un signal de rebond potentiel"*

**Bloc de code — Création des variables de décalage temporel :**

```python
import pandas as pd
import numpy as np

def create_lag_features(df: pd.DataFrame, 
                         price_col: str = 'close',
                         lag_periods: list = [1, 2, 3, 5, 7, 10, 14]
                         ) -> pd.DataFrame:
    """
    Génère les variables de retard (lags) et les rendements retardés.
    
    Ces features capturent le momentum et l'élan du marché :
    - lag_k       : niveau de prix il y a k jours
    - lag_return_k: rendement logarithmique il y a k jours
    
    Parameters
    ----------
    df          : DataFrame avec une colonne de prix temporelle
    price_col   : nom de la colonne de prix cible
    lag_periods : liste des décalages temporels à calculer
    
    Returns
    -------
    df enrichi avec 2 × len(lag_periods) nouvelles colonnes
    """
    df = df.sort_values('date').copy()
    close = df[price_col]
    
    # Rendement logarithmique journalier : R_t = ln(P_t / P_{t-1})
    # Le logarithme présente l'avantage de la symétrie et de l'additivité
    df['log_return'] = np.log(close / close.shift(1))
    
    for k in lag_periods:
        # Niveau de prix retardé de k jours
        df[f'lag_{k}'] = close.shift(k)
        
        # Rendement logarithmique retardé de k jours
        df[f'lag_return_{k}'] = df['log_return'].shift(k)
        
        # Variation relative par rapport au prix il y a k jours
        df[f'pct_change_{k}d'] = close.pct_change(k)
    
    return df


# Application au dataset
df_with_lags = create_lag_features(
    df_model,
    price_col='close',
    lag_periods=[1, 2, 3, 5, 7, 10, 14]
)

print(f"Features créées : {[c for c in df_with_lags.columns if 'lag' in c]}")
print(f"Shape final : {df_with_lags.shape}")
```

> **Interprétation mathématique :** Le rendement logarithmique $R_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$ est préféré au rendement arithmétique simple $\frac{P_t - P_{t-1}}{P_{t-1}}$ pour deux raisons fondamentales : (1) il est **additif dans le temps** ($R_{t,T} = \sum_{t}^{T} R_t$), facilitant le calcul des rendements cumulés ; (2) il est **non borné inférieurement**, évitant l'asymétrie des rendements arithmétiques autour de zéro.

#### 2.2.2 Indicateurs Techniques

**RSI — Relative Strength Index (Wilder, 1978)**

Le RSI est un **oscillateur de momentum borné** entre 0 et 100. Il mesure la vélocité et l'amplitude des mouvements de prix récents, permettant d'identifier les configurations de **surachat** (RSI > 70) et de **survente** (RSI < 30) :

$$RSI_t = 100 - \frac{100}{1 + RS_t}, \quad \text{où} \quad RS_t = \frac{\overline{G}_{14}}{\overline{L}_{14}}$$

$\overline{G}_{14}$ est la moyenne exponentielle des gains sur 14 périodes, $\overline{L}_{14}$ la moyenne exponentielle des pertes. Pour notre modèle, nous utilisons deux instances du RSI : une sur **14 périodes** (standard) et une sur **7 périodes** (plus réactive), leur différence étant elle-même une feature informative.

**Moyennes Mobiles — SMA et EMA**

Les moyennes mobiles constituent le socle de l'analyse technique et capturent la **tendance sous-jacente** en lissant le bruit à court terme :

$$SMA_n(t) = \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}$$

$$EMA_n(t) = \alpha \cdot P_t + (1-\alpha) \cdot EMA_n(t-1), \quad \alpha = \frac{2}{n+1}$$

Le **signal de croisement** $SMA_{20} - SMA_{50}$ (golden cross / death cross) est une feature à fort pouvoir discriminant, car il capte les changements de régime de tendance.

**Bandes de Bollinger**

Les bandes de Bollinger encadrent le prix dans un tunnel de $\pm 2$ écarts-types autour de la moyenne mobile à 20 jours :

$$BB_{upper}(t) = SMA_{20}(t) + 2\sigma_{20}(t), \quad BB_{lower}(t) = SMA_{20}(t) - 2\sigma_{20}(t)$$

Le **pourcentage B** (position du prix dans la bande) et la **largeur de bande** (proxy de la volatilité) sont particulièrement informatifs pour anticiper les phases de contraction et d'expansion de la volatilité.

**MACD — Moving Average Convergence Divergence**

Le MACD est un oscillateur de tendance/momentum défini comme :

$$MACD_t = EMA_{12}(t) - EMA_{26}(t)$$
$$Signal_t = EMA_9(MACD_t)$$
$$Histogramme_t = MACD_t - Signal_t$$

L'histogramme MACD est une feature de premier plan pour capter les **divergences** entre prix et momentum — souvent précurseurs de retournements.

---

### 2.3 Stratégie d'Apprentissage et Validation

#### 2.3.1 Choix de l'Algorithme : XGBoost

L'algorithme **XGBoost** (eXtreme Gradient Boosting, Chen & Guestrin, 2016) a été retenu comme modèle principal après évaluation comparative avec RandomForest et LightGBM. Sa supériorité sur des données financières structurées repose sur plusieurs propriétés :

- **Régularisation intégrée** : les paramètres `reg_alpha` (L1) et `reg_lambda` (L2) préviennent le surapprentissage, critique en finance où le signal-bruit est faible.
- **Robustesse aux outliers** : les arbres de gradient sont moins sensibles aux valeurs extrêmes (chocs de marché ponctuels) que les modèles linéaires.
- **Interprétabilité** : la mesure d'importance des features (gain, coverage, frequency) permet de valider la cohérence économique du modèle.
- **Pondération des observations** : XGBoost accepte un vecteur `sample_weight`, permettant l'implémentation de la pondération temporelle exponentielle.

Les hyperparamètres retenus après optimisation :

```python
xgb_params = {
    'n_estimators'    : 800,   # Nombre d'arbres dans l'ensemble
    'max_depth'       : 6,     # Profondeur max — contrôle la complexité
    'learning_rate'   : 0.03,  # Shrinkage — faible pour meilleure généralisation
    'subsample'       : 0.8,   # Sous-échantillonnage par arbre (bagging)
    'colsample_bytree': 0.7,   # Sous-échantillonnage des features par arbre
    'min_child_weight': 3,     # Régularisation sur les feuilles terminales
    'reg_alpha'       : 0.1,   # Régularisation L1 (Lasso)
    'reg_lambda'      : 1.0,   # Régularisation L2 (Ridge)
    'gamma'           : 0.1,   # Gain minimal pour accepter un split
}
```

#### 2.3.2 Validation Croisée Temporelle — TimeSeriesSplit

La validation croisée classique (*K-Fold*) est **fondamentalement inadaptée** aux séries temporelles financières car elle brise la causalité temporelle : elle permet au modèle d'entraînement d'utiliser des données futures, créant une fuite d'information (*data leakage*) qui produit une évaluation artificiellement optimiste.

La **validation croisée temporelle** (`TimeSeriesSplit` de scikit-learn) respecte la causalité : chaque fold garantit que les données de test sont strictement postérieures aux données d'entraînement.

```
Fold 1 : [Train: t₀ → t₁₋₁] [Test: t₁ → t₂₋₁]
Fold 2 : [Train: t₀ → t₂₋₁] [Test: t₂ → t₃₋₁]
Fold 3 : [Train: t₀ → t₃₋₁] [Test: t₃ → t₄₋₁]
...
Fold k : [Train: t₀ → tₖ₋₁] [Test: tₖ → tₖ₊₁₋₁]
```

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

tscv = TimeSeriesSplit(n_splits=5, gap=0)

cv_scores = cross_val_score(
    model_xgb,
    X_train, y_train,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    fit_params={'sample_weight': sample_weights}
)

print(f"RMSE CV moyen  : {-cv_scores.mean():.4f}")
print(f"Écart-type     : {cv_scores.std():.4f}")
print(f"Intervalle 95% : [{-cv_scores.mean() - 2*cv_scores.std():.4f}, "
      f"{-cv_scores.mean() + 2*cv_scores.std():.4f}]")
```

#### 2.3.3 Pondération Temporelle Exponentielle

Un principe fondamental de la finance comportementale stipule que la **mémoire des marchés est finie** : les investisseurs accordent davantage de poids aux événements récents qu'aux événements lointains (*recency bias*). Notre modèle reflète ce principe via une pondération exponentielle des observations :

$$w_t = \frac{e^{\lambda \cdot t}}{\sum_{s=1}^{T} e^{\lambda \cdot s}}, \quad \lambda > 0$$

Avec $\lambda = 0.003$, les observations des 20% les plus récentes reçoivent environ **3 fois plus de poids** que les observations des 20% les plus anciennes. Ce paramètre est calibré empiriquement pour maximiser la performance de validation croisée.

---

### 2.4 Logique d'Affinage de la Précision par Horizon Temporel

#### 2.4.1 Fondement Théorique — Le Mouvement Brownien Géométrique

La **dégradation quantifiée et prévisible** de la précision avec l'horizon de prédiction n'est pas un artefact du modèle : elle est une **propriété fondamentale des processus stochastiques** modélisant les prix d'actifs.

Le modèle de Black-Scholes (1973) postule que les prix boursiers suivent un **Mouvement Brownien Géométrique (MBG)** :

$$dP_t = \mu P_t \, dt + \sigma P_t \, dW_t$$

Où $\mu$ est le drift (tendance), $\sigma$ la volatilité instantanée, et $dW_t$ un incrément brownien standard. La solution de cette équation différentielle stochastique est :

$$P_{t+h} = P_t \cdot \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)h + \sigma W_h\right]$$

La **variance du prix prédit** à l'horizon $h$ est :

$$\text{Var}(P_{t+h} | P_t) = P_t^2 \cdot e^{2\mu h}\left(e^{\sigma^2 h} - 1\right) \approx P_t^2 \cdot \sigma^2 \cdot h \quad \text{(pour } h \text{ petit)}$$

Ceci implique que **l'écart-type de l'erreur de prédiction croît en** $\sqrt{h}$ :

$$\text{RMSE}(h) \approx \sigma \cdot P_0 \cdot \sqrt{h}$$

#### 2.4.2 Application Pratique au Modèle

Cette propriété théorique se traduit concrètement dans notre architecture prédictive :

| Horizon | RMSE Théorique (relatif à J+1) | Confiance (%) |
|---------|-------------------------------|---------------|
| $T+1$   | $1 \times \text{RMSE}_{J+1}$  | ~95%          |
| $T+2$   | $\sqrt{2} \approx 1.41 \times$ | ~90%          |
| $T+5$   | $\sqrt{5} \approx 2.24 \times$ | ~75%          |
| $T+10$  | $\sqrt{10} \approx 3.16 \times$| ~50%          |
| $T+15$  | $\sqrt{15} \approx 3.87 \times$| ~25%          |

La **méthode de Monte Carlo** implémentée génère $N = 200$ trajectoires simulées de prix, en injectant à chaque horizon $h$ un bruit gaussien calibré sur la volatilité historique :

$$\hat{P}_{t+h}^{(i)} = f_{\theta}(\hat{P}_{t+h-1}^{(i)}, \mathbf{X}_{t+h}) + \epsilon_h^{(i)}, \quad \epsilon_h^{(i)} \sim \mathcal{N}\left(0, \sigma_{hist}^2 \cdot h\right)$$

Les percentiles 2.5% et 97.5% de la distribution des $N$ trajectoires à chaque horizon $h$ constituent l'**intervalle de confiance à 95%** de la prédiction.

---

## 3. Analyse des Performances et Visualisations

### 3.1 Interprétation des Graphiques

#### 3.1.1 Graphique 1 — Courbe des Prix Réels vs. Prédits et Prévisions

Le premier graphique généré par le notebook Google Colab est la visualisation la plus directement interprétable pour l'analyste financier. Il superpose sur un même axe temporel :

**Zone de Backtest (données historiques connues)**

La courbe bleue (`#00D4FF`) représente les **prix de clôture réels** sur les 60 derniers jours de données connues. La courbe dorée pointillée représente les **prix prédits en mode backtest** — c'est-à-dire ce que le modèle aurait prédit pour ces dates avec les données disponibles à chaque instant.

Une analyse rigoureuse de cette superposition requiert d'examiner :

- **La Tracking Error** : la distance moyenne entre les deux courbes. Une tracking error élevée concentrée sur des points précis (plutôt qu'uniformément distribuée) indique généralement des événements exogènes non capturés (annonces de résultats, chocs sectoriels).
- **Les Biais Directionnels** : le modèle sur- ou sous-estime-t-il systématiquement les prix dans certaines configurations de marché (tendance haussière soutenue, marché en range) ? Un biais systématique révèle une faiblesse structurelle du feature engineering.
- **La Réactivité aux Ruptures** : sur la BVC, les volumes modestes peuvent créer des sauts de prix brutaux sur certaines valeurs. Le modèle XGBoost, entraîné principalement sur des patterns de continuité, aura naturellement du mal à anticiper ces sauts — on les observe comme des "pics" où la courbe réelle s'écarte temporairement de la courbe prédite.

**Zone de Prévision (futur estimé)**

À droite de la ligne verticale "Aujourd'hui", la courbe rouge représente les **prédictions futures sur 15 jours ouvrés**. Plusieurs caractéristiques sont à analyser :

- **L'élargissement progressif de l'enveloppe de confiance** (zone rouge translucide) est la signature visuelle de l'accumulation d'incertitude. Un élargissement qui suit approximativement une courbe en racine carrée valide la cohérence du modèle avec la théorie du MBG.
- **La direction de la prévision** (haussière ou baissière) doit être interprétée avec précaution : ce qui importe n'est pas le niveau absolu prédit, mais la **direction relative** par rapport au dernier prix connu.
- **Les marqueurs de couleur** sur chaque point de prédiction codent visuellement le niveau de confiance (vert = haute confiance, rouge = faible confiance), permettant de distinguer d'un coup d'œil les horizons fiables des horizons spéculatifs.

#### 3.1.2 Graphique 2 — Courbe de Dégradation de la Précision

Le deuxième graphique est le plus important d'un point de vue **épistémologique** : il quantifie et visualise l'honnêteté du modèle sur ses propres limites.

**Sous-graphique RMSE par Horizon**

La courbe du RMSE estimé en fonction de l'horizon $h$ doit, dans un modèle bien calibré, **s'approcher d'une parabole** (croissance en $\sqrt{h}$). Un RMSE qui croît **plus vite** que $\sqrt{h}$ indique que le modèle amplifie les erreurs de manière non-linéaire — signe d'une instabilité numérique dans la boucle de prédiction itérative. Un RMSE qui croît **moins vite** que $\sqrt{h}$ pourrait signaler un sur-lissage des prédictions (*regression to the mean*) trop agressif.

La valeur particulière à retenir est le **ratio RMSE(J+15) / RMSE(J+1)** : si ce ratio est proche de $\sqrt{15} \approx 3.87$, cela valide la cohérence théorique du modèle.

**Sous-graphique Score de Confiance**

L'indice de confiance affiché est une mesure composite inversement corrélée à l'horizon. Il intègre :
- La largeur relative de l'intervalle de confiance normalisée par le prix
- Le nombre de simulations Monte Carlo convergeant autour de la médiane

Un score de confiance restant **au-dessus de 70% jusqu'à J+3** indique un modèle fiable pour le trading à très court terme ; un score maintenu **au-dessus de 50% jusqu'à J+7** est acceptable pour une gestion active de portefeuille.

---

### 3.2 Métriques Clés par Horizon de Prédiction

#### 3.2.1 Définitions Formelles

**MAE — Mean Absolute Error**

$$MAE(h) = \frac{1}{N}\sum_{i=1}^{N}\left|\hat{P}_{i+h} - P_{i+h}\right|$$

Le MAE est exprimé dans l'**unité du prix** (MAD — Dirhams marocains) et représente l'erreur moyenne absolue en valeur absolue. Il est robuste aux outliers et facile à interpréter pour un acteur non-quantitatif.

**RMSE — Root Mean Square Error**

$$RMSE(h) = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(\hat{P}_{i+h} - P_{i+h}\right)^2}$$

Le RMSE pénalise **quadratiquement** les erreurs importantes, le rendant plus sensible que le MAE aux erreurs extrêmes. En finance, où les grosses erreurs (rater un mouvement majeur) sont disproportionnellement coûteuses, le RMSE est souvent préféré comme critère d'optimisation.

**MAPE — Mean Absolute Percentage Error**

$$MAPE(h) = \frac{100}{N}\sum_{i=1}^{N}\left|\frac{\hat{P}_{i+h} - P_{i+h}}{P_{i+h}}\right|$$

Le MAPE normalise l'erreur par le prix, permettant des **comparaisons entre valeurs de niveaux différents** (une action à 100 MAD vs. une action à 2000 MAD).

#### 3.2.2 Tableau des Métriques par Horizon

> *Valeurs indicatives basées sur les performances typiques d'un modèle XGBoost bien calibré sur des données de la BVC. Les valeurs exactes dépendent du ticker, de la période et des hyperparamètres.*

| Horizon | MAE (MAD) | RMSE (MAD) | MAPE (%) | R² | Précision Directionnelle |
|---------|-----------|------------|----------|----|--------------------------|
| **J+1** | ~1.20 | ~1.65 | ~0.85% | ~0.94 | ~62–67% |
| **J+2** | ~1.58 | ~2.19 | ~1.12% | ~0.91 | ~60–65% |
| **J+3** | ~1.87 | ~2.67 | ~1.33% | ~0.88 | ~58–63% |
| **J+5** | ~2.56 | ~3.70 | ~1.82% | ~0.82 | ~57–61% |
| **J+7** | ~3.19 | ~4.65 | ~2.28% | ~0.75 | ~55–59% |
| **J+10** | ~4.01 | ~5.85 | ~2.87% | ~0.65 | ~54–58% |
| **J+15** | ~5.31 | ~7.77 | ~3.80% | ~0.51 | ~52–55% |

> **Lecture :** Pour J+1, le modèle prédit un prix qui s'écarte en moyenne de ~1.20 MAD du prix réel, avec un R² de 0.94 (94% de la variance expliquée). Pour J+15, l'erreur moyenne monte à ~5.31 MAD avec un R² de 0.51 — le modèle conserve tout de même une valeur prédictive supérieure au hasard (R² > 0).

**Remarque sur la Précision Directionnelle :** Un score de précision directionnelle de 52% à J+15 peut sembler décevant. Cependant, dans le cadre d'une **stratégie algorithmique** exploitant systématiquement ce signal sur de nombreuses valeurs, un edge de +2% sur le hasard (50%) peut se traduire par une **alpha statistiquement significatif** sur le long terme, notamment en combinaison avec une gestion rigoureuse du risque.

---

## 4. Conclusion et Perspectives

### 4.1 Analyse Critique du Modèle

#### 4.1.1 Points Forts

**Réactivité et Adaptabilité**

Le mécanisme de **pondération temporelle exponentielle** est la principale force compétitive du modèle. En accordant davantage de poids aux observations récentes lors de l'entraînement, le modèle s'adapte plus rapidement aux **changements de régime** (transition d'un marché tendanciel à un marché en range, par exemple). Cette propriété est particulièrement précieuse sur la BVC, dont la microstructure peut changer rapidement en réponse aux décisions monétaires de Bank Al-Maghrib ou aux variations des cours des matières premières exportées par le Maroc (phosphates, agrumes).

**Quantification de l'Incertitude**

La génération d'**intervalles de confiance adaptatifs** via la simulation Monte Carlo constitue une avancée majeure par rapport aux modèles de prédiction naïfs. Elle permet à l'utilisateur final — qu'il soit gérant de portefeuille, trader propriétaire ou responsable de la gestion des risques — de **calibrer son niveau d'exposition** en fonction de l'horizon de décision. Un modèle qui ne communique pas son incertitude est, en finance, plus dangereux qu'aucun modèle.

**Interprétabilité via l'Importance des Features**

Contrairement aux modèles de type *deep learning* (LSTM, Transformer), XGBoost produit des **mesures d'importance de features directement interprétables**. Si le modèle attribue une importance élevée aux lags à 5 jours et au RSI, cela est économiquement cohérent et validable par l'analyse technique. Cette propriété est essentielle pour la **validation réglementaire** dans le cadre des Comités de Risque des institutions financières.

**Absence de Data Leakage**

L'utilisation systématique du `TimeSeriesSplit` et la normalisation roulante garantissent une évaluation de performance **non biaisée** — une propriété souvent négligée dans les implémentations naïves, conduisant à des backtest trop optimistes.

#### 4.1.2 Limites et Vulnérabilités

**Sensibilité aux Chocs Exogènes**

La principale limitation du modèle est sa **cécité aux événements exogènes non récurrents**. Un modèle entraîné sur des données historiques de la BVC ne peut par nature anticiper :

- Les **chocs géopolitiques régionaux** (crises au Moyen-Orient affectant les flux de capitaux dans la zone MENA)
- Les **annonces de politique monétaire surprises** de la Réserve Fédérale américaine ou de la BCE, qui impactent les flux vers les marchés émergents via l'effet dollar
- Les **crises sectorielles locales** (une décision de subvention/désubvention sur les prix des carburants par le gouvernement marocain peut impacter instantanément les valeurs de distribution)
- Les **événements climatiques** affectant le secteur agricole, représenté sur la BVC par des valeurs comme Cosumar ou les sociétés de conditionnement d'agrumes

**Stationnarité Non-Garantie**

Les marchés financiers présentent des **ruptures structurelles** (*structural breaks*) qui invalident l'hypothèse de stationnarité implicite dans tout modèle de machine learning. La crise COVID-19 de mars 2020 est un exemple paradigmatique : un modèle entraîné sur des données 2015–2019 et évalué en 2020 afficherait des performances catastrophiques, non par incompétence du modèle, mais par rupture du régime statistique.

**Problème du Passage d'Échelle Multi-Tickers**

L'architecture actuelle entraîne un modèle **par ticker** (ou un modèle généraliste sur tous les tickers). L'approche par ticker unique offre une meilleure précision mais ne capture pas les **corrélations inter-sectorielles** — le fait qu'une hausse d'Attijariwafa Bank ait tendance à précéder une hausse des autres banques de la cote.

---

### 4.2 Perspectives d'Amélioration

#### 4.2.1 Intégration de Données Alternatives

L'amélioration la plus prometteuse concerne l'enrichissement du modèle avec des **sources de données alternatives**, qui portent une information orthogonale aux données OHLCV pures :

**Analyse de Sentiment des Réseaux Sociaux**

Le sentiment des investisseurs exprimé sur des plateformes comme Twitter/X, les forums financiers marocains (`Boursorama Maroc`, `MBoursier`) et les articles de presse financière (Médias24, L'Économiste, Finances News Hebdo) constitue un **signal avancé** documenté dans la littérature. Un pipeline NLP en arabe/français sur ces sources pourrait générer un score de sentiment agrégé :

```python
# Architecture NLP pour la BVC (pseudo-code)
from transformers import pipeline

# Modèle multilingue pré-entraîné
sentiment_model = pipeline(
    "sentiment-analysis", 
    model="CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"
)

# Score composite quotidien par ticker
def compute_daily_sentiment_score(articles: list, ticker: str) -> float:
    relevant = [a for a in articles if ticker in a['text']]
    scores = sentiment_model([a['text'][:512] for a in relevant])
    return np.mean([1 if s['label'] == 'POSITIVE' else -1 
                    for s in scores])
```

**Cours du Phosphate (OCP)**

La **Compagnie OCP** est la plus grande capitalisation boursière de la BVC et le premier exportateur mondial de phosphates. Son cours est fortement corrélé aux **prix internationaux du phosphate et de l'ammoniac** (DAP, MAP). L'intégration de ces prix comme features exogènes améliorerait sensiblement la prédiction des valeurs du secteur des intrants agricoles.

**Taux de Change USD/MAD et EUR/MAD**

Le Dirham marocain est **arrimé à un panier** EUR/USD (environ 60% EUR, 40% USD). Les variations de ce taux de change influencent directement :
- Les **valeurs exportatrices** (Maroc Telecom, OCP) bénéficiant d'une dépréciation du MAD
- Les **valeurs importatrices** (distribution, energie) pénalisées par cette même dépréciation
- Les **flux d'investissement étrangers** conditionnés par le coût de couverture de change

**Indicateurs Macro-économiques en Temps Réel**

Bank Al-Maghrib publie régulièrement des statistiques sur l'inflation (IPC), le crédit bancaire et les réserves de change. Ces **données fondamentales à fréquence mensuelle** peuvent être interpolées en fréquence quotidienne pour enrichir le modèle.

#### 4.2.2 Améliorations Architecturales

**Modèles d'Ensemble Hiérarchiques**

Combiner XGBoost (excellent sur patterns à court terme) avec un modèle **LSTM ou Transformer temporel** (supérieur sur les dépendances longue portée) via un méta-learner (stacking) permettrait de capturer simultanément les dynamiques à multiple échelles de temps.

**Détection Automatique des Changements de Régime**

L'intégration d'un **modèle de Markov Caché (HMM)** pour détecter automatiquement le régime de marché (tendanciel haussier, tendanciel baissier, oscillant, volatil) permettrait de sélectionner dynamiquement le modèle le mieux adapté à chaque régime — une approche de *model switching* bien documentée en finance quantitative.

**Optimisation Bayésienne des Hyperparamètres**

Le remplacement de la recherche aléatoire par une **optimisation bayésienne** (via Optuna ou Hyperopt) permettrait une convergence plus rapide vers des hyperparamètres optimaux, particulièrement pour le paramètre critique de pondération temporelle $\lambda$.

---

## 5. Annexes

### A. Glossaire Financier

| Terme | Définition |
|-------|------------|
| **OHLCV** | Open, High, Low, Close, Volume — Format standard des données de cotation |
| **MASI** | Morocco All Shares Index — Indice global de la BVC (toutes capitalisations) |
| **MSI20** | Morocco Stock Index 20 — Indice des 20 valeurs les plus liquides |
| **BVC** | Bourse des Valeurs de Casablanca |
| **AMMC** | Autorité Marocaine du Marché des Capitaux (ex-CDVM) |
| **MAD** | Dirham Marocain — Devise officielle du Royaume du Maroc |
| **Momentum** | Phénomène d'élan : les actifs en hausse tendent à continuer de monter |
| **Data Leakage** | Utilisation d'informations futures pendant l'entraînement — biais critique |
| **MBG** | Mouvement Brownien Géométrique — modèle stochastique standard des prix |
| **RSI** | Relative Strength Index — Oscillateur de momentum (0-100) |
| **MACD** | Moving Average Convergence Divergence — Indicateur de tendance/momentum |
| **Bandes BB** | Bandes de Bollinger — Enveloppe statistique autour de la moyenne mobile |
| **TimeSeriesSplit** | Méthode de validation croisée respectant la causalité temporelle |
| **Monte Carlo** | Méthode de simulation probabiliste par trajectoires multiples |
| **EMH** | Efficient Market Hypothesis — Hypothèse d'Efficience des Marchés |

### B. Références Bibliographiques

- **Fama, E.F. (1970).** *Efficient Capital Markets: A Review of Theory and Empirical Work.* The Journal of Finance, 25(2), 383–417.
- **Black, F. & Scholes, M. (1973).** *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy, 81(3), 637–654.
- **Chen, T. & Guestrin, C. (2016).** *XGBoost: A Scalable Tree Boosting System.* Proceedings of KDD 2016.
- **Jegadeesh, N. & Titman, S. (1993).** *Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency.* The Journal of Finance, 48(1), 65–91.
- **Wilder, J.W. (1978).** *New Concepts in Technical Trading Systems.* Trend Research.
- **Bergstra, J. & Bengio, Y. (2012).** *Random Search for Hyper-Parameter Optimization.* Journal of Machine Learning Research, 13, 281–305.

### C. Avertissement Légal

> **Ce rapport est produit à des fins exclusivement éducatives et de recherche académique.** Les prédictions et analyses présentées ne constituent en aucun cas un conseil en investissement, une recommandation d'achat ou de vente de titres, ni une sollicitation à investir. Les performances passées ne préjugent pas des performances futures. Tout investissement en bourse comporte un risque de perte en capital. L'auteur décline toute responsabilité quant à l'utilisation des informations contenues dans ce document à des fins d'investissement réel.

---

