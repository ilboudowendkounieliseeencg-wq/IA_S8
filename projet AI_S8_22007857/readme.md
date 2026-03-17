Voici un descriptif synthétique de la structure type de votre base de données `stocks.csv`, telle qu'elle est généralement articulée pour la **Bourse de Casablanca** :

Lien vers le jeu de donnée:    https://drive.google.com/file/d/1nhK4jq_UTkaGZz79yNyoHR1S-hmvicBD/view?usp=drive_link

### 📊 Fiche d'Identité du Dataset

Le fichier est une **série temporelle (Time Series)** multivariée. Il regroupe l'historique des cotations des entreprises cotées à Casablanca. Voici sa composition probable :

* **Format :** CSV (Comma Separated Values).
* **Granularité :** Journalière (Daily).
* **Portée :** Données historiques des actifs (Actions) et potentiellement des indices (MASI/MSI20).

---

### 🔍 Colonnes Clés (Features)

Généralement, votre fichier s'articule autour des dimensions suivantes :

1.  **Temporelle (`Date`) :** La colonne pivot. Elle permet de séquencer les données pour l'apprentissage du modèle (format `AAAA-MM-JJ`).
2.  **Identifiant (`Ticker` ou `Nom`) :** Le code de l'entreprise (ex: *IAM* pour Itissalat Al-Maghrib, *ATW* pour Attijariwafa Bank). C'est ce qui permet au modèle de distinguer les performances d'une entité à l'autre.
3.  **Cours de Clôture (`Close`) :** La donnée la plus critique. C'est la variable cible ($Y$) que le modèle cherche à prédire.
4.  **Métriques de Séance (OHLC) :**
    * **Open :** Cours d'ouverture.
    * **High / Low :** Les plus hauts et plus bas atteints (utile pour calculer la volatilité).
5.  **Volume :** Le nombre de titres échangés. Un indicateur de force : une variation de prix avec un fort volume est plus "fiable" pour l'IA qu'une variation avec peu d'échanges.
6.  **Variation (%) :** Le rendement quotidien, souvent utilisé pour normaliser les données avant l'entraînement.

---

### 💡 Pourquoi cette base est exploitable par l'IA ?

* **Séquençage :** Le modèle va "apprendre" que si le prix baisse 3 jours de suite avec un volume croissant, il y a une forte probabilité de poursuite de la baisse à $J+1$.
* **Indicateurs Dérivés :** À partir de ces colonnes brutes, le code va générer des **indicateurs techniques** (RSI, Moyennes Mobiles) qui transformeront de simples chiffres en signaux de tendance.
