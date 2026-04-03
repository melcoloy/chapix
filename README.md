# 🎲 Générateur de Mosaïque en Dominos (Projet P4)

Bienvenue sur le dépôt de notre projet de génération de mosaïques en dominos. Cet outil interactif permet de transformer n'importe quelle image (ou photo prise en direct) en un plan de montage physiquement réalisable à l'aide de boîtes de dominos standard (Double-6 ou Double-9).

Développé par **Matteo Hanon Obsomer** et **Clément Leroy**.

## ✨ Fonctionnalités Principales
* **Sources Multiples :** Importation de fichiers (JPG/PNG) ou prise de photo en direct via la webcam.
* **Prétraitement Avancé :** Conversion en niveaux de gris, segmentation des contours et propagation d'erreur (Dithering de Floyd-Steinberg) pour conserver les dégradés.
* **Deux Algorithmes de Résolution :**
  1. *Heuristique Gloutonne :* Distribution par le centre de l'image (tri radial) combinée à une optimisation topologique locale (Swapping 2x2). Extrêmement rapide et 100% sans trou.
  2. *Méthode Hongroise :* Utilisation de l'algorithme de Kuhn-Munkres pour trouver l'affectation mathématique parfaite (optimum absolu) au détriment du temps de calcul.
* **Exportation du Plan :** Sauvegarde au format PNG avec nom personnalisé ou impression directe depuis le navigateur web.
* **Métriques en temps réel :** Affichage du temps d'exécution (chronomètre) et du score de fidélité.

## ⚙️ Installation et Exécution

Ce projet nécessite **Python 3.x** et les bibliothèques suivantes :
* `streamlit` (Interface web)
* `Pillow` (Traitement et dessin d'images)
* `numpy` (Calculs matriciels)
* `scipy` (Méthode Hongroise)

**1. Cloner le dépôt et installer les dépendances :**
Dans le terminal : 
pip install streamlit Pillow numpy scipy
**2. Lancer l'application locale :**
Dans le terminal :
py -m streamlit run app.py
