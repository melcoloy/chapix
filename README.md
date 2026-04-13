# 🎲 Générateur de Mosaïque en Dominos (Projet P4)

Bienvenue sur le dépôt de notre projet de génération de mosaïques en dominos. Cet outil interactif permet de transformer n'importe quelle image (ou photo prise en direct) en un plan de montage physiquement réalisable à l'aide de boîtes de dominos standard (Double-6 ou Double-9).

Développé par **Matteo Hanon Obsomer** et **Clément Leroy**.

---

## ✨ Fonctionnalités

- **Sources multiples :** importation de fichiers (JPG/PNG) ou prise de photo via la webcam.
- **Prétraitement avancé :** conversion en niveaux de gris, segmentation des contours, dithering de Floyd-Steinberg.
- **Trois algorithmes de résolution :**
  1. *Glouton (par le centre) :* rapide, 100% sans trou, priorité au centre de l'image.
  2. *Hongrois (Kuhn-Munkres) :* optimum mathématique exact, plus lent.
  3. *Recuit simulé :* méta-heuristique aléatoire, bon compromis qualité/temps.
- **Inspecteur visuel :** mise en évidence d'un type de domino par cadre rouge.
- **Exportation :** sauvegarde PNG (avec ou sans surbrillance) et impression directe.
- **Métriques :** score de fidélité et temps d'exécution.

---

## 📁 Structure du projet

```
kkbox/
├── app.py              # Interface Streamlit (UI uniquement)
├── core/
│   ├── __init__.py
│   ├── inventaire.py   # Génération du stock de dominos
│   ├── algorithmes.py  # Glouton, Hongrois, Recuit simulé
│   └── image.py        # Prétraitement, dessin, mise en évidence
├── .gitignore
└── README.md
```

---

## ⚙️ Installation et exécution

**Prérequis : Python 3.10+**

**1. Cloner le dépôt :**
```bash
git clone https://github.com/melcoloy/chapix.git
cd chapix
```

**2. Installer les dépendances :**
```bash
pip install streamlit Pillow numpy scipy streamlit-image-coordinates
```

**3. Lancer l'application :**
```bash
py -m streamlit run app.py
```

---

## 🧠 Détail des algorithmes

### Glouton (par le centre)
1. Pavage initial sans trou (horizontal si colonnes pair, vertical sinon).
2. Optimisation par swapping 2×2 pour mieux suivre les contours.
3. Assignation gloutonne du stock, du centre de l'image vers les bords.

### Hongrois (Kuhn-Munkres)
Construit une matrice de coûts N×N entre chaque emplacement et chaque domino disponible, puis résout le problème d'affectation optimale via `scipy.optimize.linear_sum_assignment`. Garantit le minimum d'erreur global mais limité à 5 000 emplacements (mémoire).

### Recuit simulé
Échange aléatoire de dominos entre emplacements, accepté selon le critère de Metropolis. Permet d'échapper aux minima locaux du glouton sans le coût mémoire du hongrois.