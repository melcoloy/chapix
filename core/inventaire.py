#Génération et gestion du stock de dominos.

import random
import numpy as np

_TYPES_JEU = {"double_six": 6, "double_neuf": 9}


#Retourne la valeur max d'un type de jeu
def valeur_max(type_jeu: str) -> int:
    if type_jeu not in _TYPES_JEU:
        raise ValueError(
            f"Type de jeu invalide : '{type_jeu}'. "
            f"Valeurs acceptées : {list(_TYPES_JEU.keys())}"
        )
    return _TYPES_JEU[type_jeu]

#Retourne la liste des 28 ou 55 dominos d'une boîte standard.
def boite_complete(type_jeu: str) -> list[tuple]:
    vmax = valeur_max(type_jeu)
    return [(i, j) for i in range(vmax + 1) for j in range(i, vmax + 1)]

#Génère le nombre de dominos nécessaire, en prenant des boîtes complètes puis en complétant par les dominos les plus adaptés
def completer_inventaire(nb_dominos_necessaires: int, type_jeu: str = "double_six", matrice_cibles: np.ndarray | None = None) -> list[tuple]:
    
    jeu_de_base = boite_complete(type_jeu)
    taille_jeu = len(jeu_de_base)
    nb_jeux_complets = nb_dominos_necessaires // taille_jeu
    reste = nb_dominos_necessaires % taille_jeu
    inventaire = jeu_de_base * nb_jeux_complets

    #Calcule les dominos les plus adaptés à l'image en regardant les fréquences des valeurs dans matrice_cibles
    if reste > 0:
        if matrice_cibles is not None:
            valeurs, counts = np.unique(matrice_cibles, return_counts=True)
            frequences = dict(zip(valeurs, counts))
            scores = [
                (frequences.get(d[0], 0) + frequences.get(d[1], 0), d)
                for d in jeu_de_base
            ]
            scores.sort(key=lambda x: x[0], reverse=True)
            inventaire += [d for _, d in scores[:reste]]
        else:
            inventaire += random.sample(jeu_de_base, reste)

    return inventaire