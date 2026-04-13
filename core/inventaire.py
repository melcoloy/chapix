"""
Génération et gestion du stock de dominos.
"""
import random
import numpy as np

_TYPES_JEU = {"double_six": 6, "double_neuf": 9}


def valeur_max(type_jeu: str) -> int:
    """Retourne la valeur maximale d'un pip pour le type de jeu donné."""
    if type_jeu not in _TYPES_JEU:
        raise ValueError(
            f"Type de jeu invalide : '{type_jeu}'. "
            f"Valeurs acceptées : {list(_TYPES_JEU.keys())}"
        )
    return _TYPES_JEU[type_jeu]


def boite_complete(type_jeu: str) -> list[tuple]:
    """Retourne la liste des 28 (ou 55) dominos d'une boîte standard."""
    vmax = valeur_max(type_jeu)
    return [(i, j) for i in range(vmax + 1) for j in range(i, vmax + 1)]


def generer_stock(type_jeu: str = "double_six", nb_boites: int = 1) -> list[tuple]:
    """
    Génère le stock complet de dominos pour nb_boites boîtes.

    Args:
        type_jeu: "double_six" ou "double_neuf".
        nb_boites: nombre de boîtes (entier >= 1).

    Returns:
        Liste de tuples (i, j).
    """
    if not isinstance(nb_boites, int) or nb_boites < 1:
        raise ValueError(f"nb_boites doit être un entier >= 1, reçu : {nb_boites!r}")
    return boite_complete(type_jeu) * nb_boites


def completer_inventaire(
    nb_dominos_necessaires: int,
    type_jeu: str = "double_six",
    matrice_cibles: np.ndarray | None = None,
) -> list[tuple]:
    """
    Génère exactement nb_dominos_necessaires dominos.
    Si le nombre n'est pas un multiple de la taille d'une boîte, complète
    intelligemment en privilégiant les pièces les plus utiles selon la matrice cible.

    Args:
        nb_dominos_necessaires: nombre exact de dominos requis.
        type_jeu: "double_six" ou "double_neuf".
        matrice_cibles: np.ndarray optionnel pour guider le choix des pièces de complétion.

    Returns:
        Liste de tuples (i, j).
    """
    jeu_de_base = boite_complete(type_jeu)
    taille_jeu = len(jeu_de_base)
    nb_jeux_complets = nb_dominos_necessaires // taille_jeu
    reste = nb_dominos_necessaires % taille_jeu

    inventaire = jeu_de_base * nb_jeux_complets

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