"""
Algorithmes de placement de dominos.

Tous les algorithmes retournent une liste de dicts :
    [{"case1": (i, j), "case2": (i, j), "valeurs": (v1, v2)}, ...]
"""
import math
import random
import numpy as np
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────
# Utilitaires partagés
# ─────────────────────────────────────────────────────────────────────

def _erreur_domino(domino: tuple, cible1: int, cible2: int) -> tuple[int, tuple]:
    """Retourne (erreur_min, domino_orienté_au_mieux)."""
    err_norm = abs(domino[0] - cible1) + abs(domino[1] - cible2)
    err_inv  = abs(domino[1] - cible1) + abs(domino[0] - cible2)
    return (err_norm, domino) if err_norm <= err_inv else (err_inv, (domino[1], domino[0]))


def _paver_grille(lignes: int, colonnes: int) -> tuple[list, np.ndarray]:
    """
    Pavage initial sans trou : tout horizontal si colonnes pair, tout vertical sinon.
    Retourne (placements_slots, grille_slots).
    """
    placements_slots = []
    grille_slots = np.zeros((lignes, colonnes), dtype=int)
    idx = 0

    if colonnes % 2 == 0:
        for i in range(lignes):
            for j in range(0, colonnes, 2):
                placements_slots.append([(i, j), (i, j + 1)])
                grille_slots[i, j] = grille_slots[i, j + 1] = idx
                idx += 1
    else:
        for j in range(colonnes):
            for i in range(0, lignes, 2):
                placements_slots.append([(i, j), (i + 1, j)])
                grille_slots[i, j] = grille_slots[i + 1, j] = idx
                idx += 1

    return placements_slots, grille_slots


def _optimiser_orientation(
    placements_slots: list, grille_slots: np.ndarray, matrice: np.ndarray
) -> list:
    """
    Swapping 2×2 : réoriente les dominos pour mieux suivre les contours.
    Modifie placements_slots en place et retourne la liste.
    """
    lignes, colonnes = matrice.shape
    amelioration = True
    iterations = 0

    while amelioration and iterations < 10:
        amelioration = False
        iterations += 1
        for i in range(lignes - 1):
            for j in range(colonnes - 1):
                idx1 = grille_slots[i, j]
                idx2 = grille_slots[i, j + 1]
                idx3 = grille_slots[i + 1, j]
                idx4 = grille_slots[i + 1, j + 1]

                v_hg = matrice[i, j]
                v_hd = matrice[i, j + 1]
                v_bg = matrice[i + 1, j]
                v_bd = matrice[i + 1, j + 1]
                diff_H = abs(v_hg - v_hd) + abs(v_bg - v_bd)
                diff_V = abs(v_hg - v_bg) + abs(v_hd - v_bd)

                # Cas A : 2 dominos horizontaux empilés → tenter vertical
                if idx1 == idx2 and idx3 == idx4 and idx1 != idx3 and diff_V < diff_H:
                    placements_slots[idx1] = [(i, j),     (i + 1, j)]
                    placements_slots[idx3] = [(i, j + 1), (i + 1, j + 1)]
                    grille_slots[i, j] = grille_slots[i + 1, j] = idx1
                    grille_slots[i, j + 1] = grille_slots[i + 1, j + 1] = idx3
                    amelioration = True

                # Cas B : 2 dominos verticaux côte à côte → tenter horizontal
                elif idx1 == idx3 and idx2 == idx4 and idx1 != idx2 and diff_H < diff_V:
                    placements_slots[idx1] = [(i, j),     (i, j + 1)]
                    placements_slots[idx2] = [(i + 1, j), (i + 1, j + 1)]
                    grille_slots[i, j] = grille_slots[i, j + 1] = idx1
                    grille_slots[i + 1, j] = grille_slots[i + 1, j + 1] = idx2
                    amelioration = True

    return placements_slots


def _generer_emplacements(largeur: int, hauteur: int) -> list[tuple]:
    """Génère la liste des paires de cases (emplacements) pour les algos V2."""
    occupee = [[False] * largeur for _ in range(hauteur)]
    emplacements = []
    for y in range(hauteur):
        for x in range(largeur):
            if occupee[y][x]:
                continue
            if x + 1 < largeur and not occupee[y][x + 1]:
                emplacements.append(((x, y), (x + 1, y)))
                occupee[y][x] = occupee[y][x + 1] = True
            elif y + 1 < hauteur and not occupee[y + 1][x]:
                emplacements.append(((x, y), (x, y + 1)))
                occupee[y][x] = occupee[y + 1][x] = True
    return emplacements


def _tuples_vers_dicts(placements_bruts: list, emplacements: list) -> list[dict]:
    """Convertit une liste de tuples (v1, v2) en liste de dicts universels."""
    return [
        {"case1": (y1, x1), "case2": (y2, x2), "valeurs": (v1, v2)}
        for (v1, v2), ((x1, y1), (x2, y2)) in zip(placements_bruts, emplacements)
    ]


# ─────────────────────────────────────────────────────────────────────
# Algorithme 1 : Glouton par le centre
# ─────────────────────────────────────────────────────────────────────

def glouton(matrice: np.ndarray, stock: list) -> list[dict]:
    """
    Pavage glouton sans trou, priorité au centre de l'image.

    1. Pavage initial horizontal/vertical selon parité des colonnes.
    2. Swapping 2×2 pour mieux suivre les contours.
    3. Assignation gloutonne du stock, du centre vers les bords.

    Args:
        matrice: np.ndarray 2D de valeurs entières.
        stock: liste de tuples (v1, v2).

    Returns:
        Liste de dicts {"case1", "case2", "valeurs"}.
    """
    if not isinstance(matrice, np.ndarray) or matrice.ndim != 2 or matrice.size == 0:
        raise ValueError("matrice doit être un numpy 2D non vide.")

    lignes, colonnes = matrice.shape
    nb_emplacements = (lignes * colonnes) // 2

    if len(stock) < nb_emplacements:
        raise ValueError(
            f"Stock insuffisant : {len(stock)} dominos pour {nb_emplacements} emplacements."
        )

    slots, grille = _paver_grille(lignes, colonnes)
    slots = _optimiser_orientation(slots, grille, matrice)

    centre_i, centre_j = lignes / 2.0, colonnes / 2.0

    def distance(slot):
        c1, c2 = slot
        mi = (c1[0] + c2[0]) / 2.0
        mj = (c1[1] + c2[1]) / 2.0
        return (mi - centre_i) ** 2 + (mj - centre_j) ** 2

    slots.sort(key=distance)
    stock_restant = list(stock)
    placements = []

    for slot in slots:
        c1, c2 = slot
        val1, val2 = matrice[c1[0], c1[1]], matrice[c2[0], c2[1]]

        meilleur_ecart = float("inf")
        meilleur_idx = -1
        inv = False

        for idx_dom, domino in enumerate(stock_restant):
            err_n = abs(val1 - domino[0]) + abs(val2 - domino[1])
            err_i = abs(val1 - domino[1]) + abs(val2 - domino[0])
            if min(err_n, err_i) < meilleur_ecart:
                meilleur_ecart = min(err_n, err_i)
                meilleur_idx = idx_dom
                inv = err_i < err_n

        if meilleur_idx == -1:
            raise RuntimeError(f"Stock épuisé de manière inattendue au slot {slot}.")

        domino = stock_restant.pop(meilleur_idx)
        if inv:
            domino = (domino[1], domino[0])
        placements.append({"case1": c1, "case2": c2, "valeurs": domino})

    return placements


# ─────────────────────────────────────────────────────────────────────
# Algorithme 2 : Hongrois (Kuhn-Munkres exact)
# ─────────────────────────────────────────────────────────────────────

LIMITE_HONGROIS = 5000  # Au-delà, la matrice de coûts devient trop lourde


def hongrois(
    matrice: np.ndarray,
    stock: list,
    progress_callback=None,
) -> list[dict]:
    """
    Affectation optimale exacte par l'algorithme de Kuhn-Munkres.

    Args:
        matrice: np.ndarray 2D de valeurs entières.
        stock: liste de tuples (v1, v2).
        progress_callback: callable(ratio: float, texte: str) optionnel.

    Returns:
        Liste de dicts {"case1", "case2", "valeurs"}.
    """
    lignes, colonnes = matrice.shape
    emplacements = _generer_emplacements(colonnes, lignes)
    nb = len(emplacements)

    if nb > LIMITE_HONGROIS:
        raise ValueError(
            f"Grille trop grande pour l'algorithme Hongrois "
            f"({nb} emplacements > limite {LIMITE_HONGROIS}). "
            "Réduisez la largeur ou utilisez l'algorithme Glouton."
        )

    if progress_callback:
        progress_callback(0.1, "Étape 1/2 : Calcul de la matrice des coûts...")

    valeurs_cibles = [(matrice[y1, x1], matrice[y2, x2]) for ((x1, y1), (x2, y2)) in emplacements]
    matrice_couts = np.zeros((nb, nb), dtype=int)

    for i, (c1, c2) in enumerate(valeurs_cibles):
        for j, domino in enumerate(stock):
            err_n = abs(domino[0] - c1) + abs(domino[1] - c2)
            err_i = abs(domino[1] - c1) + abs(domino[0] - c2)
            matrice_couts[i, j] = min(err_n, err_i)

    if progress_callback:
        progress_callback(0.5, "Étape 2/2 : Résolution mathématique exacte...")

    _, col_ind = linear_sum_assignment(matrice_couts)

    placements_bruts = []
    for i, j in enumerate(col_ind):
        domino = stock[j]
        c1, c2 = valeurs_cibles[i]
        err_n = abs(domino[0] - c1) + abs(domino[1] - c2)
        err_i = abs(domino[1] - c1) + abs(domino[0] - c2)
        placements_bruts.append(domino if err_n <= err_i else (domino[1], domino[0]))

    return _tuples_vers_dicts(placements_bruts, emplacements)


# ─────────────────────────────────────────────────────────────────────
# Algorithme 3 : Recuit simulé (méta-heuristique)
# ─────────────────────────────────────────────────────────────────────

def recuit(
    matrice: np.ndarray,
    stock: list,
    iterations: int = 150_000,
    progress_callback=None,
) -> list[dict]:
    """
    Optimisation par recuit simulé : échange aléatoire de dominos
    accepté selon un critère de Metropolis.

    Args:
        matrice: np.ndarray 2D de valeurs entières.
        stock: liste de tuples (v1, v2).
        iterations: nombre d'itérations (défaut 150 000).
        progress_callback: callable(ratio: float, texte: str) optionnel.

    Returns:
        Liste de dicts {"case1", "case2", "valeurs"}.
    """
    emplacements = _generer_emplacements(matrice.shape[1], matrice.shape[0])
    inventaire = list(stock)
    random.shuffle(inventaire)
    placement_actuel = list(inventaire)

    temp = 10.0
    temp_finale = 0.01
    alpha = (temp_finale / temp) ** (1 / iterations)
    step = max(1, iterations // 20)

    for i in range(iterations):
        idx1, idx2 = random.sample(range(len(emplacements)), 2)
        (x1A, y1A), (x1B, y1B) = emplacements[idx1]
        (x2A, y2A), (x2B, y2B) = emplacements[idx2]

        c1A, c1B = matrice[y1A, x1A], matrice[y1B, x1B]
        c2A, c2B = matrice[y2A, x2A], matrice[y2B, x2B]

        err1_av, _ = _erreur_domino(placement_actuel[idx1], c1A, c1B)
        err2_av, _ = _erreur_domino(placement_actuel[idx2], c2A, c2B)
        err1_ap, _ = _erreur_domino(placement_actuel[idx2], c1A, c1B)
        err2_ap, _ = _erreur_domino(placement_actuel[idx1], c2A, c2B)

        delta = (err1_ap + err2_ap) - (err1_av + err2_av)
        if delta < 0 or random.random() < math.exp(-delta / temp):
            placement_actuel[idx1], placement_actuel[idx2] = placement_actuel[idx2], placement_actuel[idx1]

        temp *= alpha

        if progress_callback and i % step == 0:
            progress_callback(i / iterations, "Optimisation des pièces en cours...")

    placements_bruts = [
        _erreur_domino(dom, matrice[empl[0][1], empl[0][0]], matrice[empl[1][1], empl[1][0]])[1]
        for dom, empl in zip(placement_actuel, emplacements)
    ]
    return _tuples_vers_dicts(placements_bruts, emplacements)


# ─────────────────────────────────────────────────────────────────────
# Utilitaire : score de fidélité
# ─────────────────────────────────────────────────────────────────────

def calculer_score(placements: list[dict], matrice: np.ndarray, valeur_max: int) -> float:
    """Retourne le score de fidélité en % (0–100)."""
    erreur = sum(
        abs(matrice[p["case1"][0], p["case1"][1]] - p["valeurs"][0]) +
        abs(matrice[p["case2"][0], p["case2"][1]] - p["valeurs"][1])
        for p in placements
    )
    return 100 * (1 - erreur / (matrice.size * valeur_max))