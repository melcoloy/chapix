import numpy as np
from scipy.optimize import linear_sum_assignment

def generer_inventaire(type_jeu="double_six", nb_boites=1):
    """Génère la liste des dominos (tuples) selon le jeu choisi."""
    valeur_max = 6 if type_jeu == "double_six" else 9
    boite_unique = []
    
    for i in range(valeur_max + 1):
        for j in range(i, valeur_max + 1):
            boite_unique.append((i, j))
            
    jeu_complet = boite_unique * nb_boites
    return jeu_complet

def placer_dominos(matrice_valeurs, stock_dominos):
    """
    Algorithme Ultime : 
    1. Pavage sans trou (Swapping)
    2. Assignation globale OPTIMALE via la Méthode Hongroise (Kuhn-Munkres)
    """
    lignes, colonnes = matrice_valeurs.shape
    placements_slots = []
    grille_slots = np.zeros((lignes, colonnes), dtype=int)
    
    # --- ETAPE 1 : Pavage de base (Zéro trou garanti) ---
    idx = 0
    if colonnes % 2 == 0:
        for i in range(lignes):
            for j in range(0, colonnes, 2):
                placements_slots.append([(i, j), (i, j+1)])
                grille_slots[i, j] = idx
                grille_slots[i, j+1] = idx
                idx += 1
    else:
        for j in range(colonnes):
            for i in range(0, lignes, 2):
                placements_slots.append([(i, j), (i+1, j)])
                grille_slots[i, j] = idx
                grille_slots[i+1, j] = idx
                idx += 1
                
    # --- ETAPE 2 : Optimisation (Swapping intelligent des contours) ---
    amelioration = True
    iterations = 0
    while amelioration and iterations < 10:
        amelioration = False
        iterations += 1
        for i in range(lignes - 1):
            for j in range(colonnes - 1):
                idx1 = grille_slots[i, j]
                idx2 = grille_slots[i, j+1]
                idx3 = grille_slots[i+1, j]
                idx4 = grille_slots[i+1, j+1]
                
                # Cas A : 2 horizontaux empilés
                if idx1 == idx2 and idx3 == idx4 and idx1 != idx3:
                    v_hg, v_hd = matrice_valeurs[i, j], matrice_valeurs[i, j+1]
                    v_bg, v_bd = matrice_valeurs[i+1, j], matrice_valeurs[i+1, j+1]
                    diff_H = abs(v_hg - v_hd) + abs(v_bg - v_bd)
                    diff_V = abs(v_hg - v_bg) + abs(v_hd - v_bd)
                    if diff_V < diff_H:
                        placements_slots[idx1] = [(i, j), (i+1, j)]
                        placements_slots[idx3] = [(i, j+1), (i+1, j+1)]
                        grille_slots[i, j], grille_slots[i+1, j] = idx1, idx1
                        grille_slots[i, j+1], grille_slots[i+1, j+1] = idx3, idx3
                        amelioration = True
                        
                # Cas B : 2 verticaux côte à côte
                elif idx1 == idx3 and idx2 == idx4 and idx1 != idx2:
                    v_hg, v_hd = matrice_valeurs[i, j], matrice_valeurs[i, j+1]
                    v_bg, v_bd = matrice_valeurs[i+1, j], matrice_valeurs[i+1, j+1]
                    diff_H = abs(v_hg - v_hd) + abs(v_bg - v_bd)
                    diff_V = abs(v_hg - v_bg) + abs(v_hd - v_bd)
                    if diff_H < diff_V:
                        placements_slots[idx1] = [(i, j), (i, j+1)]
                        placements_slots[idx2] = [(i+1, j), (i+1, j+1)]
                        grille_slots[i, j], grille_slots[i, j+1] = idx1, idx1
                        grille_slots[i+1, j], grille_slots[i+1, j+1] = idx2, idx2
                        amelioration = True

    # --- ETAPE 3 : LA MÉTHODE HONGROISE (Affectation Optimale) ---
    N = len(placements_slots)
    matrice_couts = np.zeros((N, N))
    orientations = np.zeros((N, N), dtype=bool) 
    
    for i, slot in enumerate(placements_slots):
        c1, c2 = slot
        val1 = matrice_valeurs[c1[0], c1[1]]
        val2 = matrice_valeurs[c2[0], c2[1]]
        
        for j, domino in enumerate(stock_dominos):
            d1, d2 = domino
            err_norm = abs(val1 - d1) + abs(val2 - d2)
            err_inv = abs(val1 - d2) + abs(val2 - d1)
            
            if err_norm <= err_inv:
                matrice_couts[i, j] = err_norm
                orientations[i, j] = False
            else:
                matrice_couts[i, j] = err_inv
                orientations[i, j] = True
                
    # L'algorithme de Kuhn-Munkres 
    row_ind, col_ind = linear_sum_assignment(matrice_couts)
    
    # --- ETAPE 4 : Assemblage final ---
    placements = []
    for i in range(N):
        idx_slot = row_ind[i]
        idx_domino = col_ind[i]
        
        slot = placements_slots[idx_slot]
        domino = stock_dominos[idx_domino]
        est_inverse = orientations[idx_slot, idx_domino]
        
        if est_inverse:
            domino_choisi = (domino[1], domino[0])
        else:
            domino_choisi = domino
            
        placements.append({
            "case1": slot[0],
            "case2": slot[1],
            "valeurs": domino_choisi
        })
        
    return placements