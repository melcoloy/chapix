import numpy as np

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
    Algorithme 100% garanti SANS TROUS avec PRIORITÉ CENTRALE :
    1. Pavage initial trivial (tout horizontal ou vertical).
    2. Optimisation par permutations 2x2 (Swapping) pour suivre les contours.
    3. Remplissage glouton en commençant par le CENTRE de l'image.
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
    # On repasse sur l'image plusieurs fois (max 10) pour lisser les contours
    while amelioration and iterations < 10:
        amelioration = False
        iterations += 1
        for i in range(lignes - 1):
            for j in range(colonnes - 1):
                idx1 = grille_slots[i, j]
                idx2 = grille_slots[i, j+1]
                idx3 = grille_slots[i+1, j]
                idx4 = grille_slots[i+1, j+1]
                
                # Cas A : On trouve 2 dominos horizontaux empilés
                if idx1 == idx2 and idx3 == idx4 and idx1 != idx3:
                    v_hg, v_hd = matrice_valeurs[i, j], matrice_valeurs[i, j+1]
                    v_bg, v_bd = matrice_valeurs[i+1, j], matrice_valeurs[i+1, j+1]
                    
                    diff_H = abs(v_hg - v_hd) + abs(v_bg - v_bd)
                    diff_V = abs(v_hg - v_bg) + abs(v_hd - v_bd)
                    
                    if diff_V < diff_H: # Le sens Vertical épouse mieux l'image !
                        placements_slots[idx1] = [(i, j), (i+1, j)]
                        placements_slots[idx3] = [(i, j+1), (i+1, j+1)]
                        grille_slots[i, j] = idx1
                        grille_slots[i+1, j] = idx1
                        grille_slots[i, j+1] = idx3
                        grille_slots[i+1, j+1] = idx3
                        amelioration = True
                        
                # Cas B : On trouve 2 dominos verticaux côte à côte
                elif idx1 == idx3 and idx2 == idx4 and idx1 != idx2:
                    v_hg, v_hd = matrice_valeurs[i, j], matrice_valeurs[i, j+1]
                    v_bg, v_bd = matrice_valeurs[i+1, j], matrice_valeurs[i+1, j+1]
                    
                    diff_H = abs(v_hg - v_hd) + abs(v_bg - v_bd)
                    diff_V = abs(v_hg - v_bg) + abs(v_hd - v_bd)
                    
                    if diff_H < diff_V: # Le sens Horizontal épouse mieux l'image !
                        placements_slots[idx1] = [(i, j), (i, j+1)]
                        placements_slots[idx2] = [(i+1, j), (i+1, j+1)]
                        grille_slots[i, j] = idx1
                        grille_slots[i, j+1] = idx1
                        grille_slots[i+1, j] = idx2
                        grille_slots[i+1, j+1] = idx2
                        amelioration = True

    # --- ETAPE 3 : Assignation gloutonne du stock (CENTRE D'ABORD) ---
    placements = []
    stock_restant = stock_dominos.copy()
    
    # 1. On calcule les coordonnées du centre de l'image
    centre_i = lignes / 2.0
    centre_j = colonnes / 2.0
    
    # 2. Fonction mathématique pour calculer la distance d'un slot au centre
    def distance_au_centre(slot):
        c1, c2 = slot
        milieu_i = (c1[0] + c2[0]) / 2.0
        milieu_j = (c1[1] + c2[1]) / 2.0
        return (milieu_i - centre_i)**2 + (milieu_j - centre_j)**2

    # 3. La magie : on trie tous les emplacements du centre vers les bords !
    placements_slots.sort(key=distance_au_centre)
    
    # 4. On distribue les dominos (le centre recevra les meilleures pièces)
    for slot in placements_slots:
        c1, c2 = slot
        val1 = matrice_valeurs[c1[0], c1[1]]
        val2 = matrice_valeurs[c2[0], c2[1]]
        
        meilleur_ecart = 999
        meilleur_idx = -1
        inv = False
        
        # On cherche la meilleure pièce restante
        for idx_dom, domino in enumerate(stock_restant):
            d1, d2 = domino
            err_norm = abs(val1 - d1) + abs(val2 - d2)
            err_inv = abs(val1 - d2) + abs(val2 - d1)
            min_err = min(err_norm, err_inv)
            
            if min_err < meilleur_ecart:
                meilleur_ecart = min_err
                meilleur_idx = idx_dom
                inv = (err_inv < err_norm)
                
        domino_choisi = stock_restant.pop(meilleur_idx)
        
        # Inversion du domino si cela permet une meilleure correspondance
        if inv:
            domino_choisi = (domino_choisi[1], domino_choisi[0])
            
        placements.append({
            "case1": c1,
            "case2": c2,
            "valeurs": domino_choisi
        })
        
    return placements