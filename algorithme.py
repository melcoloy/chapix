import numpy as np

def generer_inventaire(type_jeu="double_six", nb_boites=1):
    """Génère la liste des dominos (tuples) selon le jeu choisi."""
    if type_jeu == "double_six":
        valeur_max = 6
    else:
        valeur_max = 9
    boite_unique = []
    
    # Création d'une boîte complète de dominos
    for i in range(valeur_max + 1):
        for j in range(i, valeur_max + 1):
            boite_unique.append((i, j))
            
    # On multiplie par le nombre total de boîtes 
    jeu_complet = boite_unique * nb_boites
    return jeu_complet

def placer_dominos(matrice_valeurs, stock_dominos):
    """
    Parcourt la matrice et place les dominos de façon gloutonne INTELLIGENTE.
    Teste l'orientation Horizontale ET Verticale pour épouser les contours de l'image.
    """
    lignes, colonnes = matrice_valeurs.shape
    couvert = np.zeros((lignes, colonnes), dtype=bool)
    placements = []
    stock_restant = stock_dominos.copy()
    
    for i in range(lignes):
        for j in range(colonnes):
            if couvert[i, j]:
                continue
                
            valeur_1 = matrice_valeurs[i, j]
            
            # On vérifie quelles directions sont possibles
            peut_H = (j + 1 < colonnes) and not couvert[i, j + 1]
            peut_V = (i + 1 < lignes) and not couvert[i + 1, j]
            
            meilleur_ecart_H, meilleur_idx_H, inv_H = 999, -1, False
            meilleur_ecart_V, meilleur_idx_V, inv_V = 999, -1, False
            
            # On parcourt le stock pour évaluer les deux directions
            for index, domino in enumerate(stock_restant):
                d1, d2 = domino
                
                # Test Horizontal
                if peut_H:
                    v2_H = matrice_valeurs[i, j + 1]
                    err_H_norm = abs(valeur_1 - d1) + abs(v2_H - d2)
                    err_H_inv = abs(valeur_1 - d2) + abs(v2_H - d1)
                    min_err_H = min(err_H_norm, err_H_inv)
                    if min_err_H < meilleur_ecart_H:
                        meilleur_ecart_H = min_err_H
                        meilleur_idx_H = index
                        inv_H = (err_H_inv < err_H_norm)
                
                # Test Vertical
                if peut_V:
                    v2_V = matrice_valeurs[i + 1, j]
                    err_V_norm = abs(valeur_1 - d1) + abs(v2_V - d2)
                    err_V_inv = abs(valeur_1 - d2) + abs(v2_V - d1)
                    min_err_V = min(err_V_norm, err_V_inv)
                    if min_err_V < meilleur_ecart_V:
                        meilleur_ecart_V = min_err_V
                        meilleur_idx_V = index
                        inv_V = (err_V_inv < err_V_norm)

            # --- CORRECTION DU BUG : Le choix intelligent de l'orientation ---
            if peut_H and peut_V:
                choix = 'H' if meilleur_ecart_H <= meilleur_ecart_V else 'V'
            elif peut_H:
                choix = 'H'
            elif peut_V:
                choix = 'V'
            else:
                # La case est isolée ! On ne peut placer ni H ni V.
                # On passe à la case suivante pour ne pas déborder de l'image.
                continue
                
            # Application du choix
            if choix == 'H':
                i2, j2 = i, j + 1
                best_idx = meilleur_idx_H
                is_inv = inv_H
            else:
                i2, j2 = i + 1, j
                best_idx = meilleur_idx_V
                is_inv = inv_V
                
            domino_choisi = stock_restant.pop(best_idx)
            
            if is_inv:
                domino_choisi = (domino_choisi[1], domino_choisi[0])
                
            placements.append({
                "case1": (i, j),
                "case2": (i2, j2),
                "valeurs": domino_choisi
            })
            
            couvert[i, j] = True
            couvert[i2, j2] = True
            
    return placements

if __name__ == "__main__":
    print("Fichier algorithme.py prêt.")
