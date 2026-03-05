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
    Parcourt la matrice et place les dominos de façon gloutonne.
    Retourne une liste contenant les coordonnées et les valeurs des dominos placés.
    """
    lignes, colonnes = matrice_valeurs.shape
    couvert = np.zeros((lignes, colonnes), dtype=bool)
    placements = []
    stock_restant = stock_dominos.copy()
    
    for i in range(lignes):
        for j in range(colonnes):
            if couvert[i, j]:
                continue
                
            valeur_pixel_1 = matrice_valeurs[i, j]
            
            if j + 1 < colonnes and not couvert[i, j + 1]:
                i2, j2 = i, j + 1
            elif i + 1 < lignes and not couvert[i + 1, j]:
                i2, j2 = i + 1, j
            else:
                continue
                
            valeur_pixel_2 = matrice_valeurs[i2, j2]
            
            meilleur_index = 0
            meilleur_ecart = 999
            domino_inverse = False
            
            for index, domino in enumerate(stock_restant):
                d1, d2 = domino
                ecart_normal = abs(valeur_pixel_1 - d1) + abs(valeur_pixel_2 - d2)
                ecart_retourne = abs(valeur_pixel_1 - d2) + abs(valeur_pixel_2 - d1)
                
                if ecart_normal < meilleur_ecart:
                    meilleur_ecart = ecart_normal
                    meilleur_index = index
                    domino_inverse = False
                    
                if ecart_retourne < meilleur_ecart:
                    meilleur_ecart = ecart_retourne
                    meilleur_index = index
                    domino_inverse = True
            
            domino_choisi = stock_restant.pop(meilleur_index)
            
            if domino_inverse:
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