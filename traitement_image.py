from PIL import Image, ImageDraw, ImageOps, ImageFilter
import math
import numpy as np

def preparer_image(image_originale, total_dominos, renforcer_contours=False):
    """
    Convertit l'image en noir et blanc (niveau de gris) et la redimensionne 
    en utilisant les diviseurs parfaits pour vider 100% du stock.
    """
    image_nb = image_originale.convert("L") 
    image_nb = ImageOps.autocontrast(image_nb)
    
    # Segmentation optionnelle demandée par l'utilisateur
    if renforcer_contours:
        image_nb = image_nb.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    surface_cible = total_dominos * 2  
    
    largeur_orig, hauteur_orig = image_nb.size
    ratio_cible = largeur_orig / hauteur_orig
    
    nouvelle_largeur = surface_cible
    nouvelle_hauteur = 1
    meilleure_diff = float('inf')
    
    for h in range(1, surface_cible + 1):
        if surface_cible % h == 0:  
            l = surface_cible // h
            ratio_actuel = l / h
            diff = abs(ratio_actuel - ratio_cible)
            
            if diff < meilleure_diff:
                meilleure_diff = diff
                nouvelle_largeur = l
                nouvelle_hauteur = h
                
    image_redimensionnee = image_nb.resize((nouvelle_largeur, nouvelle_hauteur))
    return image_redimensionnee

def image_vers_matrice(image_pil, type_jeu="double_six", appliquer_dithering=True):
    """Convertit l'image en une matrice de dominos, avec propagation d'erreur (Floyd-Steinberg)."""
    valeur_max = 6 if type_jeu == "double_six" else 9
    
    # On travaille avec des nombres à virgule (float) pour garder l'erreur
    matrice = np.array(image_pil, dtype=float)
    matrice = (matrice / 255.0) * valeur_max
    
    lignes, colonnes = matrice.shape
    
    if appliquer_dithering:
        # Algorithme de Floyd-Steinberg
        for i in range(lignes):
            for j in range(colonnes):
                ancienne_valeur = matrice[i, j]
                nouvelle_valeur = np.round(ancienne_valeur)
                # On s'assure de ne pas dépasser les limites des dominos
                nouvelle_valeur = min(valeur_max, max(0, nouvelle_valeur))
                matrice[i, j] = nouvelle_valeur
                
                # Calcul de l'erreur d'arrondi
                erreur = ancienne_valeur - nouvelle_valeur
                
                # On propage l'erreur aux voisins (selon les fractions de Floyd-Steinberg)
                if j + 1 < colonnes:
                    matrice[i, j + 1] += erreur * (7 / 16)
                if i + 1 < lignes:
                    if j - 1 >= 0:
                        matrice[i + 1, j - 1] += erreur * (3 / 16)
                    matrice[i + 1, j] += erreur * (5 / 16)
                    if j + 1 < colonnes:
                        matrice[i + 1, j + 1] += erreur * (1 / 16)
    else:
        matrice = np.round(matrice)
        
    matrice_dominos = matrice.astype(int)
    matrice_dominos = np.clip(matrice_dominos, 0, valeur_max)
    
    # CORRECTION DE L'EFFET NÉGATIF (Dominos blancs = fond blanc)
    matrice_dominos = valeur_max - matrice_dominos
    
    return matrice_dominos

def dessiner_mosaique(placements, lignes, colonnes, taille_case=40):
    """Crée l'image finale avec des dominos blancs séparés visuellement."""
    largeur_img = colonnes * taille_case
    hauteur_img = lignes * taille_case
    
    # Fond global de l'image (Gris très foncé pour faire ressortir les dominos blancs)
    couleur_fond = (40, 40, 40)
    image_finale = Image.new("RGB", (largeur_img, hauteur_img), couleur_fond)
    dessin = ImageDraw.Draw(image_finale)

    def dessiner_points(x, y, valeur):
        marge = taille_case // 4
        # Le choix de l'utilisateur : // 10 pour l'équilibre parfait
        r = taille_case // 10 
        cx, cy = x + taille_case//2, y + taille_case//2 
        
        pos = {
            'c': (cx, cy),
            'hg': (x + marge, y + marge), 'hd': (x + taille_case - marge, y + marge),
            'bg': (x + marge, y + taille_case - marge), 'bd': (x + taille_case - marge, y + taille_case - marge),
            'mg': (x + marge, cy), 'md': (x + taille_case - marge, cy),
            'hm': (cx, y + marge), 'bm': (cx, y + taille_case - marge)
        }
        
        pts = []
        if valeur == 1: pts = ['c']
        elif valeur == 2: pts = ['hg', 'bd']
        elif valeur == 3: pts = ['hg', 'c', 'bd']
        elif valeur == 4: pts = ['hg', 'hd', 'bg', 'bd']
        elif valeur == 5: pts = ['hg', 'hd', 'c', 'bg', 'bd']
        elif valeur == 6: pts = ['hg', 'hd', 'mg', 'md', 'bg', 'bd']
        elif valeur == 7: pts = ['hg', 'hd', 'mg', 'md', 'bg', 'bd', 'c']
        elif valeur == 8: pts = ['hg', 'hd', 'mg', 'md', 'bg', 'bd', 'hm', 'bm']
        elif valeur == 9: pts = ['hg', 'hd', 'mg', 'md', 'bg', 'bd', 'hm', 'bm', 'c']

        for p in pts:
            px, py = pos[p]
            # Les points (pips) sont noirs
            dessin.ellipse([px-r, py-r, px+r, py+r], fill="black")

    # Calcul de l'espacement et de l'arrondi
    padding = max(1, taille_case // 15) 
    rayon_arrondi = taille_case // 5    

    for p in placements:
        i1, j1 = p["case1"]
        i2, j2 = p["case2"]
        v1, v2 = p["valeurs"]

        x1, y1 = j1 * taille_case, i1 * taille_case
        x2, y2 = j2 * taille_case, i2 * taille_case

        x_min, y_min = min(x1, x2), min(y1, y2)
        x_max, y_max = max(x1, x2) + taille_case, max(y1, y2) + taille_case

        # Coordonnées du domino AVEC l'espacement (padding)
        x1_pad = x_min + padding
        y1_pad = y_min + padding
        x2_pad = x_max - padding
        y2_pad = y_max - padding

        # On dessine le domino avec des bordures fines (width=1)
        try:
            dessin.rounded_rectangle([x1_pad, y1_pad, x2_pad, y2_pad], radius=rayon_arrondi, fill="white", outline="black", width=1)
        except AttributeError:
            dessin.rectangle([x1_pad, y1_pad, x2_pad, y2_pad], fill="white", outline="black", width=1)
        
        # La ligne de séparation au milieu fine (width=1)
        if i1 == i2: # Domino horizontal
            dessin.line([x2, y1_pad, x2, y2_pad], fill="black", width=1)
        else: # Domino vertical
            dessin.line([x1_pad, y2, x2_pad, y2], fill="black", width=1)

        # On dessine les points par-dessus
        dessiner_points(x1, y1, v1)
        dessiner_points(x2, y2, v2)

    return image_finale

def mettre_en_evidence(image_base, placements, colonnes, domino_cible):
    """Dessine un cadre rouge autour des dominos ciblés en s'adaptant parfaitement à l'échelle."""
    from PIL import ImageDraw
    
    img_out = image_base.copy()
    draw = ImageDraw.Draw(img_out)
    
    taille_case = image_base.width // colonnes
    epaisseur = max(4, taille_case // 15) 
    
    c1, c2 = domino_cible
    
    for p in placements:
        v1, v2 = p["valeurs"]
        if min(v1, v2) == min(c1, c2) and max(v1, v2) == max(c1, c2):
            i1, j1 = p["case1"] 
            i2, j2 = p["case2"]
            
            x_min = min(j1, j2) * taille_case
            y_min = min(i1, i2) * taille_case
            x_max = max(j1, j2) * taille_case + taille_case
            y_max = max(i1, i2) * taille_case + taille_case
            
            draw.rectangle(
                [x_min + epaisseur, y_min + epaisseur, x_max - epaisseur, y_max - epaisseur], 
                outline="#FF0044", 
                width=epaisseur
            )
            
    return img_out