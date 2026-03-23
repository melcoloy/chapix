from PIL import Image, ImageDraw, ImageOps, ImageFilter
import math
import numpy as np

def preparer_image(image_originale, total_dominos, renforcer_contraste=False):
    """
    Convertit l'image en noir et blanc(niveau de gris) et la redimensionne 
    en utilisant les diviseurs parfaits pour vider 100% du stock.
    """
    image_nb = image_originale.convert("L") 
    image_nb = ImageOps.autocontrast(image_nb)
    if renforcer_contraste:
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

def image_vers_matrice(image_pil, type_jeu="double_six"):
    """Convertit l'image (0-255) en une matrice de dominos (0-6 ou 0-9)."""
    valeur_max = 6 if type_jeu == "double_six" else 9
    matrice_pixels = np.array(image_pil) 
    matrice_dominos = matrice_pixels / 255.0 
    matrice_dominos = matrice_dominos * valeur_max
    matrice_dominos = np.round(matrice_dominos).astype(int)
    
    # --- CORRECTION DE L'EFFET NÉGATIF ---
    # Pour des dominos blancs à points noirs, un pixel noir sur la photo
    # doit correspondre au domino avec le plus de points noirs (le 6 ou le 9).
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
        r = taille_case // 12 
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
    padding = max(1, taille_case // 15) # Espace entre les dominos
    rayon_arrondi = taille_case // 5    # Pour les coins doux

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

        # On dessine le domino blanc avec des coins arrondis
        try:
            dessin.rounded_rectangle([x1_pad, y1_pad, x2_pad, y2_pad], radius=rayon_arrondi, fill="white", outline="black", width=2)
        except AttributeError:
            # Sécurité si une très vieille version de Pillow est installée
            dessin.rectangle([x1_pad, y1_pad, x2_pad, y2_pad], fill="white", outline="black", width=2)
        
        # La ligne de séparation au milieu (plus fine que la bordure globale)
        if i1 == i2: # Domino horizontal
            dessin.line([x2, y1_pad, x2, y2_pad], fill="black", width=2)
        else: # Domino vertical
            dessin.line([x1_pad, y2, x2_pad, y2], fill="black", width=2)

        # On dessine les points par-dessus
        dessiner_points(x1, y1, v1)
        dessiner_points(x2, y2, v2)

    return image_finale