from PIL import Image, ImageDraw, ImageOps
import math
import numpy as np

def preparer_image(image_originale, total_dominos):
    """
    Convertit l'image en noir et blanc(niveau de gris) et la redimensionne 
    en utilisant les diviseurs parfaits pour vider 100% du stock. et bien respecter la proportion des jeux des dominos
    """
    image_nb = image_originale.convert("L") # conversion en NB

    image_nb = ImageOps.autocontrast(image_nb)
    
    surface_cible = total_dominos * 2  # nb pixels = 2* nb_dominos
    
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
    # convertit chaque valeur des pixels de 0-255 à 0-6
    valeur_max = 6 if type_jeu == "double_six" else 9
    matrice_pixels = np.array(image_pil) 
    matrice_dominos = matrice_pixels / 255.0 
    matrice_dominos = matrice_dominos * valeur_max
    matrice_dominos = np.round(matrice_dominos).astype(int)
    return matrice_dominos

def dessiner_mosaique(placements, lignes, colonnes, taille_case=40):
    """Crée l'image finale en dessinant les dominos un par un."""
    largeur_img = colonnes * taille_case
    hauteur_img = lignes * taille_case
    
    image_finale = Image.new("RGB", (largeur_img, hauteur_img), "black")
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
            dessin.ellipse([px-r, py-r, px+r, py+r], fill="white")

    for p in placements:
        i1, j1 = p["case1"]
        i2, j2 = p["case2"]
        v1, v2 = p["valeurs"]

        x1, y1 = j1 * taille_case, i1 * taille_case
        x2, y2 = j2 * taille_case, i2 * taille_case

        x_min, y_min = min(x1, x2), min(y1, y2)
        x_max, y_max = max(x1, x2) + taille_case, max(y1, y2) + taille_case

        dessin.rectangle([x_min, y_min, x_max, y_max], fill=(30, 30, 30), outline="white", width=2)
        
        if i1 == i2: 
            dessin.line([x2, y_min, x2, y_max], fill="white", width=2)
        else: 
            dessin.line([x_min, y2, x_max, y2], fill="white", width=2)

        dessiner_points(x1, y1, v1)
        dessiner_points(x2, y2, v2)

    return image_finale

if __name__ == "__main__":
    print("Fichier traitement_image.py prêt.")
    