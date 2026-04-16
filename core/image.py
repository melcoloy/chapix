#Traitement d'image : préparation, conversion en matrice, dessin, mise en évidence.

from PIL import Image, ImageDraw, ImageOps, ImageFilter
import numpy as np

_DISPOSITION_PIPS = {
    0: [],
    1: ["c"],
    2: ["hg", "bd"],
    3: ["hg", "c", "bd"],
    4: ["hg", "hd", "bg", "bd"],
    5: ["hg", "hd", "c", "bg", "bd"],
    6: ["hg", "hd", "mg", "md", "bg", "bd"],
    7: ["hg", "hd", "mg", "md", "bg", "bd", "c"],
    8: ["hg", "hd", "mg", "md", "bg", "bd", "hm", "bm"],
    9: ["hg", "hd", "mg", "md", "bg", "bd", "hm", "bm", "c"],
}


#Redimensionne l'image si nécessaire, la convertit en niveaux de gris et met en évidence les contours
def preparer_image(
    image_originale: Image.Image,
    largeur: int,
    hauteur: int,
    renforcer_contours: bool = False,
) -> Image.Image:
    
    if not isinstance(image_originale, Image.Image):
        raise TypeError(f"Attendu PIL.Image, reçu : {type(image_originale).__name__}")
    image_redimensionnee = image_originale.resize((largeur, hauteur), Image.Resampling.LANCZOS)
    image_nb = ImageOps.autocontrast(image_redimensionnee.convert("L"))
    if renforcer_contours:
        image_nb = image_nb.filter(ImageFilter.EDGE_ENHANCE_MORE)

    return image_nb

#Convertit l'image vers une matrice de valeurs, en appliquant le dithering (propagation de l'erreur)
def image_vers_matrice(
    image_pil: Image.Image,
    type_jeu: str = "double_six",
    appliquer_dithering: bool = True,
) -> np.ndarray:
    
    from core.inventaire import valeur_max as get_valeur_max

    if not isinstance(image_pil, Image.Image):
        raise TypeError(f"Attendu PIL.Image, reçu : {type(image_pil).__name__}")
    if image_pil.mode != "L":
        image_pil = image_pil.convert("L")

    vmax = get_valeur_max(type_jeu)
    matrice = np.array(image_pil, dtype=float) / 255.0 * vmax
    lignes, colonnes = matrice.shape

    if appliquer_dithering:
        for i in range(lignes):
            for j in range(colonnes):
                ancienne = matrice[i, j]
                nouvelle = float(np.clip(round(ancienne), 0, vmax))
                matrice[i, j] = nouvelle
                erreur = ancienne - nouvelle
                if j + 1 < colonnes:
                    matrice[i, j + 1] += erreur * 7 / 16
                if i + 1 < lignes:
                    if j - 1 >= 0:
                        matrice[i + 1, j - 1] += erreur * 3 / 16
                    matrice[i + 1, j] += erreur * 5 / 16
                    if j + 1 < colonnes:
                        matrice[i + 1, j + 1] += erreur * 1 / 16
    else:
        matrice = np.round(matrice)

    matrice = np.clip(matrice, 0, vmax).astype(int)
    return vmax - matrice  # inversion : blanc = fond blanc

#Retourne la mosaïque finale dessinée à partir de l'emplacement des dominos
def dessiner_mosaique(
    placements: list[dict],
    lignes: int,
    colonnes: int,
    taille_case: int = 40,
) -> Image.Image:
    
    if not placements:
        raise ValueError("La liste de placements est vide.")
    if lignes < 1 or colonnes < 1:
        raise ValueError(f"Dimensions invalides : {lignes}×{colonnes}.")
    if taille_case < 10:
        raise ValueError(f"taille_case trop petite : {taille_case} (minimum 10).")

    image_finale = Image.new("RGB", (colonnes * taille_case, lignes * taille_case), (40, 40, 40))
    dessin = ImageDraw.Draw(image_finale)
    padding = max(1, taille_case // 15)
    rayon = taille_case // 5

    def positions_pips(x: int, y: int) -> dict:
        m, cx, cy = taille_case // 4, x + taille_case // 2, y + taille_case // 2
        return {
            "c":  (cx, cy),
            "hg": (x + m,               y + m),
            "hd": (x + taille_case - m, y + m),
            "bg": (x + m,               y + taille_case - m),
            "bd": (x + taille_case - m, y + taille_case - m),
            "mg": (x + m,               cy),
            "md": (x + taille_case - m, cy),
            "hm": (cx,                  y + m),
            "bm": (cx,                  y + taille_case - m),
        }

    def dessiner_pips(x: int, y: int, valeur: int, couleur="black") -> None:
        r = taille_case // 10
        pos = positions_pips(x, y)
        for p in _DISPOSITION_PIPS.get(valeur, []):
            px, py = pos[p]
            dessin.ellipse([px - r, py - r, px + r, py + r], fill=couleur)

    for p in placements:
        i1, j1 = p["case1"]
        i2, j2 = p["case2"]
        v1, v2 = int(p["valeurs"][0]), int(p["valeurs"][1])

        x1, y1 = j1 * taille_case, i1 * taille_case
        x2, y2 = j2 * taille_case, i2 * taille_case
        x_min, y_min = min(x1, x2), min(y1, y2)
        x_max, y_max = max(x1, x2) + taille_case, max(y1, y2) + taille_case

        rect = [x_min + padding, y_min + padding, x_max - padding, y_max - padding]
        try:
            dessin.rounded_rectangle(rect, radius=rayon, fill="white", outline="black", width=1)
        except AttributeError:
            dessin.rectangle(rect, fill="white", outline="black", width=1)

        if i1 == i2:
            dessin.line([x2, y_min + padding, x2, y_max - padding], fill="black", width=1)
        else:
            dessin.line([x_min + padding, y2, x_max - padding, y2], fill="black", width=1)

        dessiner_pips(x1, y1, v1)
        dessiner_pips(x2, y2, v2)

    return image_finale

#Encadre en rouge les dominos recherchés sur l'image fournie
def mettre_en_evidence(
    image_base: Image.Image,
    placements: list[dict],
    colonnes: int,
    domino_cible: tuple,
) -> Image.Image:
    """
    Dessine un cadre rouge autour des dominos du type sélectionné.

    Args:
        image_base: image PIL de la mosaïque de base (issue de dessiner_mosaique).
        placements: liste de dicts {"case1", "case2", "valeurs"}.
        colonnes: nombre de colonnes de la grille (pour calculer taille_case).
        domino_cible: tuple (v1, v2) du type de domino à mettre en évidence.

    Returns:
        Copie de image_base avec les cadres rouges dessinés.
    """
    img_out = image_base.copy()
    draw = ImageDraw.Draw(img_out)

    taille_case = image_base.width // colonnes
    epaisseur = max(4, taille_case // 15)
    c1, c2 = int(domino_cible[0]), int(domino_cible[1])

    for p in placements:
        v1, v2 = int(p["valeurs"][0]), int(p["valeurs"][1])
        # Comparaison normalisée (ordre indépendant)
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
                width=epaisseur,
            )

    return img_out