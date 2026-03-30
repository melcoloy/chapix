import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageChops
import random
import math
import io
import pandas as pd
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. FONCTIONS MATHÉMATIQUES ET ALGORITHMES
# ==========================================
def transfo_image(image, colonnes, lignes):
    image = image.convert("L")
    image = ImageOps.fit(image, (colonnes, lignes), method=Image.Resampling.LANCZOS)
    return image

def valeurs_grille(image, type_jeu="double_six"):
    val_max = 6 if type_jeu == "double_six" else 9
    matrice_pixels = np.array(image)
    # Inversion pour dominos blancs (pixel blanc 255 -> 0 point / pixel noir 0 -> 6 points)
    matrice_valeurs = np.round((matrice_pixels / 255.0) * val_max).astype(int)
    matrice_valeurs = val_max - matrice_valeurs 
    return matrice_valeurs

def accentuer_contours(image_pil):
    """Dessine des traits noirs sur les contours pour détacher le sujet du fond."""
    img_gray = image_pil.convert("L")
    edges = img_gray.filter(ImageFilter.FIND_EDGES)
    
    # On nettoie pour ne garder que les lignes franches (seuil de tolérance > 30)
    edges = edges.point(lambda p: 255 if p > 30 else 0)
    
    # Inversion : on veut des lignes noires sur fond blanc
    edges_inv = ImageOps.invert(edges)
    
    # Produit : On superpose les lignes noires sur l'image d'origine
    img_finale = ImageChops.multiply(img_gray, edges_inv)
    return img_finale

def generer_inventaire(nb_dominos_necessaires, type_jeu="double_six", matrice_cibles=None):
    val_max = 6 if type_jeu == "double_six" else 9
    jeu_de_base = [(i, j) for i in range(val_max + 1) for j in range(i, val_max + 1)]
    taille_jeu = len(jeu_de_base)
    nb_jeux_complets = nb_dominos_necessaires // taille_jeu
    reste = nb_dominos_necessaires % taille_jeu

    inventaire_final = []
    for _ in range(nb_jeux_complets):
        inventaire_final.extend(jeu_de_base)
        
    if reste > 0:
        if matrice_cibles is not None:
            # --- MÉTHODE INTELLIGENTE ---
            # 1. Compter combien de fois chaque valeur (0 à 6) est demandée par l'image
            valeurs, clics = np.unique(matrice_cibles, return_counts=True)
            frequences = dict(zip(valeurs, clics))
            
            # 2. Donner un "score d'utilité" à chaque domino
            dominos_scores = []
            for domino in jeu_de_base:
                score = frequences.get(domino[0], 0) + frequences.get(domino[1], 0)
                dominos_scores.append((score, domino))
                
            # 3. Trier du plus utile au moins utile
            dominos_scores.sort(key=lambda x: x[0], reverse=True)
            
            # 4. Prendre exactement les pièces manquantes dont on a le plus besoin !
            pieces_restantes = [domino for score, domino in dominos_scores[:reste]]
            inventaire_final.extend(pieces_restantes)
        else:
            # Sécurité : au hasard si aucune image n'est fournie
            inventaire_final.extend(random.sample(jeu_de_base, reste))
            
    return inventaire_final

def calculer_largeur_ideale(image_pil):
    """
    Analyse la complexité de l'image (densité des contours) 
    pour recommander un nombre de dominos en largeur.
    """
    # 1. On passe l'image en niveaux de gris et on détecte les contours
    image_bords = image_pil.convert("L").filter(ImageFilter.FIND_EDGES)
    matrice_bords = np.array(image_bords)
    
    # 2. On calcule le pourcentage de pixels "blancs" (qui sont des contours)
    densite_contours = np.sum(matrice_bords) / (matrice_bords.shape[0] * matrice_bords.shape[1] * 255)
    
    # 3. Étalonnage mathématique :
    # Si densité très faible (logo) -> ~50 dominos
    # Si densité moyenne (portrait) -> ~80 dominos
    # Si densité forte (paysage) -> ~120 dominos
    largeur_estimee = int(40 + (densite_contours * 800))
    
    # On borne le résultat pour rester raisonnable
    return max(40, min(150, largeur_estimee))

def generer_emplacements(largeur_grille, hauteur_grille):
    grille_occupee = [[False] * largeur_grille for _ in range(hauteur_grille)]
    emplacements = []
    
    for y in range(hauteur_grille):
        for x in range(largeur_grille):
            if grille_occupee[y][x]: continue
                
            if x + 1 < largeur_grille and not grille_occupee[y][x + 1]:
                emplacements.append(((x, y), (x + 1, y)))
                grille_occupee[y][x] = grille_occupee[y][x + 1] = True
            elif y + 1 < hauteur_grille and not grille_occupee[y + 1][x]:
                emplacements.append(((x, y), (x, y + 1)))
                grille_occupee[y][x] = grille_occupee[y + 1][x] = True
    return emplacements

def calculer_erreur(domino, cible1, cible2):
    err_norm = abs(domino[0] - cible1) + abs(domino[1] - cible2)
    err_inv = abs(domino[1] - cible1) + abs(domino[0] - cible2)
    return (err_norm, domino) if err_norm <= err_inv else (err_inv, (domino[1], domino[0]))

def optimiser_placement_recuit(cibles, emplacements, inventaire, iterations=1e7, st_progress_bar=None):
    random.shuffle(inventaire)
    placement_actuel = list(inventaire)
    
    temp_initiale = 10.0
    temp_finale = 0.01
    alpha = (temp_finale / temp_initiale) ** (1 / iterations)
    temperature = temp_initiale
    
    # Mise à jour de la barre de progression tous les 5% pour ne pas figer Streamlit
    step_progress = iterations // 20 
    
    for i in range(iterations):
        idx1, idx2 = random.sample(range(len(emplacements)), 2)
        (x1A, y1A), (x1B, y1B) = emplacements[idx1]
        (x2A, y2A), (x2B, y2B) = emplacements[idx2]
        
        c1A, c1B = cibles[y1A, x1A], cibles[y1B, x1B]
        c2A, c2B = cibles[y2A, x2A], cibles[y2B, x2B]
        
        err1_av, _ = calculer_erreur(placement_actuel[idx1], c1A, c1B)
        err2_av, _ = calculer_erreur(placement_actuel[idx2], c2A, c2B)
        err1_ap, _ = calculer_erreur(placement_actuel[idx2], c1A, c1B)
        err2_ap, _ = calculer_erreur(placement_actuel[idx1], c2A, c2B)
        
        delta_erreur = (err1_ap + err2_ap) - (err1_av + err2_av)
        
        if delta_erreur < 0 or random.random() < math.exp(-delta_erreur / temperature):
            placement_actuel[idx1], placement_actuel[idx2] = placement_actuel[idx2], placement_actuel[idx1]
            
        temperature *= alpha
        
        if st_progress_bar and i % step_progress == 0:
            st_progress_bar.progress(i / iterations, text="Optimisation des pièces en cours...")

    # Assigner le bon sens final
    placement_final = [calculer_erreur(dom, cibles[empl[0][1], empl[0][0]], cibles[empl[1][1], empl[1][0]])[1] 
                       for dom, empl in zip(placement_actuel, emplacements)]
    
    if st_progress_bar: st_progress_bar.empty()
    return placement_final

def optimiser_placement_hongrois(cibles, emplacements, inventaire, st_progress_bar=None):
    """
    Trouve l'affectation 100% optimale en utilisant la méthode hongroise (Linear Sum Assignment).
    """
    nb_emplacements = len(emplacements)
    
    if st_progress_bar:
        st_progress_bar.progress(0.1, text="Étape 1 : Calcul de la matrice des coûts (Peut prendre quelques secondes)...")
        
    # 1. Création de la matrice des coûts (Lignes = Emplacements, Colonnes = Dominos)
    matrice_couts = np.zeros((nb_emplacements, nb_emplacements), dtype=int)
    
    # On pré-calcule les cibles pour aller plus vite dans la boucle
    valeurs_cibles = [(cibles[y1, x1], cibles[y2, x2]) for ((x1, y1), (x2, y2)) in emplacements]
    
    for i, (cible1, cible2) in enumerate(valeurs_cibles):
        for j, domino in enumerate(inventaire):
            # On calcule l'erreur dans les deux sens et on garde la meilleure
            err_norm = abs(domino[0] - cible1) + abs(domino[1] - cible2)
            err_inv = abs(domino[1] - cible1) + abs(domino[0] - cible2)
            matrice_couts[i, j] = err_norm if err_norm < err_inv else err_inv

    if st_progress_bar:
        st_progress_bar.progress(0.5, text="Étape 2/2 : Résolution mathématique exacte...")

    # 2. Résolution du problème d'affectation
    row_ind, col_ind = linear_sum_assignment(matrice_couts)
    
    # 3. Construction du placement final dans le bon sens
    placement_final = []
    for i, j in enumerate(col_ind): # i = index emplacement, j = index du domino choisi
        domino = inventaire[j]
        cible1, cible2 = valeurs_cibles[i]
        
        err_norm = abs(domino[0] - cible1) + abs(domino[1] - cible2)
        err_inv = abs(domino[1] - cible1) + abs(domino[0] - cible2)
        
        if err_norm <= err_inv:
            placement_final.append(domino)
        else:
            placement_final.append((domino[1], domino[0]))
            
    if st_progress_bar: 
        st_progress_bar.empty()
        
    return placement_final

# ================
# 2. RENDU VISUEL 
# ================
def dessiner_demi_domino(draw, x_px, y_px, taille_case, valeur, epaisseur_bordure):
    # Fond blanc, bordure grise adaptative
    draw.rectangle(
        [x_px, y_px, x_px + taille_case, y_px + taille_case], 
        fill="white", 
        outline="darkgray", 
        width=epaisseur_bordure
    )
    if valeur == 0: return 

    r = max(1, taille_case // 6) # Points plus gros et bien ronds
    c, g, d, h, b, m = 0.5, 0.25, 0.75, 0.25, 0.75, 0.5  
    positions_points = {
        1: [(c, c)], 2: [(g, h), (d, b)], 3: [(g, h), (c, c), (d, b)],
        4: [(g, h), (d, h), (g, b), (d, b)], 5: [(g, h), (d, h), (c, c), (g, b), (d, b)],
        6: [(g, h), (d, h), (g, m), (d, m), (g, b), (d, b)],
        7: [(g, h), (d, h), (g, m), (d, m), (g, b), (d, b), (c, c)],
        8: [(g, h), (d, h), (g, m), (d, m), (g, b), (d, b), (c, h), (c, b)],
        9: [(g, h), (d, h), (g, m), (d, m), (g, b), (d, b), (c, h), (c, b), (c, c)]
    }
    
    for pos_x, pos_y in positions_points.get(valeur, []):
        cx, cy = x_px + pos_x * taille_case, y_px + pos_y * taille_case
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="black")

def creer_mosaique_finale(emplacements, placement_final, colonnes, lignes, taille_case=80):
    image_finale = Image.new("RGB", (colonnes * taille_case, lignes * taille_case), "white")
    draw = ImageDraw.Draw(image_finale)
    
    # Épaisseurs intelligentes qui s'adaptent à la HD
    epaisseur_bordure = max(2, taille_case // 20)
    epaisseur_gomme = (epaisseur_bordure * 2) + 2
    
    for i, ((x1_g, y1_g), (x2_g, y2_g)) in enumerate(emplacements):
        val1, val2 = placement_final[i]
        x1, y1 = x1_g * taille_case, y1_g * taille_case
        x2, y2 = x2_g * taille_case, y2_g * taille_case
        
        dessiner_demi_domino(draw, x1, y1, taille_case, val1, epaisseur_bordure)
        dessiner_demi_domino(draw, x2, y2, taille_case, val2, epaisseur_bordure)
        
        # Gommer la séparation interne sans écraser les points
        if y1_g == y2_g: 
            draw.line([x2, y1 + epaisseur_bordure, x2, y1 + taille_case - epaisseur_bordure], fill="white", width=epaisseur_gomme)
        else:            
            draw.line([x1 + epaisseur_bordure, y2, x1 + taille_case - epaisseur_bordure, y2], fill="white", width=epaisseur_gomme)
            
    return image_finale

# ======================
# 3. INTERFACE STREAMLIT
# ======================
st.set_page_config(page_title="Mosaïque de dominos", layout="wide")
st.title("🎲 Générateur de Mosaïque en Dominos")
st.write("Projet P4 - Par Matteo Hanon Obsomer & Clément Leroy")

col1, col2 = st.columns(2)

with col1:
    st.header("Image originale")
    fichier_upload = st.file_uploader("Chargez votre image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if fichier_upload is not None:
        image_originale = Image.open(fichier_upload)
        st.image(image_originale, caption="Image importée", width='stretch')

# --- Barre latérale ---
st.sidebar.header("Paramètres")
type_jeu = st.sidebar.radio("Type de jeu :", ("double_six", "double_neuf"))
if st.sidebar.button("Calculer la largeur optimale"):
    if fichier_upload is not None:
        # 1. On récupère la valeur recommandée par ton algorithme
        largeur = calculer_largeur_ideale(image_originale)
        
        # 2. On l'arrondit à la dizaine la plus proche
        largeur_arrondie = round(largeur / 10) * 10
        
        # 3. On contraint la valeur entre 80 et 160
        st.session_state.slider_largeur = max(60, min(160, largeur_arrondie))
    else:
        st.sidebar.warning("Chargez d'abord une image au centre !")
largeur_grille = st.sidebar.slider("Largeur (en nombre de dominos)", min_value=60, max_value=160, step=10, key="slider_largeur")
contour_fort = st.sidebar.checkbox("Accentuer les contours")
# Choix de l'algorithme d'optimisation
methode_calcul = st.sidebar.radio("Algorithme de placement :", ("Recuit Simulé", "Méthode Hongroise"))

btn_generer = st.sidebar.button("Générer la mosaïque", type="primary")

with col2:
    st.header("Résultat")
    if fichier_upload is not None and btn_generer:
        # 1. Calcul des dimensions exactes
        largeur_px, hauteur_px = image_originale.size
        ratio = hauteur_px / largeur_px
        hauteur_grille = int(largeur_grille * ratio)
        
        # Sécurité pour le pavagement : l'aire totale doit être paire
        if (largeur_grille * hauteur_grille) % 2 != 0:
            hauteur_grille += 1
            
        nb_dominos = (largeur_grille * hauteur_grille) // 2
        
        # 2. Exécution du flux avec indicateurs visuels
        st.info(f"📐 Grille calculée : {largeur_grille}x{hauteur_grille} cases ({nb_dominos} dominos au total)")
        if contour_fort:
            image_prete = accentuer_contours(image_originale)
        else:
            image_prete = image_originale
        image_prete = transfo_image(image_prete, largeur_grille, hauteur_grille)
        matrice = valeurs_grille(image_prete, type_jeu)
        inventaire = generer_inventaire(nb_dominos, type_jeu,matrice)
        emplacements = generer_emplacements(largeur_grille, hauteur_grille)
        
        # Barre de progression passée à l'algorithme !
        my_bar = st.progress(0, text="Optimisation des pièces en cours...")
        if "Hongroise" in methode_calcul:
            placement = optimiser_placement_hongrois(matrice, emplacements, inventaire, st_progress_bar=my_bar)
        else:
            placement = optimiser_placement_recuit(matrice, emplacements, inventaire, iterations=150000, st_progress_bar=my_bar)
        
        # Rendu final
        with st.spinner("Dessin des dominos..."):
            image_mosaique = creer_mosaique_finale(emplacements, placement, largeur_grille, hauteur_grille, taille_case=80)
            
        # 5. Téléchargement
        buf = io.BytesIO()
        image_mosaique.save(buf, format="PNG")
        st.download_button(
            label="Télécharger",
            data=buf.getvalue(),
            file_name="mosaique_dominos.png",
            mime="image/png"
        )    
        st.image(image_mosaique, caption="Mosaïque générée avec succès !", width='stretch')
        st.balloons()

        # 3. Calcul du score de fidélité
        erreur_totale = 0
        for idx, (case1, case2) in enumerate(emplacements):
            v1, v2 = placement[idx]
            erreur_totale += abs(matrice[case1[1], case1[0]] - v1) + abs(matrice[case2[1], case2[0]] - v2)
            
        valeur_max = 6 if type_jeu == "double_six" else 9
        score_fidelite = 100 * (1 - (erreur_totale / (matrice.size * valeur_max)))
        
        st.metric(label="🎯 Score de fidélité de la mosaïque", value=f"{score_fidelite:.2f} %")

        # 4. Rapport d'inventaire
        st.divider()
        st.subheader("📊 Rapport d'inventaire")
        st.write(f"Stock généré pour couvrir les {nb_dominos} emplacements.")
        
        inventaire_utilise = {}
        for val1, val2 in placement:
            nom_domino = f"[{min(val1, val2)} | {max(val1, val2)}]"
            inventaire_utilise[nom_domino] = inventaire_utilise.get(nom_domino, 0) + 1
            
        df_inventaire = pd.DataFrame(list(inventaire_utilise.items()), columns=["Type de domino", "Quantité placée"])
        df_inventaire = df_inventaire.sort_values(by="Type de domino").reset_index(drop=True)
        st.dataframe(df_inventaire, width='stretch')

        