import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import random
import math
import io
import pandas as pd
from collections import Counter

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

def generer_inventaire(nb_dominos_necessaires, type_jeu="double_six"):
    val_max = 6 if type_jeu == "double_six" else 9
    jeu_de_base = [(i, j) for i in range(val_max + 1) for j in range(i, val_max + 1)]
    taille_jeu = len(jeu_de_base)
    nb_jeux_complets = nb_dominos_necessaires // taille_jeu
    reste = nb_dominos_necessaires % taille_jeu

    inventaire_final = []
    for _ in range(nb_jeux_complets):
        inventaire_final.extend(jeu_de_base)
    if reste > 0:
        inventaire_final.extend(random.sample(jeu_de_base, reste))
        
    return inventaire_final

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

# ================
# 2. RENDU VISUEL 
# ================
def dessiner_demi_domino(draw, x_px, y_px, taille_case, valeur):
    # Fond blanc, bordure grise
    draw.rectangle([x_px, y_px, x_px + taille_case, y_px + taille_case], fill="white", outline="lightgray")
    if valeur == 0: return 

    r = max(1, taille_case // 12)
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

def creer_mosaique_finale(emplacements, placement_final, colonnes, lignes, taille_case=20):
    image_finale = Image.new("RGB", (colonnes * taille_case, lignes * taille_case), "white")
    draw = ImageDraw.Draw(image_finale)
    
    for i, ((x1_g, y1_g), (x2_g, y2_g)) in enumerate(emplacements):
        val1, val2 = placement_final[i]
        x1, y1 = x1_g * taille_case, y1_g * taille_case
        x2, y2 = x2_g * taille_case, y2_g * taille_case
        
        dessiner_demi_domino(draw, x1, y1, taille_case, val1)
        dessiner_demi_domino(draw, x2, y2, taille_case, val2)
        
        # Effacer la ligne de séparation centrale avec du blanc
        if y1_g == y2_g: draw.line([x2, y1 + 1, x2, y1 + taille_case - 1], fill="white", width=2)
        else:            draw.line([x1 + 1, y2, x1 + taille_case - 1, y2], fill="white", width=2)
            
    return image_finale


# ======================
# 3. INTERFACE STREAMLIT
# ======================
st.set_page_config(page_title="Mosaïque de dominos (V2)", layout="wide")
st.title("🎲 Générateur de Mosaïque en Dominos")
st.write("Projet P4 - Par Matteo Hanon Obsomer & Clément Leroy")

# --- Barre latérale ---
st.sidebar.header("Paramètres")
type_jeu = st.sidebar.radio("Type de jeu :", ("double_six", "double_neuf"))
largeur_grille = st.sidebar.slider("Largeur (en nombre de dominos)", min_value=60, max_value=120, value=80, step=10)
btn_generer = st.sidebar.button("Générer la mosaïque", type="primary")

col1, col2 = st.columns(2)

with col1:
    st.header("Image originale")
    fichier_upload = st.file_uploader("Chargez votre image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if fichier_upload is not None:
        image_originale = Image.open(fichier_upload)
        st.image(image_originale, caption="Image importée", width='stretch')

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
        taille_case = max(1, largeur_px // largeur_grille)
        
        # 2. Exécution du flux avec indicateurs visuels
        st.info(f"📐 Grille calculée : {largeur_grille}x{hauteur_grille} cases ({nb_dominos} dominos au total)")
        
        image_prete = transfo_image(image_originale, largeur_grille, hauteur_grille)
        matrice = valeurs_grille(image_prete, type_jeu)
        inventaire = generer_inventaire(nb_dominos, type_jeu)
        emplacements = generer_emplacements(largeur_grille, hauteur_grille)
        
        # Barre de progression passée à l'algorithme !
        my_bar = st.progress(0, text="Optimisation des pièces en cours...")
        placement = optimiser_placement_recuit(matrice, emplacements, inventaire, iterations=150000, st_progress_bar=my_bar)
        
        # Rendu final
        with st.spinner("Dessin des dominos..."):
            image_mosaique = creer_mosaique_finale(emplacements, placement, largeur_grille, hauteur_grille, taille_case)
            # Redimensionnement exact à la taille d'origine
            image_mosaique = image_mosaique.resize((largeur_px, hauteur_px), Image.Resampling.LANCZOS)
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

        