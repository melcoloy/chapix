import streamlit as st
from PIL import Image 
import io
import time
import algorithme
import hongrois_v1
import traitement_image
import app_v2
import base64
import streamlit.components.v1 as components

st.set_page_config(page_title="Mosaïque de dominos", layout="wide")
st.title("🎲 Générateur de Mosaïque en Dominos")
st.write("Projet P4 - Par Matteo Hanon Obsomer & Clément Leroy")

# --- Barre latérale (Paramètres) ---
st.sidebar.header("Paramètres")
type_jeu = st.sidebar.radio("Type de jeu :", ("double_six", "double_neuf"), key="radio_type_jeu_main")
nb_boites = st.sidebar.number_input("Nombre de boîtes", min_value=1, value=10, step=1)
largeur_grille = st.sidebar.slider("Largeur (en nombre de dominos)", min_value=60, max_value=160, step=10, key="slider_largeur")


# Choix de l'algorithme
choix_algo = st.sidebar.radio(
    "Choix de l'algorithme :", 
    ("Glouton (Rapide, par le centre)", "Hongrois (Lent, optimum mathématique)", "Méta-Heuristique (Aléatoire)","Hongrois (Matteo)"),key="radio_algo_main"
)

btn_generer = st.sidebar.button("Générer la mosaïque")

# --- Zone principale ---
col1, col2 = st.columns(2)

with col1:
    st.header("Image originale")
    
    # Choix de la source de l'image
    choix_source = st.radio("Source de l'image :", ["📁 Importer un fichier", "📸 Prendre une photo"], key="radio_source_main")
    
    fichier_upload = None
    
    if choix_source == "📁 Importer un fichier":
        fichier_upload = st.file_uploader("Chargez votre image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    else:
        # Streamlit gère nativement l'accès à la webcam !
        fichier_upload = st.camera_input("Prenez une photo avec votre webcam")
    
    if fichier_upload is not None:
        image_originale = Image.open(fichier_upload)
        st.image(image_originale, caption="Image importée", width=400)
        st.success("Image chargée avec succès ! Prête pour l'algo.")

with col2:
    st.header("Résultat")
    
    if fichier_upload is not None and btn_generer:
        with st.spinner("Calculs et assemblage en cours..."):
            # 1. Génération du stock
            stock_dominos = algorithme.generer_inventaire(type_jeu, nb_boites)
            total_dominos = len(stock_dominos)

            #Matteo
            largeur_px, hauteur_px = image_originale.size
            ratio = hauteur_px / largeur_px
            hauteur_grille = int(largeur_grille * ratio)
            nb_dominos = (largeur_grille * hauteur_grille) // 2
            image_fin = app_v2.transfo_image(image_originale, largeur_grille, hauteur_grille)
            matrice = app_v2.valeurs_grille(image_fin, type_jeu)
            inventaire = app_v2.generer_inventaire(nb_dominos, type_jeu,matrice)
            emplacements = app_v2.generer_emplacements(largeur_grille, hauteur_grille)
            
            # 2. Prétraitement de l'image
            
            image_prete = traitement_image.preparer_image(image_originale, total_dominos)
                
            st.image(image_prete, caption=f"Image N&B ajustée ({image_prete.width}x{image_prete.height} px)", width=400)
            
            # 3. Conversion en matrice mathématique
            matrice_valeurs = traitement_image.image_vers_matrice(image_prete, type_jeu)
            
            # 4. Lancement de l'algorithme choisi avec CHRONOMÈTRE
            heure_debut = time.time() 
            my_bar = st.progress(0, text="Optimisation des pièces en cours...")
            
            # --- NOUVEAU : On garde une trace de la matrice utilisée ---
            matrice_reference = matrice_valeurs 
            
            if choix_algo == "Glouton (Rapide, par le centre)":
                placements = algorithme.placer_dominos(matrice_valeurs, stock_dominos)
                
            elif choix_algo == "Méta-Heuristique (Aléatoire)":
                placements_bruts = app_v2.optimiser_placement_recuit(matrice, emplacements, inventaire, iterations=150000, st_progress_bar=my_bar)
                matrice_reference = matrice # On utilisera cette matrice pour le score !
                
            elif choix_algo == "Hongrois (Matteo)":
                placements_bruts = app_v2.optimiser_placement_hongrois(matrice, emplacements, inventaire, st_progress_bar=my_bar)
                matrice_reference = matrice # On utilisera cette matrice pour le score !
                
            else:
                placements = hongrois_v1.placer_dominos(matrice_valeurs, stock_dominos)
                
            heure_fin = time.time()
            temps_execution = heure_fin - heure_debut 
            
            # --- CORRECTION VITALE : Conversion des Tuples V2 en Dictionnaires universels ---
            if choix_algo in ["Méta-Heuristique (Aléatoire)", "Hongrois (Matteo)"]:
                placements = []
                for idx, (val1, val2) in enumerate(placements_bruts):
                    (x1, y1), (x2, y2) = emplacements[idx]
                    placements.append({
                        "case1": (y1, x1), # On inverse x et y pour correspondre à (ligne, colonne)
                        "case2": (y2, x2),
                        "valeurs": (val1, val2)
                    })
                    
            st.success(f"🎉 Succès ! L'algorithme a placé {len(placements)} dominos (soit 100% du stock) !")
            
            # --- CALCUL DU SCORE DE FIDÉLITÉ ---
            erreur_totale = 0
            for p in placements:
                i1, j1 = p["case1"]
                i2, j2 = p["case2"]
                v1, v2 = p["valeurs"]
                
                # On utilise matrice_reference (qui s'adapte à l'algo choisi) au lieu de matrice_valeurs
                erreur_totale += abs(matrice_reference[i1, j1] - v1)
                erreur_totale += abs(matrice_reference[i2, j2] - v2)
            
            valeur_max = 6 if type_jeu == "double_six" else 9
            nb_pixels = matrice_reference.size
            score_fidelite = 100 * (1 - (erreur_totale / (nb_pixels * valeur_max)))

            # --- AFFICHAGE DES MÉTRIQUES ---
            col_met1, col_met2 = st.columns(2)
            with col_met1:
                st.metric(label="🎯 Score de fidélité", value=f"{score_fidelite:.2f} %")
            with col_met2:
                st.metric(label="⏱️ Temps d'exécution", value=f"{temps_execution:.3f} s")
            
            if score_fidelite > 90:
                st.write("✨ *Excellent ! La ressemblance est quasi-parfaite.*")
            elif score_fidelite > 75:
                st.write("👍 *Bon résultat, les formes principales sont bien respectées.*")
            else:
                st.write("⚠️ *Le stock de dominos était peut-être trop limité pour cette image.*")
            
            # 5. Dessin final de la mosaïque
            st.subheader("🖼️ Votre Mosaïque")
            lignes, colonnes = matrice_reference.shape
            image_mosaique = traitement_image.dessiner_mosaique(placements, lignes, colonnes)
            
            st.image(image_mosaique, caption="Mosaïque générée avec succès !", use_container_width=True)
            st.balloons() 

            # 6. Preuve d'inventaire
            st.divider()
            st.subheader("📊 Rapport d'inventaire")
            st.write("Vérification stricte des pièces utilisées :")

            inventaire_utilise = {}
            for placement in placements:
                v1, v2 = placement["valeurs"]
                nom_domino = f"[{min(v1, v2)} | {max(v1, v2)}]"
                inventaire_utilise[nom_domino] = inventaire_utilise.get(nom_domino, 0) + 1

            inventaire_trie = dict(sorted(inventaire_utilise.items()))

            col_tab, col_vide = st.columns([1.5, 2])
            with col_tab:
                st.dataframe(inventaire_trie, column_config={
                    "index": "Type de domino",
                    "value": "Quantité placée"
                })

            # 7. Téléchargement personnalisé
            st.divider()
            st.subheader("💾 Téléchargement")
            
            nom_fichier = st.text_input("Nommez votre fichier :", value="ma_mosaique_dominos")
            
            if not nom_fichier.endswith(".png"):
                nom_fichier += ".png"

            buf = io.BytesIO()
            image_mosaique.save(buf, format="PNG")
            donnees_image = buf.getvalue()
            
            st.download_button(
                label=f"📥 Télécharger : {nom_fichier}",
                data=donnees_image,
                file_name=nom_fichier,
                mime="image/png"
            )

            # bouton d'impression (injection via html/js)
            st.divider()
            st.subheader("🖨️ Impression")
            st.write("Vous pouvez imprimer directement votre mosaïque depuis votre navigateur :")

            # encondage de l'image en texte  (base64) pour pouvoir l'envoyer au navigateur html
            b64_image = base64.b64encode(donnees_image).decode()

            # code html et JavaScript du bouton d'impression
            html_bouton = f"""
            <div style="text-align: left;">
                <button onclick="
                    var w = window.open('');
                    w.document.write('<html><head><title>Impression Mosaique</title></head><body style=\\'margin:0;display:flex;justify-content:center;align-items:center;height:100vh;\\'><img src=\\'data:image/png;base64,{b64_image}\\' style=\\'max-width:100%;max-height:100%;\\'></body></html>');
                    w.document.close();
                    w.focus();
                    setTimeout(function() {{ w.print(); w.close(); }}, 500);
                " style="background-color: #ffffff; color: #31333F; padding: 10px 24px; border: 1px solid #dcdcdc; border-radius: 8px; cursor: pointer; font-size: 16px; font-family: sans-serif; transition: 0.3s;"
                onmouseover="this.style.borderColor='#FF4B4B'; this.style.color='#FF4B4B';"
                onmouseout="this.style.borderColor='#dcdcdc'; this.style.color='#31333F';">
                    🖨️ Lancer l'impression
                </button>
            </div>
            """
            # affichage du bouton dans Streamlit
            components.html(html_bouton, height=60)