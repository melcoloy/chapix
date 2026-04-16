#Interface streamlit

import io
import base64
import math
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageEnhance

from core.inventaire import completer_inventaire, valeur_max
from core.image import preparer_image, image_vers_matrice, dessiner_mosaique, mettre_en_evidence
from core.algorithmes import glouton, hongrois, recuit, calculer_score, LIMITE_HONGROIS

ALGOS = {
    "Glouton (Par le centre)":       "glouton",
    "Méta-Heuristique (Très rapide)":      "recuit",
    "Hongrois (Lent, optimum mathématique)": "hongrois",
}

expli_algos = """
- **Glouton :** Rapidité moyenne, priorise le centre de l'image.
- **Méta-heuristique :** Très rapide, idéal avec un grand nombre de boîtes.
- **Hongrois :** Très lent, solution mathématique optimale (maximum 235 boîtes double-six ou 120 boîtes double-neuf).
"""

# ─────────────────────────────────────────────────────────────────────
# Mise en page
# ─────────────────────────────────────────────────────────────────────

st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem; 
        }
    </style>
""", unsafe_allow_html=True) #Permet de remonter le titre pour pas laisser un espace vide trop grand

st.set_page_config(page_title="Mosaïque de dominos", layout="wide")
st.title("🎲 Générateur de Mosaïque en Dominos")
st.write("Par Matteo Hanon Obsomer & Clément Leroy")

# ── Barre latérale ────────────────────────────────────────────────────
st.sidebar.header("Paramètres")
#Choix des paramètres par l'utilisateur
type_jeu       = st.sidebar.radio("Type de jeu :", ("double_six", "double_neuf"), key="widget_type_jeu")
nb_boites      = st.sidebar.number_input("Nombre de boîtes disponibles :", min_value=1, value=80, step=10)
choix_algo     = st.sidebar.radio("Algorithme :", list(ALGOS.keys()), key="widget_algo", help=expli_algos)
contraste = st.sidebar.slider("Contraste :", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
activer_contours  = st.sidebar.checkbox("Segmentation des contours")
btn_generer    = st.sidebar.button("Générer la mosaïque")

# ── Colonnes principales ──────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.header("Image originale")
    source = st.radio("Source :", ["📁 Importer un fichier", "📸 Webcam"], key="widget_source")
    fichier = (
        st.file_uploader("Image (JPG, PNG)", type=["jpg", "jpeg", "png"])
        if source == "📁 Importer un fichier"
        else st.camera_input("Prendre une photo")
    )
    if fichier:
        # Ouverture de l'image
        bytes_data = fichier.getvalue()
        image_originale = Image.open(io.BytesIO(bytes_data))
        image_originale.load()
        if contraste != 1.0:
            image_originale = ImageEnhance.Contrast(image_originale).enhance(contraste)
        st.image(image_originale, caption="Image importée", width=400)

        #Calcul du nombre de dominos et des dimensions de la grille
        # 1. Calcul du stock maximum de cases
        taille_boite = 28 if type_jeu == "double_six" else 55
        stock_max_dominos = nb_boites * taille_boite
        cases_max_dispo = stock_max_dominos * 2
                
        # 2. Analyse des proportions de l'image
        largeur_px, hauteur_px = image_originale.size
        ratio = hauteur_px / largeur_px
                
        # Calcul de la taille maximale possible
        # On sait que : (largeur * (largeur * ratio)) ne doit pas dépasser cases_max_dispo
        largeur_grille = int(math.sqrt(cases_max_dispo / ratio))
        hauteur_grille = int(largeur_grille * ratio)
                
        # 4. Sécurité mathématique : forcer un nombre pair de cases
        if (largeur_grille * hauteur_grille) % 2 != 0:
            hauteur_grille -= 1
                    
        nb_dominos = (largeur_grille * hauteur_grille) // 2        
        
        # Aperçu en temps réel
        st.divider()
        st.subheader("👁️ Aperçu de la grille N&B")
        image_prete = preparer_image(image_originale, largeur_grille, hauteur_grille, activer_contours)
        st.image(image_prete, caption=f"Dimensions : {image_prete.width} × {image_prete.height} cases, soit {nb_dominos} dominos", width=400)
        st.info("💡 Modifiez le nombre de boîtes, le contraste ou la case 'Contours' à gauche pour ajuster cet aperçu en temps réel. Lancez la génération quand le rendu vous plaît !")

with col2:
    st.header("Résultat")

    # ______ Calcul (uniquement au clic "Générer") ____________________
    if fichier and btn_generer:
        try:
            with st.spinner("Calculs en cours..."):

                #Création des données utiles aux fonctions d'optimisation
                vmax  = valeur_max(type_jeu)
                matrice_valeurs = image_vers_matrice(image_prete, type_jeu)
                inventaire = completer_inventaire(nb_dominos, type_jeu, matrice_valeurs)

                # Garde-fou Hongrois
                if ALGOS[choix_algo] == "hongrois" and nb_dominos > LIMITE_HONGROIS:
                    st.error(
                        f"⚠️ Grille trop grande pour l'algorithme Hongrois "
                        f"({nb_dominos} emplacements > limite {LIMITE_HONGROIS}). "
                        "Réduisez le nombre de boîtes ou utilisez un autre algorithme."
                    )
                    st.stop()

                # Lancement
                my_bar = st.progress(0, text="Optimisation en cours...")

                def progress(ratio, texte):
                    my_bar.progress(ratio, text=texte)

                debut = time.time()

                if ALGOS[choix_algo] == "glouton":
                    placements = glouton(matrice_valeurs, inventaire, progress_callback=progress)
                elif ALGOS[choix_algo] == "hongrois":
                    placements = hongrois(matrice_valeurs, inventaire, progress_callback=progress)
                else:  # recuit
                    placements = recuit(matrice_valeurs, inventaire, progress_callback=progress)

                temps = time.time() - debut

                # Dessin de la mosaïque de base (sans mise en évidence)
                progress(0.95, "Dessin de la mosaïque")
                image_mosaique = dessiner_mosaique(placements, hauteur_grille, largeur_grille)

                my_bar.empty()

                # Inventaire utilisé
                inventaire_utilise = {}
                for p in placements:
                    v1, v2 = p["valeurs"]
                    cle = f"[{min(v1, v2)} | {max(v1, v2)}]"
                    inventaire_utilise[cle] = inventaire_utilise.get(cle, 0) + 1

                # Sauvegarde session
                st.session_state.update({
                    "placements":        placements,
                    "matrice_reference": matrice_valeurs,
                    "image_mosaique":    image_mosaique,
                    "inventaire":        dict(sorted(inventaire_utilise.items())),
                    "colonnes":          largeur_grille,
                    "temps":             temps,
                    "type_jeu":          type_jeu,
                    "vmax":              vmax,
                })

        except ValueError as e:
            st.error(f"❌ Paramètre invalide : {e}")
            st.stop()
        except MemoryError:
            st.error("❌ Mémoire insuffisante. Réduisez la taille de la grille.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Erreur inattendue : {e}")
            st.stop()

    # ______ Affichage et Analyse des résultats ____________________
    if fichier and "placements" in st.session_state:
        placements     = st.session_state["placements"]
        matrice_ref    = st.session_state["matrice_reference"]
        image_mosaique = st.session_state["image_mosaique"]
        inventaire     = st.session_state["inventaire"]
        colonnes_g     = st.session_state["colonnes"]
        temps          = st.session_state["temps"]
        vmax           = st.session_state["vmax"]
        type_jeu_s     = st.session_state["type_jeu"]

        st.success(f"🎉 {len(placements)} dominos placés !")

        # Inspecteur de dominos
        col_titre, col_boutons = st.columns([3.3, 1])
        
        with col_titre:
            st.subheader("🔍 Inspecteur de dominos")
        
        # Impression et téléchargement
        # 1. On prépare l'image
        buf = io.BytesIO()
        image_mosaique.save(buf, format="PNG")
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 2. On crée deux petits boutons carrés alignés à droite
        html_boutons = f"""
        <div style="display: flex; gap: 10px; justify-content: flex-end; align-items: center; margin-top: 15px;">
            <a href="data:image/png;base64,{b64_image}" download="mosaique_dominos.png" title="Télécharger l'image"
               style="display: flex; align-items: center; justify-content: center; width: 35px; height: 35px; background: transparent; border: 1px solid #dcdcdc; border-radius: 8px; text-decoration: none; font-size: 20px; transition: 0.2s;"
               onmouseover="this.style.borderColor='#FF4B4B'; this.style.backgroundColor='#FFF0F0';"
               onmouseout="this.style.borderColor='#dcdcdc'; this.style.backgroundColor='transparent';">
               💾
            </a>
            
            <button title="Imprimer l'image" onclick="
                var w = window.open('');
                w.document.write('<html><head><title>Impression</title></head><body style=\\'margin:0;display:flex;justify-content:center;align-items:center;height:100vh;\\'><img src=\\'data:image/png;base64,{b64_image}\\' style=\\'max-width:100%;max-height:100%;\\'></body></html>');
                w.document.close();
                w.focus();
                setTimeout(function() {{ w.print(); w.close(); }}, 500);
            " style="display: flex; align-items: center; justify-content: center; width: 35px; height: 35px; background: transparent; border: 1px solid #dcdcdc; border-radius: 8px; cursor: pointer; font-size: 20px; padding: 0; transition: 0.2s;"
               onmouseover="this.style.borderColor='#FF4B4B'; this.style.backgroundColor='#FFF0F0';"
               onmouseout="this.style.borderColor='#dcdcdc'; this.style.backgroundColor='transparent';">
               🖨️
            </button>
        </div>
        """
        with col_boutons:
            components.html(html_boutons, height=55)

        # Choix du domino à rechercher
        liste_options = ["Afficher l'image normale"] + list(inventaire.keys())
        choix_domino  = st.selectbox("Mettre en évidence un type de domino :", liste_options)

        if choix_domino == "Afficher l'image normale":
            image_a_afficher = image_mosaique
        else:
            valeurs  = choix_domino.replace("[", "").replace("]", "").split("|")
            d1, d2   = int(valeurs[0].strip()), int(valeurs[1].strip())
            image_a_afficher = mettre_en_evidence(image_mosaique, placements, colonnes_g, (d1, d2))

        # Affichage de l'image avec les dominos mis en évidence
        st.image(image_a_afficher, caption="Mosaïque", width='stretch')

        # Métriques
        score = calculer_score(placements, matrice_ref, vmax)
        c1, c2 = st.columns(2)
        c1.metric("🎯 Fidélité", f"{score:.2f} %")
        c2.metric("⏱️ Durée", f"{temps:.3f} s")

        st.divider()


        # Rapport d'inventaire
        st.subheader("📊 Rapport d'inventaire")

        col_tab, _ = st.columns([1.5, 2])
        with col_tab:
            df_inv = pd.DataFrame(
                list(inventaire.items()), columns=["Domino", "Quantité"]
            ).set_index("Domino")
            st.dataframe(df_inv)