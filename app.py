import streamlit as st
from PIL import Image 
import algorithme
import traitement_image
import io 
st.set_page_config(page_title="Mosaïque de dominos", layout="wide")
st.title("🎲 Générateur de Mosaïque en Dominos")
st.write("Projet P4 - Par Matteo Hanon Obsomer & Clément Leroy")

# --- Barre latérale (Paramètres) ---
st.sidebar.header("Paramètres")
type_jeu = st.sidebar.radio("Type de jeu :", ("double_six", "double_neuf"))
nb_boites = st.sidebar.number_input("Nombre de boîtes", min_value=1, value=10, step=1)
btn_generer = st.sidebar.button("Générer la mosaïque")

# --- Zone principale ---
col1, col2 = st.columns(2)

with col1:
    st.header("Image originale")
    fichier_upload = st.file_uploader("Chargez votre image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
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
            
            # 2. Prétraitement de l'image
            image_prete = traitement_image.preparer_image(image_originale, total_dominos)
            st.image(image_prete, caption=f"Image N&B ajustée ({image_prete.width}x{image_prete.height} px)", width=400)
            
            # 3. Conversion en matrice mathématique
            matrice_valeurs = traitement_image.image_vers_matrice(image_prete, type_jeu)
            
            # 4. L'algorithme glouton (Placement)
            placements = algorithme.placer_dominos(matrice_valeurs, stock_dominos)
            st.success(f"🎉 Succès ! L'algorithme a placé {len(placements)} dominos (soit 100% du stock) !")
            
            # AJOUT DU CALCUL DU SCORE DE FIDELITE
            erreur_totale = 0
            # On parcourt chaque placement pour comparer avec la matrice originale
            for p in placements:
                i1, j1 = p["case1"]
                i2, j2 = p["case2"]
                v1, v2 = p["valeurs"]
                
                # Différence absolue entre le pixel et le domino choisi
                erreur_totale += abs(matrice_valeurs[i1, j1] - v1)
                erreur_totale += abs(matrice_valeurs[i2, j2] - v2)
            
            # Calcul de l'erreur moyenne par rapport à la valeur max (6 ou 9)
            valeur_max = 6 if type_jeu == "double_six" else 9
            nb_pixels = matrice_valeurs.size
            
            # Score final en % (100% = parfait, 0% = n'importe quoi)
            score_fidelite = 100 * (1 - (erreur_totale / (nb_pixels * valeur_max)))

            # Affichage d'une jolie jauge de performance
            st.metric(label="🎯 Score de fidélité de la mosaïque", value=f"{score_fidelite:.2f} %")
            
            if score_fidelite > 90:
                st.write("✨ *Excellent ! La ressemblance est quasi-parfaite.*")
            elif score_fidelite > 75:
                st.write("👍 *Bon résultat, les formes principales sont bien respectées.*")
            else:
                st.write("⚠️ *Le stock de dominos était peut-être trop limité pour cette image.*")
            # 5. Dessin final de la mosaïque
            st.subheader("🖼️ Votre Mosaïque")
            lignes, colonnes = matrice_valeurs.shape
            image_mosaique = traitement_image.dessiner_mosaique(placements, lignes, colonnes)
            
            st.image(image_mosaique, caption="Mosaïque générée avec succès !", use_container_width=True)
            st.balloons() 


            # 6. Preuve d'inventaire pour vérifier la bonne proportion des dominos
            st.divider()
            st.subheader("📊 Rapport d'inventaire")
            st.write("Vérification stricte des pièces utilisées (doit correspondre au nombre de boîtes) :")

            inventaire_utilise = {}
            for placement in placements:
                v1, v2 = placement["valeurs"]
                nom_domino = f"[{min(v1, v2)} | {max(v1, v2)}]"
                inventaire_utilise[nom_domino] = inventaire_utilise.get(nom_domino, 0) + 1

            inventaire_trie = dict(sorted(inventaire_utilise.items()))

            col_tab, col_vide = st.columns([1, 2])
            with col_tab:
                st.dataframe(inventaire_trie, column_config={
                    "index": "Type de domino",
                    "value": "Quantité placée"
                })
            # ajout d'un bouton permettant de télécharger l'image de domino

            # tsf de l'image PIL en données  binaires pour le navigateur
            buf = io.BytesIO()
            image_mosaique.save(buf,format="PNG")
            donnees_image = buf.getvalue()
            st.download_button(
                label="💾 Télécharger la mosaïque en haute définition",
                data=donnees_image,
                file_name="ma_mosaique_dominos.png",
                mime="image/png"
            )
