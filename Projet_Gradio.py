import gradio as gr
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import random
import math

def transfo_image(image,colonnes,lignes):
    image = image.convert("L")
    image = ImageOps.fit(image, (colonnes, lignes), method=Image.Resampling.LANCZOS)
    return image

def valeurs_grille(image, type_jeu="double-six"):
    val_max = 6 if type_jeu == "double-six" else 9
    matrice_pixels = np.array(image)
    matrice_valeurs = np.round((matrice_pixels / 255.0) * val_max).astype(int)
    matrice_valeurs = val_max - matrice_valeurs
    return matrice_valeurs

def generer_inventaire(nb_dominos_necessaires, type_jeu="double-six"):
    val_max = 6 if type_jeu == "double-six" else 9
    jeu_de_base = []
    
    for i in range(val_max + 1):
        for j in range(i, val_max + 1):
            jeu_de_base.append((i, j))
            
    taille_jeu = len(jeu_de_base)
    nb_jeux_complets = nb_dominos_necessaires // taille_jeu
    reste = nb_dominos_necessaires % taille_jeu

    inventaire_final = []
    for _ in range(nb_jeux_complets):
        inventaire_final.extend(jeu_de_base)
        
    if reste > 0:
        pieces_restantes = random.sample(jeu_de_base, reste)
        inventaire_final.extend(pieces_restantes)
        
    return inventaire_final

def generer_emplacements(largeur_grille, hauteur_grille):
    # 1. Création d'une grille mémoire pour savoir quelles cases sont déjà occupées
    # False = vide, True = occupé
    grille_occupee = [[False for _ in range(largeur_grille)] for _ in range(hauteur_grille)]
    
    emplacements = []
    
    # 2. Parcours de la grille (de gauche à droite, puis de haut en bas)
    for y in range(hauteur_grille):
        for x in range(largeur_grille):
            
            # Si la case est déjà prise par un précédent domino, on passe
            if grille_occupee[y][x]:
                continue
                
            # 3. Tentative de placement horizontal (vers la droite)
            # On vérifie qu'on ne sort pas de l'image ET que la case de droite est libre
            if x + 1 < largeur_grille and not grille_occupee[y][x + 1]:
                emplacements.append(((x, y), (x + 1, y)))
                grille_occupee[y][x] = True
                grille_occupee[y][x + 1] = True
                
            # 4. Tentative de placement vertical (vers le bas)
            # Si on ne peut pas aller à droite (bord de l'image), on va en bas
            elif y + 1 < hauteur_grille and not grille_occupee[y + 1][x]:
                emplacements.append(((x, y), (x, y + 1)))
                grille_occupee[y][x] = True
                grille_occupee[y + 1][x] = True
                
            else:
                # Sécurité : Si on arrive ici, c'est que la grille est impaire et invalide
                raise ValueError(f"Impossible de paver la case ({x}, {y}). La grille est mal dimensionnée.")

    return emplacements

def calculer_erreur(domino, cible1, cible2):
    """
    Calcule la différence entre les points du domino et l'image.
    Teste les deux sens du domino (ex: [1|4] et [4|1]) pour trouver le meilleur.
    """
    erreur_sens_normal = abs(domino[0] - cible1) + abs(domino[1] - cible2)
    erreur_sens_inverse = abs(domino[1] - cible1) + abs(domino[0] - cible2)
    if erreur_sens_normal <= erreur_sens_inverse:
        return erreur_sens_normal, domino
    else:
        return erreur_sens_inverse, (domino[1], domino[0])

def optimiser_placement_recuit(cibles, emplacements, inventaire, iterations=10000000):
    """
    Algorithme de recuit simulé pour placer les dominos.
    
    - cibles : La matrice NumPy des valeurs idéales (0 à 6)
    - emplacements : Liste des coordonnées des paires de cases, ex: [((0,0), (0,1)), ...]
    - inventaire : Liste des dominos disponibles, ex: [(0,0), (0,1), ..., (6,6)]
    """
    # 1. Placement initial aléatoire
    random.shuffle(inventaire)
    placement_actuel = list(inventaire) # Copie de l'inventaire assignée aux emplacements
    
    # 2. Paramètres du recuit simulé
    temp_initiale = 10.0
    temp_finale = 0.01
    # Calcul du facteur de refroidissement pour atteindre temp_finale à la dernière itération
    alpha = (temp_finale / temp_initiale) ** (1 / iterations)
    temperature = temp_initiale
    
    print("Début de l'optimisation...")
    
    # 3. Boucle principale
    for i in range(iterations):
        # Choisir deux dominos au hasard à échanger
        idx1, idx2 = random.sample(range(len(emplacements)), 2)
        
        # Récupérer les coordonnées des 4 cases concernées
        case1_A, case1_B = emplacements[idx1]
        case2_A, case2_B = emplacements[idx2]
        
        # Récupérer les valeurs idéales de ces 4 cases sur l'image
        cible1_A, cible1_B = cibles[case1_A[1], case1_A[0]], cibles[case1_B[1], case1_B[0]]
        cible2_A, cible2_B = cibles[case2_A[1], case2_A[0]], cibles[case2_B[1], case2_B[0]]
        
        # Calculer l'erreur AVANT l'échange
        err1_avant, _ = calculer_erreur(placement_actuel[idx1], cible1_A, cible1_B)
        err2_avant, _ = calculer_erreur(placement_actuel[idx2], cible2_A, cible2_B)
        erreur_avant = err1_avant + err2_avant
        
        # Calculer l'erreur APRÈS l'échange simulé
        err1_apres, _ = calculer_erreur(placement_actuel[idx2], cible1_A, cible1_B)
        err2_apres, _ = calculer_erreur(placement_actuel[idx1], cible2_A, cible2_B)
        erreur_apres = err1_apres + err2_apres
        
        # 4. Décision : Accepte-t-on l'échange ?
        delta_erreur = erreur_apres - erreur_avant
        
        accepter = False
        if delta_erreur < 0:
            accepter = True # L'image est meilleure, on accepte toujours
        else:
            # L'image est pire, on accepte avec une certaine probabilité mathématique
            probabilite = math.exp(-delta_erreur / temperature)
            if random.random() < probabilite:
                accepter = True
                
        # 5. Application de l'échange
        if accepter:
            placement_actuel[idx1], placement_actuel[idx2] = placement_actuel[idx2], placement_actuel[idx1]
            
        # 6. Refroidissement
        temperature *= alpha
        
        # Petit affichage de la progression
        if i % (iterations // 10) == 0:
            print(f"Progression : {i/iterations*100:.0f}% (Temp: {temperature:.2f})")
            
    # Étape finale : s'assurer que tous les dominos sont dans le bon sens
    placement_final = []
    for idx, domino in enumerate(placement_actuel):
        caseA, caseB = emplacements[idx]
        cibleA, cibleB = cibles[caseA[1], caseA[0]], cibles[caseB[1], caseB[0]]
        _, meilleur_sens = calculer_erreur(domino, cibleA, cibleB)
        placement_final.append(meilleur_sens)
        
    print("Optimisation terminée !")
    return placement_final

def dessiner_demi_domino(draw, x_px, y_px, taille_case, valeur):
    """
    Dessine un carré blanc avec une bordure grise et les points noirs correspondants.
    """
    # 1. Dessiner le fond de la case (un demi-domino blanc avec bordure)
    draw.rectangle(
        [x_px, y_px, x_px + taille_case, y_px + taille_case], 
        fill="white", 
        outline="lightgray"
    )
    
    if valeur == 0:
        return 

    # 2. Définir le rayon des points (proportionnel à la taille de la case)
    r = max(1, taille_case // 10)
    
    # 3. Calculer les coordonnées relatives (Centre, Gauche, Droite, Haut, Bas, Milieu)
    c = 0.5  
    g, d = 0.25, 0.75  
    h, b = 0.25, 0.75  
    m = 0.5  
    
    # 4. Dictionnaire des positions des points pour les valeurs de 1 à 6
    positions_points = {
        1: [(c, c)],
        2: [(g, h), (d, b)],
        3: [(g, h), (c, c), (d, b)],
        4: [(g, h), (d, h), (g, b), (d, b)],
        5: [(g, h), (d, h), (c, c), (g, b), (d, b)],
        6: [(g, h), (d, h), (g, m), (d, m), (g, b), (d, b)]
    }
    
    # 5. Dessiner les cercles noirs aux bonnes positions
    for pos_x, pos_y in positions_points.get(valeur, []):
        cx = x_px + pos_x * taille_case
        cy = y_px + pos_y * taille_case
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="black")

def creer_mosaique_finale(emplacements, placement_final, colonnes, lignes, taille_case=20):
    """
    Génère l'image finale de la mosaïque en parcourant toutes les paires.
    
    - taille_case : Le nombre de pixels par demi-domino (ex: 20px donne de belles images)
    """
    # 1. Création de la toile de fond (noire pour faire ressortir les bordures)
    largeur_img = colonnes * taille_case
    hauteur_img = lignes * taille_case
    image_finale = Image.new("RGB", (largeur_img, hauteur_img), "black")
    draw = ImageDraw.Draw(image_finale)
    
    print(f"Génération de l'image ({largeur_img}x{hauteur_img} pixels)...")
    
    # 2. Boucle sur tous les dominos
    for i in range(len(emplacements)):
        case1, case2 = emplacements[i]
        val1, val2 = placement_final[i]
        
        # Convertir les coordonnées de la grille en pixels
        x1, y1 = case1[0] * taille_case, case1[1] * taille_case
        x2, y2 = case2[0] * taille_case, case2[1] * taille_case
        
        # Dessiner les deux moitiés du domino
        dessiner_demi_domino(draw, x1, y1, taille_case, val1)
        dessiner_demi_domino(draw, x2, y2, taille_case, val2)
        
        # 3. Astuce visuelle : Effacer la ligne de séparation entre les deux moitiés
        # pour donner l'illusion d'une seule pièce de domino solide (1x2)
        if case1[1] == case2[1]:  # Domino Horizontal
            draw.line([x2, y1 + 1, x2, y1 + taille_case - 1], fill="white", width=2)
        else:                     # Domino Vertical
            draw.line([x1 + 1, y2, x1 + taille_case - 1, y2], fill="white", width=2)
            
    print("Mosaïque terminée !")
    return image_finale


# ====================
# FONCTION DE LIAISON 
# ====================
def creer_mosaique_interface(image_pil, choix_jeu):
    if image_pil is None:
        return None, "Veuillez charger une image."
    type_jeu = "double-six" if "Six" in choix_jeu else "double-neuf"
    largeur_img, hauteur_img = image_pil.size
    ratio = hauteur_img / largeur_img
    largeur = 100
    hauteur = int(largeur * ratio)
    taille_case = largeur_img//largeur
    image = transfo_image(image_pil,largeur,hauteur)
    matrice = valeurs_grille(image,type_jeu)
    inventaire = generer_inventaire(largeur*hauteur//2,type_jeu)
    emplacements = generer_emplacements(largeur,hauteur)
    placement = optimiser_placement_recuit(matrice,emplacements,inventaire)
    image = creer_mosaique_finale(emplacements,placement,largeur,hauteur,taille_case)
    texte ='rien'
    return image, texte

# ==========================================
# CONSTRUCTION DE L'INTERFACE WEB
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🎲 Créateur de Mosaïque en Dominos")
    gr.Markdown("Générez un portrait unique en utilisant un inventaire strict de dominos.")
    
    with gr.Row():
        # --- COLONNE DE GAUCHE : ENTRÉES ---
        with gr.Column():
            entree_image = gr.Image(type="pil", label="Glissez votre image ici")
            choix_jeu = gr.Radio(
                choices=["Double-Six (Standard)", "Double-Neuf (Plus de contraste)"], 
                value="Double-Six (Standard)", 
                label="Choisissez le type de jeu"
            )
            bouton_go = gr.Button("Générer la mosaïque", variant="primary")
            
        # --- COLONNE DE DROITE : RÉSULTATS ---
        with gr.Column():
            # Gradio ajoute automatiquement l'icône "Télécharger" sur ce composant !
            sortie_image = gr.Image(type="pil", label="Résultat final", interactive=False)
            sortie_stats = gr.Textbox(label="Statistiques d'inventaire")

    # --- LIAISON DU BOUTON À LA FONCTION ---
    bouton_go.click(
        fn=creer_mosaique_interface,
        inputs=[entree_image, choix_jeu],
        outputs=[sortie_image, sortie_stats]
    )

# ==========================================
# LANCEMENT
# ==========================================
if __name__ == "__main__":
    # Ouvre une URL locale (ex: http://127.0.0.1:7860) dans ton navigateur
    interface.launch()