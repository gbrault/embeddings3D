"""Déscription:
Ce script utilise la bibliothèque Sentence Transformer pour incorporer des phrases dans un espace à 384 dimensions.
Il utilise ensuite l'ACP pour réduire la dimensionnalité à 3D pour trouver 3 points utilisés comme référence pour la trilatération.
Il utilise ensuite la trilatération pour placer les embeddings dans un espace 3D.
Enfin, il crée un nuage de points 3D des embeddings en utilisant Plotly.
Chaque point représente une phrase, et la couleur du point représente le groupe auquel il appartient.
Le survol d'un point affichera la phrase qu'il représente.
Description: 
This script uses the Sentence Transformer library to embed sentences into a 384-dimensional space. 
It then uses PCA to reduce the dimensionality to 3D to find 3 points used as the trilateration reference. 
It then uses trilateration to place the embeddings in a 3D space. 
Finally, it creates a 3D scatter plot of the embeddings using Plotly.
Every dot represents a sentence, and the color of the dot represents the group it belongs to.
Hovering over a dot will show the sentence it represents.
"""
# Importation des bibliothèques
# Import libraries
print("Loading libraries...")
import time
import numpy as np
from scipy.optimize import fsolve
import plotly.graph_objs as go
import plotly.io as pio
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from plotly.colors import sample_colorscale
from scipy.optimize import minimize
from data import sentences, groups, group_names
"""
# Importer les phrases et les groupes
# Import sentences and groups
# data.py contient les listes suivantes:
# data.py contains the following lists:
# Liste des phrases
# List of sentences
sentences = [] 
# Liste des groupes
# List of groups
groups = []
# Liste des noms de groupes
# List of group names
group_names = []
"""

# Fonction pour mesurer le temps pris pour une étape spécifique
# Function to measure time taken for a specific step
def measure_time(step_name, func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"{step_name} took {end_time - start_time:.4f} seconds")
    return result

# Charger un modèle Sentence Transformer pré-entraîné
# Load a pretrained Sentence Transformer model
model_name = "all-MiniLM-L6-v2"
print(f"Loading model '{model_name}'...")
model = measure_time("Model loading", SentenceTransformer, model_name)

# Calculer les embeddings en appelant model.encode()
# Calculate embeddings by calling model.encode()
print("Calculating embeddings...")
embeddings = measure_time("Embedding calculation", model.encode, sentences)
print(f"Embeddings shape: {embeddings.shape}")  # Should be (270, 384)

# Réduction de dimension avec PCA pour trouver l'ellipsoïde
# Dimension reduction with PCA to find the ellipsoid
print("Réduction de dimension avec PCA...")
pca = PCA(n_components=3)
embeddings_reduced = measure_time("PCA reduction", pca.fit_transform, embeddings)

# Calculer le centre de l'ellipsoïde
# Calculate center of ellipsoid
center = np.mean(embeddings_reduced, axis=0)
median_center = np.median(embeddings_reduced, axis=0)
if np.linalg.norm(center) > np.linalg.norm(median_center):
    center = median_center

# Calculer les axes principaux
# Calculate principal axes
U, s, Vt = np.linalg.svd(embeddings_reduced - center, full_matrices=False)
principal_axes = U[:, :3]

# Calculer les points de référence
# Calculate reference points
ref_points = pca.inverse_transform(principal_axes)
# ref_points c'est les points A, B, C dans l'espace n dim
# ref_points are the A, B, C points in n dim space

# Calculer les distances entre les points de référence
# Calculate distances between reference points
d_AB = np.linalg.norm(ref_points[1] - ref_points[0])
d_AC = np.linalg.norm(ref_points[2] - ref_points[0])
d_BC = np.linalg.norm(ref_points[2] - ref_points[1])

# Définir les équations pour la trilatération
# Define equations for trilateration
def place_reference_points(A, B, C):
    # A_prime, B_prime, C_prime sont les points de référence dans l'espace 3D
    # A_prime, B_prime, C_prime are the reference points in 3D space
    A_prime = np.array([0, 0, 0])
    B_prime = np.array([d_AB, 0, 0])

    def equations(p):
        x, y = p
        return (x**2 + y**2 - d_AC**2, (x - d_AB)**2 + y**2 - d_BC**2)

    initial_guess = (0.5, 0.5)
    x, y = fsolve(equations, initial_guess)
    C_prime = np.array([x, y, 0])

    return A_prime, B_prime, C_prime

# Placer les points de référence
# Place reference points
A_prime, B_prime, C_prime = place_reference_points(ref_points[0], ref_points[1], ref_points[2])

# Définir les équations pour le calcul du nouveau point
# Define equations for new point calculation
def place_new_point(P, A_prime, B_prime, C_prime, A, B, C):
    # if fsolve fails, use minimize as a fallback
    d_AP = np.linalg.norm(A - P)
    d_BP = np.linalg.norm(B - P)
    d_CP = np.linalg.norm(C - P)

    def point_equations(p):
        px, py, pz = p
        eq1 = px**2 + py**2 + pz**2 - d_AP**2
        eq2 = (px - B_prime[0])**2 + py**2 + pz**2 - d_BP**2
        eq3 = (px - C_prime[0])**2 + (py - C_prime[1])**2 + pz**2 - d_CP**2
        return (eq1, eq2, eq3)

    def objective(p):
        px, py, pz = p
        dist_AP = np.sqrt((px - A_prime[0])**2 + (py - A_prime[1])**2 + (pz - A_prime[2])**2)
        dist_BP = np.sqrt((px - B_prime[0])**2 + (py - B_prime[1])**2 + (pz - B_prime[2])**2)
        dist_CP = np.sqrt((px - C_prime[0])**2 + (py - C_prime[1])**2 + (pz - C_prime[2])**2)
        return ((dist_AP - d_AP)**2 + (dist_BP - d_BP)**2 + (dist_CP - d_CP)**2)

    initial_guess_P = (0.5, 0.5, 0.5)
    try:
        px, py, pz = fsolve(point_equations, initial_guess_P)
        return np.array([px, py, pz])
    except:
        result = minimize(objective, initial_guess_P)
        return result.x

# Mapper tous les embeddings en 3D
# Map all embeddings to 3D
print("Mapping all embeddings to 3D...")
start_time = time.time()
mapped_points = [A_prime, B_prime, C_prime]
for i in range(3, len(embeddings)):
    P = embeddings[i]
    P_prime = place_new_point(P, A_prime, B_prime, C_prime, ref_points[0], ref_points[1], ref_points[2])
    mapped_points.append(P_prime)
end_time = time.time()
print(f"Mapping all points took {end_time - start_time:.4f} seconds")

mapped_points = np.array(mapped_points)

# Définir des couleurs personnalisées pour chaque groupe
# Define custom colors for each group
unique_groups = sorted(set(groups))
color_scale = sample_colorscale('Viridis', [n / (len(unique_groups) - 1) for n in range(len(unique_groups))])
color_map = {group: color for group, color in zip(unique_groups, color_scale)}

# Créer un nuage de points 3D en utilisant Plotly
# Create 3D scatter plot using Plotly
print("Creating 3D scatter plot...")
fig = go.Figure()

# Ajouter les points au nuage de points
# Add points to the scatter plot
fig.add_trace(go.Scatter3d(
    x=mapped_points[:, 0],
    y=mapped_points[:, 1],
    z=mapped_points[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=[color_map[group] for group in groups],
        opacity=0.8
    ),
    hovertext=[f"{group_names[groups[i]-1]}<br>{s}" for i,s in enumerate(sentences)],  # Add sentences as hover text
    hoverinfo='text'
))

# Mise en page du tracé
# Set plot layout
fig.update_layout(
    title="3D Scatter Plot of Sentence Embeddings",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    width=800,
    height=800,
)

# Visualiser le tracé
# Show plot
print("Displaying plot...")
measure_time("Plot display", pio.show, fig)