# Embeddings 3D

## Purpose | Objectif

main.py is a script that generates a 3D plot of the embeddings of a given dataset. 

- The embeddings are generated using sentence-transformers, a library that provides pre-trained models for generating embeddings of text data.
- Embeddings are 384-dimensional vectors.
- The embeddings are reduced to 3 dimensions using trilateration (a method that uses the distances between points to determine their positions with respect of 3 reference points in 3D space).
- The 3 référence points for trilateration are chosen using PCA.
- The 3D plot is generated using plotly as a 3D scatter plot built from the 3D embeddings.

<hr>

- Les embeddings sont générés à l'aide de sentence-transformers, une bibliothèque qui fournit des modèles pré-entraînés pour générer des embeddings de données textuelles.
- Les embeddings sont des vecteurs de dimension 384.
- Les embeddings sont réduits à 3 dimensions en utilisant la trilatération (une méthode qui utilise les distances entre les points pour déterminer leur position par rapport à 3 points de référence dans l'espace 3D).
- Les 3 points de référence pour la trilatération sont choisis en utilisant l'ACP.
- La visualisation 3D est générée à l'aide de plotly comme un nuage de points 3D construit à partir des embeddings 3D.

## Usage | Utilisation

1. With the appropriate python IDE
2. Install the required libraries using requirements.txt
3. Run the script main.py

<hr>

1. Avec l'IDE Python approprié
2. Installez les bibliothèques requises en utilisant requirements.txt
3. Exécutez le script main.py







