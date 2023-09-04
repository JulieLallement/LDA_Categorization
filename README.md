# Modèle LDA pour la catégorisation de documents

Ce modèle LDA a été développé dans le cadre de ma thèse pour catégoriser des documents en 9 clusters, conformément aux besoins d'un service métier. L'objectif principal est de permettre une catégorisation automatique et efficace des documents.

## Installation des dépendances

Assurez-vous d'installer les dépendances nécessaires en exécutant les commandes suivantes :

```bash
pip install gensim
pip install pyLDAvis
pip install --upgrade pyLDAvis
pip install nltk 
```

## Utilisation

- Importation des bibliothèques nécessaires
- Chargement de vos données normalisées (tokenizées, suppression des stopwords...), adaptées pour le modèle LDA
- Entraînement de votre modèle LDA avec la bibliothèque Gensim
- Analyse des topics et regroupement en catégorie
