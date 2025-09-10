# Système de prédiction de catégorie d'un article

Ce projet sert à prédire la catégorie d'un article dans une des 4 catégories suivantes : politique, économie, environnement et sciences-santé. Il permet aussi de détecter un texte qui n'appartient à aucune de ces 4 catégories.

## Configuration initiale

### Prérequis

-  Python 3.8 ou supérieur
- 'python3-pip'
- 'python3-venv'
- 'scikit-learn 1.3.0' : Python >=3.8
- 'joblib 1.3.1' : Python >=3.7
- `Spacy 3.6.1` : Python >=3.6,<3.10

---

## Installation des Dépendances

1. Pour installer les dépendances nécessaires au projet, assurez-vous d'abord que Python3, python3-pip et python3-tk sont installés, puis exécutez la commande suivante :

```bash
pip3 install -r /usr/share/classification_articles/requirements.txt
```

2. Téléchargez le modèle `fr_core_news_sm` pour Spacy :

```bash
python3 -m spacy download fr_core_news_sm
```

## Utilisation

### Lancement du logiciel

Pour démarrer le programme, exécutez :

/usr/bin/Classeur

Fournissez votre texte et le programme fera une prédiction sur la catégorie de ce texte.
