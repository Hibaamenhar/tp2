# 🧠 Word Embeddings: Similarité, Analogies et Débiaisage

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## 📖 Description

Implémentation d'opérations sur les **word embeddings GloVe** : similarité cosinus, résolution d'analogies et débiaisage de genre.

## 🚀 Fonctionnalités

### 1. Similarité Cosinus
```python
similarity = cosine_similarity(word_to_vec_map['father'], word_to_vec_map['mother'])
# Résultat: 0.8909
```

### 2. Analogies
```python
complete_analogy('man', 'woman', 'king', word_to_vec_map)
# Résultat: 'queen'
```

### 3. Neutralisation
```python
e_debiased = neutralize('receptionist', g, word_to_vec_map)
# Biais: 0.33 → ~0
```

### 4. Égalisation
```python
e1, e2 = equalize(('man', 'woman'), g, word_to_vec_map)
# Asymétrie: 0.47 → 0.00
```

## 📊 Résultats

| Métrique | Avant | Après |
|----------|-------|-------|
| Biais (receptionist) | 0.3308 | ~10⁻¹⁷ |
| Asymétrie (man/woman) | 0.4738 | 0.0000 |

## 🛠️ Installation
```bash
# Cloner le repo
git clone https://github.com/hibaamenhar/tp2.git
cd tp2

# Installer les dépendances
pip install numpy jupyter

# Télécharger GloVe (avec Kaggle)
kaggle datasets download -d watts2/glove6b50dtxt
unzip glove6b50dtxt.zip -d data/
```

## 💻 Utilisation
```python
from w2v_utils import *

# Charger les embeddings
words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

# Utiliser les fonctions
similarity = cosine_similarity(vec1, vec2)
answer = complete_analogy('italy', 'italian', 'spain', word_to_vec_map)
```

## 📚 Ressources

- [GloVe Vectors](https://nlp.stanford.edu/projects/glove/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/watts2/glove6b50dtxt)

## 👤 Auteur

**Hiba Amenhar** - [@hibaamenhar](https://github.com/hibaamenhar)

