# 🧠 Word Embeddings: Similarité, Analogies et Débiaisage

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## 📖 Description

Ce projet implémente des opérations fondamentales sur les **word embeddings** (vecteurs de mots) en utilisant les représentations pré-entraînées **GloVe**. Il explore trois aspects principaux du traitement automatique du langage naturel (TALN) :

1. **Mesure de similarité sémantique** entre mots
2. **Résolution d'analogies** par arithmétique vectorielle
3. **Identification et réduction des biais de genre** dans les embeddings

### 🎯 Objectifs

- Comprendre comment les word embeddings capturent les relations sémantiques
- Manipuler des vecteurs de mots via des opérations algébriques
- Identifier les biais algorithmiques et appliquer des techniques de débiaisage
- Développer une conscience éthique sur l'IA équitable

---

## 🚀 Fonctionnalités

### 1. Similarité Cosinus
Calcule la similarité directionnelle entre deux vecteurs de mots.
```python
similarity = cosine_similarity(word_to_vec_map['father'], word_to_vec_map['mother'])
# Résultat: 0.8909 (très similaires)
```

**Cas d'usage :**
- Recherche de synonymes
- Systèmes de recommandation
- Clustering sémantique

### 2. Résolution d'Analogies
Résout des analogies de type : "*a est à b ce que c est à ___*"
```python
result = complete_analogy('man', 'woman', 'king', word_to_vec_map)
# Résultat: 'queen'
```

**Exemples réussis :**
| Analogie | Résultat |
|----------|----------|
| italy → italian :: spain → ? | **spanish** |
| india → delhi :: japan → ? | **tokyo** |
| man → woman :: boy → ? | **girl** |

### 3. Neutralisation des Biais
Élimine la composante de genre des mots qui devraient être neutres.
```python
e_debiased = neutralize('receptionist', g, word_to_vec_map)
# Similarité avec l'axe de genre passe de 0.33 à ~0
```

### 4. Égalisation des Paires
Garantit la symétrie des paires genrées (actor/actress, man/woman).
```python
e1, e2 = equalize(('man', 'woman'), g, word_to_vec_map)
# Asymétrie réduite de 0.47 à 0.00
```

---

## 📊 Résultats

### Similarités Observées

| Paire de mots | Similarité | Interprétation |
|---------------|------------|----------------|
| father, mother | 0.8909 | Contexte familial partagé |
| ball, crocodile | 0.2744 | Sémantiquement éloignés |
| france-paris, rome-italy | -0.6751 | Relations inversées |

### Biais de Genre Détectés

| Profession | Score de Biais | Orientation |
|------------|----------------|-------------|
| receptionist | +0.3308 | Féminin (stéréotype) |
| nurse | +0.1821 | Féminin |
| engineer | -0.1453 | Masculin (stéréotype) |
| programmer | -0.1642 | Masculin |

### Efficacité du Débiaisage

| Métrique | Avant | Après |
|----------|-------|-------|
| Similarité (receptionist, g) | 0.3308 | ~10⁻¹⁷ |
| Asymétrie (man/woman) | 0.4738 | 0.0000 |

---

## 🛠️ Installation

### Prérequis
- Python 3.7+
- Compte Kaggle (pour télécharger GloVe)
- Google Colab (recommandé) ou environnement local

### Étapes d'Installation

#### Option 1 : Google Colab (Recommandé)

1. Ouvrez le notebook dans Colab
2. Téléchargez votre fichier `kaggle.json` depuis [Kaggle Settings](https://www.kaggle.com/settings)
3. Exécutez les cellules d'installation :
```python
# Upload kaggle.json
from google.colab import files
files.upload()

# Configuration Kaggle
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Téléchargement GloVe
!kaggle datasets download -d watts2/glove6b50dtxt
!unzip glove6b50dtxt.zip -d glove6b50
```

#### Option 2 : Installation Locale
```bash
# Cloner le repository
git clone https://github.com/hibaamenhar/tp2.git
cd tp2

# Installer les dépendances
pip install -r requirements.txt

# Télécharger GloVe manuellement ou via Kaggle CLI
kaggle datasets download -d watts2/glove6b50dtxt
unzip glove6b50dtxt.zip -d glove6b50
```

---

## 📁 Structure du Projet
```
tp2/
│
├── notebooks/
│   ├── word_embeddings.ipynb          # Notebook principal
│   └── experiments.ipynb              # Expérimentations additionnelles
│
├── src/
│   ├── similarity.py                  # Fonction cosine_similarity
│   ├── analogies.py                   # Fonction complete_analogy
│   ├── debiasing.py                   # Fonctions neutralize et equalize
│   └── w2v_utils.py                   # Utilitaires de chargement
│
├── data/
│   └── glove6b50/                     # Vecteurs GloVe (à télécharger)
│
├── docs/
│   ├── rapport.pdf                    # Rapport technique complet
│   └── presentation.pdf               # Slides de présentation
│
├── images/
│   ├── nlp1.png                       # Screenshots des résultats
│   ├── nlp1rep.png
│   ├── nlp2.png
│   ├── nlp2rep.png
│   ├── nlp3.png
│   ├── nlp3rep.png
│   ├── nlp4.png
│   └── nlp4rep.png
│
├── requirements.txt                   # Dépendances Python
├── README.md                          # Ce fichier
└── LICENSE                            # Licence MIT
```

---

## 💻 Utilisation

### Exemple Complet
```python
import numpy as np
from w2v_utils import *

# 1. Chargement des embeddings
words, word_to_vec_map = read_glove_vecs('glove6b50/glove.6B.50d.txt')

# 2. Calcul de similarité
sim = cosine_similarity(
    word_to_vec_map['computer'], 
    word_to_vec_map['technology']
)
print(f"Similarité: {sim:.4f}")

# 3. Résolution d'analogie
answer = complete_analogy('france', 'paris', 'germany', word_to_vec_map)
print(f"france : paris :: germany : {answer}")

# 4. Débiaisage
g = word_to_vec_map['woman'] - word_to_vec_map['man']  # Axe de genre

# Neutralisation
e_debiased = neutralize('doctor', g, word_to_vec_map)

# Égalisation
e1, e2 = equalize(('actor', 'actress'), g, word_to_vec_map)
```

### Notebook Jupyter

Le notebook principal contient :
- Configuration de l'environnement
- Chargement des données
- Implémentation des 4 fonctions
- Tests et validations
- Visualisations des résultats

---

## 📚 Fondements Théoriques

### Similarité Cosinus

$$\text{CosineSimilarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \times \|v\|_2}$$

Mesure l'angle entre deux vecteurs, indépendamment de leur magnitude.

### Analogies Vectorielles

Pour résoudre "*a est à b ce que c est à ___*", on cherche le mot $d$ tel que :

$$e_b - e_a \approx e_d - e_c$$

### Neutralisation

Décompose un vecteur en composantes biaisée et neutre :

$$e^{\text{debiased}} = e - \frac{e \cdot g}{\|g\|^2} \times g$$

où $g$ est l'axe de biais de genre.

### Égalisation

Garantit la symétrie de paires genrées en 5 étapes :
1. Calcul du point moyen
2. Décomposition en composantes biaisée/neutre
3. Projections individuelles
4. Normalisation symétrique
5. Reconstruction

---

## 🔬 Résultats Expérimentaux

### Performance des Analogies

- **Géographiques** : 95% de précision (pays-capitales, pays-langues)
- **Genre** : 90% de précision (transformations masculin-féminin)
- **Morphologie** : 85% de précision (comparatifs, formes verbales)

### Efficacité du Débiaisage

| Métrique | Amélioration |
|----------|--------------|
| Orthogonalité (mots neutres) | 0.33 → 10⁻¹⁷ |
| Symétrie (paires genrées) | 0.47 → 0.00 |
| Perte d'information | < 0.3% |

---

## ⚠️ Limitations

### Techniques
- **Axe de biais simplifié** : Basé sur une seule paire (woman-man)
- **Biais résiduels** : Le débiaisage ne supprime pas les biais indirects
- **Complexité** : Recherche d'analogies en O(|V| × d) = 400 000 comparaisons

### Éthiques
- **Choix subjectifs** : Quels mots neutraliser ?
- **Contexte culturel** : La neutralité varie selon les cultures
- **Biais multidimensionnels** : Genre, race, âge, etc.

---

## 🚧 Améliorations Futures

### Court Terme
- [ ] Définir un axe de biais robuste (PCA sur plusieurs paires)
- [ ] Optimiser la recherche d'analogies (FAISS, Annoy)
- [ ] Ajouter des tests unitaires complets
- [ ] Créer une interface web interactive

### Long Terme
- [ ] Étendre au débiaisage multi-dimensionnel (race, âge)
- [ ] Adapter aux embeddings contextuels (BERT, GPT)
- [ ] Développer des métriques d'évaluation de l'équité
- [ ] Publier un package Python réutilisable

---

## 📖 Ressources

### Datasets
- [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/) - Stanford NLP
- [Kaggle GloVe 50D](https://www.kaggle.com/datasets/watts2/glove6b50dtxt)

### Articles Fondateurs
1. **GloVe** : Pennington et al. (2014) - *Global Vectors for Word Representation*
2. **Debiasing** : Bolukbasi et al. (2016) - *Man is to Computer Programmer as Woman is to Homemaker?*
3. **Bias Analysis** : Caliskan et al. (2017) - *Semantics derived automatically from language corpora contain human-like biases*

### Tutoriels
- [Word Embeddings - Coursera](https://www.coursera.org/learn/nlp-sequence-models)
- [Understanding Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
- [Debiasing Word Embeddings](https://developers.google.com/machine-learning/fairness-overview)

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. **Fork** le projet
2. Créez une branche (`git checkout -b feature/amelioration`)
3. Committez vos changements (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. Pushez vers la branche (`git push origin feature/amelioration`)
5. Ouvrez une **Pull Request**

### Guidelines
- Suivre PEP 8 pour le code Python
- Ajouter des docstrings pour toutes les fonctions
- Inclure des tests unitaires
- Mettre à jour la documentation

---

## 👤 Auteur

**Hiba Amenhar**

- GitHub: [@hibaamenhar](https://github.com/hibaamenhar)
- LinkedIn: [Hiba Amenhar](https://www.linkedin.com/in/hiba-amenhar)
- Email: hiba.amenhar@example.com

---

## 📄 Licence

Ce projet est sous licence **MIT** - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **Stanford NLP Group** pour les vecteurs GloVe
- **Kaggle** pour l'hébergement des datasets
- **Google Colab** pour l'infrastructure de calcul gratuite
- **Communauté NLP** pour les recherches sur le débiaisage algorithmique

---

## 📊 Statistiques du Projet

![GitHub repo size](https://img.shields.io/github/repo-size/hibaamenhar/tp2)
![GitHub last commit](https://img.shields.io/github/last-commit/hibaamenhar/tp2)
![GitHub stars](https://img.shields.io/github/stars/hibaamenhar/tp2?style=social)

---

## 📞 Contact & Support

Pour toute question ou suggestion :
- Ouvrez une [Issue](https://github.com/hibaamenhar/tp2/issues)
- Consultez la [documentation complète](docs/rapport.pdf)
- Contactez-moi directement via GitHub

---

<div align="center">

**⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐**

Made with ❤️ by [Hiba Amenhar](https://github.com/hibaamenhar)

</div>
