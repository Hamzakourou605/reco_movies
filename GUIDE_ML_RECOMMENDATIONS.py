"""
GUIDE D'UTILISATION RAPIDE - Recommandations ML MyTflix
========================================================
"""

# 📖 GUIDE UTILISATEUR INTERFACE STREAMLIT
# =========================================

"""
1. ACCÉDER À LA PAGE "🤖 Recommandation ML"
   - Cliquez sur le bouton "🤖 Recommandation ML" dans la navigation latérale

2. SÉLECTIONNER LES GENRES
   - Utilisez le multiselect pour choisir un ou plusieurs genres
   - Genres disponibles: Action, Adventure, Comedy, Drama, Horror, Romance, Sci-Fi, etc.
   - Exemple: Sélectionnez "Action" et "Adventure" pour films d'action-aventure

3. AJUSTER LE NOMBRE DE RECOMMANDATIONS
   - Utilisez le slider (5-50 films)
   - Défaut: 15 films

4. EXPLORER LES RÉSULTATS
   - Grille visuelle avec 5 films par ligne
   - Affichage: Titre, genres, note moyenne, nombre de votes
   - Cliquez sur "📋 Tableau Détaillé" pour voir tous les films en tableau

5. ANALYSER LES STATISTIQUES
   - Voir les stats de chaque genre sélectionné
   - Graphique comparatif des 10 meilleurs films
   - Résumé avec total films, note moyenne, meilleur rating

"""

# 🐍 GUIDE PROGRAMMEUR - UTILISATION EN PYTHON
# ==============================================

from ml_model import MovieRecommender

# === EXEMPLE 1: Recommandations simples ===
recommender = MovieRecommender.load('recommender_model.pkl')

# Obtenir les 10 meilleurs films d'action
action_films = recommender.recommend_by_multiple_genres(['Action'], n=10)
print(action_films)

# === EXEMPLE 2: Recommandations multi-genres ===
# Obtenir les 20 meilleurs films combinant action ET sci-fi
multi_genre_films = recommender.recommend_by_multiple_genres(
    genres=['Action', 'Sci-Fi'],
    n=20
)

# === EXEMPLE 3: Obtenir tous les genres ===
all_genres = recommender.get_all_genres()
print(f"Genres disponibles: {all_genres}")

# === EXEMPLE 4: Statistiques d'un genre ===
action_stats = recommender.get_genre_stats('Action')
print(f"Nombre de films action: {action_stats['total_movies']}")
print(f"Note moyenne: {action_stats['avg_rating']:.2f}/5.0")

# === EXEMPLE 5: Boucler sur les résultats ===
recommendations = recommender.recommend_by_multiple_genres(['Romance'], n=5)

for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
    print(f"{idx}. {movie['title']}")
    print(f"   Note: {movie['avg_rating']}/5.0")
    print(f"   Votes: {int(movie['rating_count'])}")
    print()

"""

# 📊 COLONNES RETOURNÉES PAR recommend_by_multiple_genres()
# ===========================================================

DataFrame avec les colonnes suivantes:

- movieId: ID du film (int)
- title: Titre du film (str)
- genres: Genres du film (str, séparés par |)
- avg_rating: Note moyenne (float, 0-5)
- rating_count: Nombre d'évaluations (int)
- composite_score: Score composite ML (float, 0-1)

"""

# 🎯 CAS D'USAGE TYPIQUES
# ======================

"""

1. TROUVER UN FILM D'ACTION
   >>> recommender.recommend_by_multiple_genres(['Action'], n=15)

2. CHERCHER UN FILM ROMANTIQUE POUR SOIRÉE
   >>> recommender.recommend_by_multiple_genres(['Romance', 'Comedy'], n=10)

3. RECOMMANDATIONS HORREUR POPULAIRES
   >>> horror_films = recommender.recommend_by_multiple_genres(['Horror'], n=8)

4. DÉCOUVRIR SCI-FI AVENTURE
   >>> recommender.recommend_by_multiple_genres(['Sci-Fi', 'Adventure'], n=20)

5. DRAMES RECONNUS
   >>> dramas = recommender.recommend_by_multiple_genres(['Drama'], n=25)

6. ANIMATION ENFANTS
   >>> kids_movies = recommender.recommend_by_multiple_genres(['Children', 'Animation'], n=15)

"""

# 🔧 FILTRES APPLIQUÉS AUTOMATIQUEMENT
# ====================================

"""
Le système applique automatiquement:

1. ✅ Filtre de genres: Film doit contenir au moins un genre sélectionné
2. ✅ Filtre d'évaluations: Film doit avoir ≥ 1 évaluation
3. ✅ Tri: Ordre décroissant du score composite
4. ✅ Limite: Limité au nombre demandé
5. ✅ Scores: 70% ratings + 30% popularité

"""

# 📈 ALGORITHME DÉTAILLÉ
# ======================

"""
Score Composite = (0.7 × note_normalisée) + (0.3 × popularité_normalisée)

Où:
- note_normalisée = avg_rating / 5.0 (ramenée à 0-1)
- popularité_normalisée = (votes / max_votes_du_genre) (0-1)

Avantages:
✓ Films bien notés priorisés (70%)
✓ Films populaires aussi recommandés (30%)
✓ Équilibre entre qualité et popularité

"""

# 🚀 PERFORMANCES
# ===============

"""
Temps de recommandation typique: < 100ms
Scalabilité: Jusqu'à 10,000+ films
Précision: Basée sur données réelles MovieLens

"""

# ❓ DÉPANNAGE
# ===========

"""

Problème: Aucun film trouvé pour un genre
Solution: Vérifiez le nom du genre exact avec recommender.get_all_genres()

Problème: Recommandations identiques
Cause: C'est normal si peu de films correspondent
Solution: Sélectionnez plusieurs genres ou augmentez le nombre

Problème: Films avec peu de votes en haut
Cause: Certains films petits budgets ont de meilleures notes
Solution: C'est exact! Le système considère la qualité avant tout

"""

# 📚 RÉFÉRENCES
# =============

print("""
Documentation complète: README_ML.md
Tests: python test_ml_recommendations.py
Fichier principal: ml_model.py (classe MovieRecommender)
Interface: app.py (page "🤖 Recommandation ML")
""")
