"""
Script de test pour les recommandations ML
Démontre l'utilisation du système de recommandation par genres
"""

from ml_model import MovieRecommender
import pandas as pd

def test_recommendations():
    """Teste le système de recommandation"""
    
    print("=" * 70)
    print("🤖 TEST DU SYSTÈME DE RECOMMANDATION ML - MyTflix")
    print("=" * 70)
    
    # Charger le modèle
    print("\n📚 Chargement du modèle...")
    try:
        recommender = MovieRecommender.load('recommender_model.pkl')
        print("✅ Modèle chargé avec succès!")
    except FileNotFoundError:
        print("⚠️ Modèle non trouvé, entraînement en cours...")
        recommender = MovieRecommender()
        recommender.train()
        recommender.save('recommender_model.pkl')
        print("✅ Modèle entraîné et sauvegardé!")
    
    # Afficher les genres disponibles
    print("\n🎭 Genres disponibles:")
    all_genres = recommender.get_all_genres()
    print(f"   {', '.join(all_genres[:10])}...")
    print(f"   Total: {len(all_genres)} genres")
    
    # Test 1: Recommandations Action
    print("\n" + "=" * 70)
    print("Test 1️⃣ : RECOMMANDATIONS ACTION")
    print("=" * 70)
    
    recs_action = recommender.recommend_by_multiple_genres(['Action'], n=10)
    if not recs_action.empty:
        print("\n🎬 Top 10 Films d'Action:")
        for idx, (_, movie) in enumerate(recs_action.iterrows(), 1):
            print(f"{idx:2d}. {movie['title']:<50} | ⭐ {movie['avg_rating']:>4.2f} | 🗳️  {int(movie['rating_count']):>5} votes")
    
    # Statistiques Genre
    action_stats = recommender.get_genre_stats('Action')
    if action_stats:
        print(f"\n📊 Statistiques Genre 'Action':")
        print(f"   • Films totaux: {action_stats['total_movies']}")
        print(f"   • Évaluations: {action_stats['total_ratings']}")
        print(f"   • Note moyenne: {action_stats['avg_rating']:.2f}/5.0")
        print(f"   • Médiane: {action_stats['median_rating']:.2f}")
    
    # Test 2: Recommandations Multi-Genres
    print("\n" + "=" * 70)
    print("Test 2️⃣ : RECOMMANDATIONS MULTI-GENRES (Action + Sci-Fi)")
    print("=" * 70)
    
    recs_multi = recommender.recommend_by_multiple_genres(['Action', 'Sci-Fi'], n=8)
    if not recs_multi.empty:
        print("\n🎬 Top 8 Films Action+Sci-Fi:")
        for idx, (_, movie) in enumerate(recs_multi.iterrows(), 1):
            score = movie['composite_score']
            print(f"{idx}. {movie['title']:<48} | ⭐ {movie['avg_rating']:>4.2f} | Score: {score:.3f}")
    
    # Test 3: Recommandations Romance
    print("\n" + "=" * 70)
    print("Test 3️⃣ : RECOMMANDATIONS ROMANCE")
    print("=" * 70)
    
    recs_romance = recommender.recommend_by_multiple_genres(['Romance'], n=8)
    if not recs_romance.empty:
        print("\n🎬 Top 8 Films Romantiques:")
        for idx, (_, movie) in enumerate(recs_romance.iterrows(), 1):
            print(f"{idx}. {movie['title']:<48} | ⭐ {movie['avg_rating']:>4.2f} | 🗳️  {int(movie['rating_count']):>5} votes")
    
    # Test 4: Recommandations Comedy + Drama
    print("\n" + "=" * 70)
    print("Test 4️⃣ : RECOMMANDATIONS COMEDY + DRAMA")
    print("=" * 70)
    
    recs_comedrama = recommender.recommend_by_multiple_genres(['Comedy', 'Drama'], n=8)
    if not recs_comedrama.empty:
        print("\n🎬 Top 8 Films Comedy+Drama:")
        for idx, (_, movie) in enumerate(recs_comedrama.iterrows(), 1):
            genres = movie['genres'][:35]
            print(f"{idx}. {movie['title']:<30} | {genres:<20} | ⭐ {movie['avg_rating']:>4.2f}")
    
    # Test 5: Comparaison de genres
    print("\n" + "=" * 70)
    print("Test 5️⃣ : COMPARAISON DES NOTES MOYENNES PAR GENRE")
    print("=" * 70)
    
    test_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
    print("\n📊 Genres            | Avg Rating | Total Films | Total Ratings")
    print("-" * 60)
    
    for genre in test_genres:
        stats = recommender.get_genre_stats(genre)
        if stats:
            print(f"{genre:<17} | {stats['avg_rating']:>9.2f} | {stats['total_movies']:>11} | {stats['total_ratings']:>13}")
    
    # Résumé global
    print("\n" + "=" * 70)
    print("📈 RÉSUMÉ GLOBAL")
    print("=" * 70)
    
    print(f"\n✅ Total Genres: {len(all_genres)}")
    print(f"✅ Total Films: {len(recommender.movies)}")
    print(f"✅ Total Évaluations: {len(recommender.ratings)}")
    print(f"✅ Total Utilisateurs: {recommender.ratings['userId'].nunique()}")
    print(f"✅ Note Moyenne Globale: {recommender.ratings['rating'].mean():.2f}/5.0")
    
    print("\n" + "=" * 70)
    print("✅ TOUS LES TESTS RÉUSSIS!")
    print("=" * 70)


if __name__ == '__main__':
    test_recommendations()
