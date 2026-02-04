"""
MyTflix - API FastAPI de recommandation de films
Interface Backend avec ML
Déploiement sur Azure App Service avec Uvicorn
"""
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

# Charger les variables d'environnement
load_dotenv()

# Import du modèle ML
from ml_model import MovieRecommender
from statistics import MovieStatistics
from image_dowmload import get_or_download_poster

# ============================================================================
# VARIABLES GLOBALES
# ============================================================================
recommender = None

# ============================================================================
# GESTION DES ÉVÉNEMENTS DU CYCLE DE VIE DE L'APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le démarrage et l'arrêt de l'application"""
    # Démarrage - Charger le modèle
    global recommender
    print("🚀 Démarrage de l'API MyTflix...")
    print("📚 Chargement du modèle ML...")
    
    model_path = 'recommender_model.pkl'
    
    # Entraîne le modèle s'il n'existe pas
    if not Path(model_path).exists():
        print("📚 Entraînement du modèle (première utilisation)...")
        recommender = MovieRecommender()
        recommender.train()
        recommender.save(model_path)
        print("✅ Modèle entraîné et sauvegardé!")
    else:
        recommender = MovieRecommender.load(model_path)
        print("✅ Modèle chargé!")
    
    print("✅ API prête à recevoir des requêtes")
    
    yield  # Application en cours d'exécution
    
    # Arrêt
    print("🛑 Arrêt de l'API MyTflix...")

# ============================================================================
# INITIALISATION FASTAPI
# ============================================================================
app = FastAPI(
    title="MyTflix API",
    description="API de recommandation de films avec Machine Learning",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================================
# MODÈLES PYDANTIC POUR FASTAPI
# ============================================================================
class MovieResponse(BaseModel):
    """Modèle de réponse pour un film"""
    movieId: int
    title: str
    genres: str
    avg_rating: Optional[float] = None
    rating_count: Optional[int] = None

class RecommendationRequest(BaseModel):
    """Requête de recommandation"""
    user_id: int
    n_recommendations: int = 10

class GenreRecommendationRequest(BaseModel):
    """Requête de recommandation par genre"""
    genres: List[str]
    n_recommendations: int = 15

class APIResponse(BaseModel):
    """Réponse API générique"""
    success: bool
    message: str
    data: Optional[dict] = None

# ============================================================================
# ROUTES FASTAPI
# ============================================================================
# ============================================================================
# MODÈLES PYDANTIC POUR LES REQUÊTES/RÉPONSES
# ============================================================================

class MovieResponse(BaseModel):
    """Modèle de réponse pour un film"""
    movieId: int
    title: str
    genres: str
    avg_rating: Optional[float] = None
    rating_count: Optional[int] = None

class RecommendationRequest(BaseModel):
    """Requête de recommandation"""
    user_id: int
    n_recommendations: int = 10

class GenreRecommendationRequest(BaseModel):
    """Requête de recommandation par genre"""
    genres: List[str]
    n_recommendations: int = 15

class APIResponse(BaseModel):
    """Réponse API générique"""
    success: bool
    message: str
    data: Optional[dict] = None

# ============================================================================
# ROUTES FASTAPI
# ============================================================================

@app.get("/", tags=["Accueil"])
async def read_root():
    """Route racine de l'API - Statut et documentation"""
    return {
        "message": "MyTflix API - Système de recommandation de films",
        "version": "1.0.0",
        "status": "online" if recommender is not None else "initializing",
        "endpoints": {
            "GET /": "Cette route - Statut de l'API",
            "GET /health": "Vérifier la santé de l'API",
            "GET /top-films": "Récupère les meilleurs films",
            "GET /films/genre/{genre}": "Récupère les films d'un genre",
            "GET /utilisateur/{user_id}/films": "Récupère les films évalués par un utilisateur",
            "POST /recommandations": "Obtient des recommandations pour un utilisateur",
            "POST /recommandations/genres": "Obtient des recommandations par genres",
            "GET /stats": "Récupère les statistiques globales",
            "GET /genres": "Récupère tous les genres disponibles",
            "GET /utilisateurs/count": "Nombre total d'utilisateurs",
            "GET /docs": "Documentation Swagger interactive"
        }
    }

@app.get("/health", tags=["Santé"])
async def health_check():
    """Vérifier la santé de l'API"""
    try:
        if recommender is None:
            return {
                "status": "initializing",
                "message": "Le modèle est en cours de chargement",
                "timestamp": datetime.now().isoformat()
            }
        return {
            "status": "healthy",
            "message": "API opérationnelle",
            "model_loaded": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur santé: {str(e)}")

@app.get("/top-films", response_model=List[MovieResponse], tags=["Films"])
async def get_top_films(n: int = 20, skip: int = 0):
    """
    Récupère les meilleurs films
    - **n**: Nombre de films (par défaut 20)
    - **skip**: Nombre de films à ignorer (pour pagination)
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non encore chargé")
    
    try:
        top_movies = recommender.get_top_movies(n=n+skip)
        return [
            MovieResponse(
                movieId=int(m['movieId']),
                title=m['title'],
                genres=m['genres'],
                avg_rating=float(m.get('avg_rating', 0)),
                rating_count=int(m.get('rating_count', 0))
            )
            for _, m in top_movies.iloc[skip:skip+n].iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {str(e)}")

@app.get("/films/genre/{genre}", response_model=List[MovieResponse], tags=["Films"])
async def get_films_by_genre(genre: str, n: int = 20):
    """
    Récupère les films d'un genre spécifique
    - **genre**: Nom du genre (ex: Action, Drama, Comedy)
    - **n**: Nombre de films à retourner
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non encore chargé")
    
    try:
        movies = recommender.get_movies_by_genre(genre, n=n)
        return [
            MovieResponse(
                movieId=int(m['movieId']),
                title=m['title'],
                genres=m['genres'],
                avg_rating=float(m.get('avg_rating', 0)),
                rating_count=int(m.get('rating_count', 0))
            )
            for _, m in movies.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {str(e)}")

@app.get("/utilisateur/{user_id}/films", response_model=List[MovieResponse], tags=["Utilisateurs"])
async def get_user_films(user_id: int, n: int = 50):
    """
    Récupère les films évalués par un utilisateur
    - **user_id**: ID de l'utilisateur
    - **n**: Nombre de films à retourner
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non encore chargé")
    
    try:
        user_ratings = recommender.get_user_ratings(user_id, n=n)
        return [
            MovieResponse(
                movieId=int(m['movieId']),
                title=m['title'],
                genres=m['genres'],
                avg_rating=float(m.get('rating', 0))
            )
            for _, m in user_ratings.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {str(e)}")

@app.post("/recommandations", response_model=List[MovieResponse], tags=["Recommandations"])
async def get_recommendations(request: RecommendationRequest):
    """
    Obtient des recommandations pour un utilisateur
    - **user_id**: ID de l'utilisateur
    - **n_recommendations**: Nombre de recommandations
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non encore chargé")
    
    try:
        recs = recommender.get_recommendations_by_ratings(request.user_id, n=request.n_recommendations)
        return [
            MovieResponse(
                movieId=int(m['movieId']),
                title=m['title'],
                genres=m['genres'],
                avg_rating=float(m.get('avg_rating', 0)),
                rating_count=int(m.get('rating_count', 0))
            )
            for _, m in recs.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {str(e)}")

@app.post("/recommandations/genres", response_model=List[MovieResponse], tags=["Recommandations"])
async def get_recommendations_by_genres(request: GenreRecommendationRequest):
    """
    Obtient des recommandations basées sur plusieurs genres
    - **genres**: Liste de genres (ex: ["Action", "Drama"])
    - **n_recommendations**: Nombre de recommandations
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non encore chargé")
    
    try:
        recs = recommender.recommend_by_multiple_genres(
            request.genres, 
            n=request.n_recommendations
        )
        return [
            MovieResponse(
                movieId=int(m['movieId']),
                title=m['title'],
                genres=m['genres'],
                avg_rating=float(m.get('avg_rating', 0)),
                rating_count=int(m.get('rating_count', 0))
            )
            for _, m in recs.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {str(e)}")

@app.get("/stats", tags=["Statistiques"])
async def get_stats():
    """Récupère les statistiques globales de l'application"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non encore chargé")
    
    try:
        return {
            "total_films": int(len(recommender.movies)),
            "total_ratings": int(len(recommender.ratings)),
            "total_users": int(recommender.ratings['userId'].nunique()),
            "avg_rating": float(recommender.ratings['rating'].mean()),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {str(e)}")

@app.get("/genres", tags=["Genres"])
async def get_all_genres():
    """Récupère tous les genres disponibles"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non encore chargé")
    
    try:
        genres = recommender.get_all_genres()
        return {
            "genres": genres,
            "total": len(genres)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {str(e)}")

@app.get("/utilisateurs/count", tags=["Utilisateurs"])
async def get_users_count():
    """Récupère le nombre total d'utilisateurs"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non encore chargé")
    
    try:
        count = int(recommender.ratings['userId'].nunique())
        return {
            "total_users": count,
            "available_ids": list(recommender.ratings['userId'].unique()[:100])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {str(e)}")

@app.get("/recommend/{movie_title}", tags=["Recommandations"])
async def recommend_by_title(movie_title: str, n: int = 10):
    """
    Recommande des films similaires en fonction du titre d'un film
    - **movie_title**: Titre du film de référence
    - **n**: Nombre de recommandations (par défaut 10)
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non encore chargé")
    
    try:
        # Chercher le film par titre
        matching_movies = recommender.movies[
            recommender.movies['title'].str.contains(movie_title, case=False, na=False)
        ]
        
        if matching_movies.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"Film non trouvé: {movie_title}"
            )
        
        # Prendre le premier film trouvé
        movie_id = int(matching_movies.iloc[0]['movieId'])
        
        # Obtenir les recommandations
        recs = recommender.predict(movie_id, n=n)
        
        return [
            MovieResponse(
                movieId=int(m['movieId']),
                title=m['title'],
                genres=m['genres'],
                avg_rating=float(m.get('avg_rating', 0)),
                rating_count=int(m.get('rating_count', 0))
            )
            for _, m in recs.iterrows()
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {str(e)}")
    """Charge les statistiques en cache"""
    return MovieStatistics(recommender.movies, recommender.ratings, recommender.tags)


# ============================================================================
# POINT D'ENTRÉE DE L'APPLICATION
# ============================================================================
if __name__ == '__main__':
    import uvicorn
    # Démarrer le serveur
    # Documentation API disponible sur: http://localhost:8000/docs
    uvicorn.run(app, host='0.0.0.0', port=8000)