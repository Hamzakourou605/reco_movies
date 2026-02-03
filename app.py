"""
MyTflix - Application Streamlit de recommandation de films
Interface Frontend avec données réelles et ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import json
from datetime import datetime

# Charger les variables d'environnement
load_dotenv()

# Import du modèle ML
from ml_model import MovieRecommender
from statistics import MovieStatistics
from image_dowmload import get_or_download_poster

# Configuration Streamlit
st.set_page_config(
    page_title='MyTflix — Recommandation de Films',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={"About": "MyTflix v1.0 - Système de recommandation basé sur l'IA et données réelles"}
)

# ============================================================================
# CONFIGURATION ET CACHE
# ============================================================================

@st.cache_resource
def load_recommender():
    """Charge le modèle ML en cache"""
    model_path = 'recommender_model.pkl'
    
    # Entraîne le modèle s'il n'existe pas
    if not Path(model_path).exists():
        st.warning("📚 Entraînement du modèle (première utilisation)...")
        recommender = MovieRecommender()
        recommender.train()
        recommender.save(model_path)
        st.success("✅ Modèle entraîné!")
    else:
        recommender = MovieRecommender.load(model_path)
    
    return recommender

# Charger le recommander
recommender = load_recommender()

@st.cache_resource
def load_statistics():
    """Charge les statistiques en cache"""
    return MovieStatistics(recommender.movies, recommender.ratings, recommender.tags)


# ============================================================================
# THÈME NETFLIX
# ============================================================================
_NETFLIX_CSS = '''
<style>
    * { margin: 0; padding: 0; }
    body, html { background-color: #0f0f0f !important; color: #e6e6e6; }
    .stApp { background-color: #0f0f0f; }
    .stSidebar { background-color: #1a1a1a; }
    
    .netflix-title { 
        color: #e50914; 
        font-size: 3rem; 
        font-weight: 900; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        margin-bottom: 1rem;
    }
    
    .section-title {
        color: #e6e6e6;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 3px solid #e50914;
    }
    
    .movie-card {
        background: #1a1a1a;
        border-radius: 8px;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 2px solid #333;
        cursor: pointer;
        padding: 1rem;
    }
    
    .movie-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 16px rgba(229, 9, 20, 0.4);
        border-color: #e50914;
    }
    
    .stButton > button {
        background-color: #e50914 !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.75rem 1.5rem !important;
        transition: background-color 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #b20710 !important;
    }
    
    .badge {
        display: inline-block;
        background: #e50914;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }
    
    .rating-badge {
        display: inline-block;
        background: #ffc107;
        color: #000;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    .stats-container {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
'''

st.markdown(_NETFLIX_CSS, unsafe_allow_html=True)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def display_movie_grid(movies, cols=5, show_rating=True):
    """Affiche une grille de films"""
    if movies.empty:
        st.info("❌ Aucun film à afficher")
        return
    
    cols_per_row = cols
    rows = [movies.iloc[i:i+cols_per_row] for i in range(0, len(movies), cols_per_row)]
    
    for row in rows:
        cols = st.columns(cols_per_row, gap='medium')
        for col, (_, movie) in zip(cols, row.iterrows()):
            with col:
                st.markdown(f"""
                <div class="movie-card">
                    <div style="background: linear-gradient(135deg, #e50914, #831010); height: 200px; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                        {movie['title'][:25]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**{movie['title']}**", help=f"ID: {movie['movieId']}")
                st.caption(f"📁 {movie['genres'][:50]}")
                
                if show_rating and 'avg_rating' in movie:
                    rating = movie['avg_rating']
                    st.markdown(f'<span class="rating-badge">⭐ {rating:.1f}/5</span>', unsafe_allow_html=True)
                    if 'rating_count' in movie:
                        st.caption(f"({int(movie['rating_count'])} votes)")

def display_movie_table(movies, columns=['movieId', 'title', 'genres', 'avg_rating']):
    """Affiche un tableau de films"""
    if movies.empty:
        st.info("❌ Aucun film à afficher")
        return
    
    display_df = movies[columns].copy()
    if 'avg_rating' in columns:
        display_df['avg_rating'] = display_df['avg_rating'].round(2)
    
    st.dataframe(display_df, use_container_width=True)

# ============================================================================
# NAVIGATION LATÉRALE
# ============================================================================
with st.sidebar:
    st.markdown('<div style="text-align:center; font-size:2rem; font-weight:900; color:#e50914;">🎬 MyTflix</div>', 
                unsafe_allow_html=True)
    st.markdown('---')
    
    page = st.radio(
        '📺 NAVIGATION',
        options=['🏠 Accueil', '⭐ Top Films', '🔍 Découvrir', '👤 Mon Profil', '🤖 Recommandation ML', '📊 Statistiques'],
        index=0
    )
    
    st.markdown('---')
    st.markdown('### 📊 Votre profil')
    
    # Obtenir les IDs d'utilisateurs disponibles
    available_users = recommender.ratings['userId'].unique()
    user_id = st.selectbox(
        'Sélectionner un utilisateur',
        options=available_users[:100],
        index=0
    )
    
    st.markdown(f'<div style="color:#999; font-size:0.8rem;">Connecté: User #{user_id}</div>', 
                unsafe_allow_html=True)

# ============================================================================
# PAGE 1: ACCUEIL
# ============================================================================
if page == '🏠 Accueil':
    st.markdown('<div class="netflix-title">🎬 Bienvenue sur MyTflix</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stats-container">
            <h3>📊 Statistiques</h3>
            <p><b>Total Films:</b> {len(recommender.movies):,}</p>
            <p><b>Total Évaluations:</b> {len(recommender.ratings):,}</p>
            <p><b>Total Utilisateurs:</b> {recommender.ratings['userId'].nunique():,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-container">
            <h3>🎯 Votre Recommandation</h3>
            <p><b>Utilisateur ID:</b> {user_id}</p>
            <p><b>Films évalués:</b> {len(recommender.get_user_ratings(user_id))}</p>
            <p><b>Note moyenne donnée:</b> {recommender.ratings[recommender.ratings['userId'] == user_id]['rating'].mean():.1f}/5</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Films recommandés pour vous
    st.markdown('<div class="section-title">✨ Recommandations Personnalisées</div>', unsafe_allow_html=True)
    
    try:
        recs = recommender.get_recommendations_by_ratings(user_id, n=12)
        if not recs.empty:
            display_movie_grid(recs, cols=4)
        else:
            st.info(f"👤 Utilisateur {user_id} n'a pas encore d'historique. Voici les meilleurs films:")
            top_movies = recommender.get_top_movies(n=12)
            display_movie_grid(top_movies, cols=4)
    except Exception as e:
        st.warning(f"⚠️ Impossible de générer les recommandations: {str(e)}")
        st.info("Voici les films les plus populaires:")
        top_movies = recommender.get_top_movies(n=12)
        display_movie_grid(top_movies, cols=4)

# ============================================================================
# PAGE 2: TOP FILMS
# ============================================================================
elif page == '⭐ Top Films':
    st.markdown('<div class="netflix-title">⭐ Les Meilleurs Films</div>', unsafe_allow_html=True)
    
    n_films = st.slider('Nombre de films à afficher', 10, 50, 20)
    
    st.markdown('<div class="section-title">🏆 Films les mieux notés</div>', unsafe_allow_html=True)
    top_movies = recommender.get_top_movies(n=n_films)
    display_movie_grid(top_movies, cols=4)
    
    # Statistiques
    st.markdown('---')
    st.markdown('<div class="section-title">📈 Graphique des Notes</div>', unsafe_allow_html=True)
    
    fig = px.bar(
        top_movies.head(20),
        x='title',
        y='avg_rating',
        color='avg_rating',
        color_continuous_scale=['#e50914', '#ffc107'],
        title='Top 20 Films par Note Moyenne',
        labels={'avg_rating': 'Note Moyenne', 'title': 'Film'},
        height=400
    )
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: DÉCOUVRIR
# ============================================================================
elif page == '🔍 Découvrir':
    st.markdown('<div class="netflix-title">🔍 Découvrir des Films</div>', unsafe_allow_html=True)
    
    # Options de filtrage
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Obtenir tous les genres
        all_genres = set()
        for genres_str in recommender.movies['genres'].dropna():
            all_genres.update(genres_str.split('|'))
        
        selected_genre = st.selectbox(
            '🎭 Sélectionner un genre',
            options=sorted(all_genres)
        )
    
    with col2:
        n_films = st.number_input('Nombre de films', 10, 50, 20)
    
    st.markdown('---')
    
    # Afficher les films du genre
    st.markdown(f'<div class="section-title">📽️ Films - {selected_genre}</div>', unsafe_allow_html=True)
    
    genre_movies = recommender.get_movies_by_genre(selected_genre, n=n_films)
    
    if not genre_movies.empty:
        display_movie_grid(genre_movies, cols=4)
        
        # Tableau de synthèse
        st.markdown('---')
        st.markdown('### 📊 Détails des films')
        display_movie_table(genre_movies, columns=['title', 'genres', 'avg_rating', 'rating_count'])
    else:
        st.warning(f"Aucun film trouvé pour le genre: {selected_genre}")

# ============================================================================
# PAGE 4: MON PROFIL
# ============================================================================
elif page == '👤 Mon Profil':
    st.markdown(f'<div class="netflix-title">👤 Profil Utilisateur #{user_id}</div>', unsafe_allow_html=True)
    
    user_ratings = recommender.get_user_ratings(user_id, n=50)
    
    if not user_ratings.empty:
        # Statistiques utilisateur
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📽️ Films évalués", len(user_ratings))
        with col2:
            st.metric("⭐ Note moyenne", f"{user_ratings['rating'].mean():.1f}/5")
        with col3:
            st.metric("🎯 Meilleure note", f"{user_ratings['rating'].max():.1f}")
        with col4:
            st.metric("📉 Plus basse note", f"{user_ratings['rating'].min():.1f}")
        
        st.markdown('---')
        st.markdown('<div class="section-title">📋 Vos Films Évalués</div>', unsafe_allow_html=True)
        
        # Afficher en grille
        display_movie_grid(user_ratings.head(20), cols=4, show_rating=True)
        
        # Tableau complet
        st.markdown('---')
        st.markdown('### 📊 Tableau complet')
        display_movie_table(user_ratings, columns=['title', 'genres', 'rating'])
        
        # Graphique de distribution
        st.markdown('---')
        st.markdown('<div class="section-title">📈 Distribution des Notes</div>', unsafe_allow_html=True)
        
        fig = px.histogram(
            user_ratings,
            x='rating',
            nbins=10,
            title='Distribution de vos évaluations',
            labels={'rating': 'Note', 'count': 'Nombre de films'},
            color_discrete_sequence=['#e50914'],
            height=350
        )
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"👤 L'utilisateur {user_id} n'a pas d'historique d'évaluation")

# ============================================================================
# PAGE 5: RECOMMANDATION PAR MACHINE LEARNING
# ============================================================================
elif page == '🤖 Recommandation ML':
    st.markdown('<div class="netflix-title">🤖 Recommandation par ML</div>', unsafe_allow_html=True)
    st.markdown('**Sélectionnez un ou plusieurs genres et découvrez les meilleurs films!**')
    
    # Obtenir tous les genres disponibles
    all_genres = recommender.get_all_genres()
    
    # Sélection des genres
    st.markdown('<div class="section-title">🎭 Sélectionner les Genres</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_genres = st.multiselect(
            '📺 Choisir un ou plusieurs genres:',
            options=all_genres,
            default=['Action'],
            help='Vous pouvez sélectionner plusieurs genres pour affiner vos recommandations'
        )
    
    with col2:
        nb_recommendations = st.slider(
            '📊 Nombre de recommandations',
            min_value=5,
            max_value=50,
            value=15,
            step=5
        )
    
    st.markdown('---')
    
    if selected_genres:
        # Afficher les statistiques des genres sélectionnés
        st.markdown('<div class="section-title">📈 Statistiques des Genres</div>', unsafe_allow_html=True)
        
        stats_cols = st.columns(len(selected_genres))
        
        for idx, genre in enumerate(selected_genres):
            with stats_cols[idx]:
                genre_stats = recommender.get_genre_stats(genre)
                if genre_stats:
                    st.metric(genre, f"⭐ {genre_stats['avg_rating']:.2f}")
                    st.caption(f"{genre_stats['total_movies']} films")
        
        st.markdown('---')
        
        # Obtenir les recommandations
        st.markdown('<div class="section-title">🎬 Films Recommandés</div>', unsafe_allow_html=True)
        
        recommendations = recommender.recommend_by_multiple_genres(selected_genres, n=nb_recommendations)
        
        if not recommendations.empty:
            # Calculer métriques agrégées supplémentaires
            tv_pct_list = []
            for _, mv in recommendations.iterrows():
                mid = mv['movieId']
                rc = int(mv.get('rating_count', 0) or 0)
                if rc > 0:
                    high = int(recommender.ratings[(recommender.ratings['movieId']==mid) & (recommender.ratings['rating']>=4)].shape[0])
                    tv_pct = high / rc * 100
                else:
                    tv_pct = 0
                tv_pct_list.append(tv_pct)

            avg_movie_score = recommendations['avg_rating'].mean() if len(recommendations)>0 else 0
            avg_movie_score_pct = avg_movie_score / 5.0 * 100
            avg_tv_score_pct = float(np.mean(tv_pct_list)) if len(tv_pct_list)>0 else 0

            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("🎬 Films Trouvés", len(recommendations))
            with col_b:
                st.metric("⭐ Rating Moyen", f"{recommendations['avg_rating'].mean():.2f}/5")
            with col_c:
                st.metric("🏆 Meilleur Rating", f"{recommendations['avg_rating'].max():.2f}")
            with col_d:
                st.metric("🗳️ Total Votes", int(recommendations['rating_count'].sum()))

            # Ligne supplémentaire: pourcentages
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Average Movie Score (%)", f"{avg_movie_score_pct:.0f}%")
            with c2:
                st.metric("Average TV Score (%)", f"{avg_tv_score_pct:.0f}%")
            with c3:
                st.metric("Avg Composite", f"{recommendations['composite_score'].mean():.3f}")
            with c4:
                st.write("")

            # Affichage en grille
            cols_per_row = 5
            rows = [recommendations.iloc[i:i+cols_per_row] for i in range(0, len(recommendations), cols_per_row)]
            
            for row_idx, row in enumerate(rows):
                cols = st.columns(cols_per_row, gap='large')
                
                for col_idx, (_, movie) in enumerate(row.iterrows()):
                    with cols[col_idx]:
                        # Poster (download if missing)
                        poster_path = None
                        try:
                            poster_path = get_or_download_poster(movie['movieId'], movie['title'])
                        except Exception:
                            poster_path = None

                        if poster_path and os.path.exists(poster_path):
                            st.image(poster_path, use_column_width=True, output_format='auto')
                        else:
                            # Fallback card
                            st.markdown(f"""
                            <div class="movie-card" style="text-align: center;">
                                <div style="background: linear-gradient(135deg, #e50914, #831010); 
                                            height: 180px; border-radius: 4px; display: flex; 
                                            align-items: center; justify-content: center; 
                                            color: white; font-weight: bold; font-size: 14px;
                                            overflow: hidden; padding: 10px;">
                                    {movie['title'][:30]}...
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Titre
                        st.markdown(f"**{movie['title'][:35]}**")

                        # Genres
                        st.caption(f"🎭 {movie['genres']}")

                        # Rating + TV score
                        rating = movie.get('avg_rating', 0) or 0
                        rating_count = int(movie.get('rating_count', 0) or 0)
                        if rating_count > 0:
                            high_count = int(recommender.ratings[(recommender.ratings['movieId']==movie['movieId']) & (recommender.ratings['rating']>=4)].shape[0])
                            tv_pct = high_count / rating_count * 100
                        else:
                            tv_pct = 0

                        col_rating, col_votes = st.columns([1,1])
                        with col_rating:
                            st.metric("⭐ Avg", f"{rating:.2f}/5", delta=f"{rating/5*100:.0f}%")
                        with col_votes:
                            st.metric("📺 TV Score", f"{tv_pct:.0f}%", delta=f"{high_count}/{rating_count}" if rating_count>0 else "0/0")
            
            st.markdown('---')
            
            # Tableau détaillé
            with st.expander("📋 Tableau Détaillé"):
                display_df = recommendations[[
                    'title', 'genres', 'avg_rating', 'rating_count', 'composite_score'
                ]].copy()
                display_df.columns = ['Film', 'Genres', 'Note Moy.', 'Votes', 'Score ML']
                display_df['Note Moy.'] = display_df['Note Moy.'].round(2)
                display_df['Score ML'] = display_df['Score ML'].round(3)
                display_df['Votes'] = display_df['Votes'].astype(int)
                
                st.dataframe(display_df, use_container_width=True)
            
            # Graphique comparatif
            st.markdown('---')
            st.markdown('<div class="section-title">📊 Comparaison des Films</div>', unsafe_allow_html=True)
            
            top_10_recs = recommendations.head(10)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=top_10_recs['title'],
                x=top_10_recs['avg_rating'],
                orientation='h',
                name='Note Moyenne',
                marker=dict(color='#e50914'),
                text=top_10_recs['avg_rating'].round(2),
                textposition='auto',
            ))
            
            fig.update_layout(
                title=f'Top 10 Films - Genres: {", ".join(selected_genres)}',
                xaxis_title='Note Moyenne',
                yaxis_title='Film',
                template='plotly_dark',
                height=500,
                hovermode='y unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques combinées
            st.markdown('---')
            st.markdown('<div class="section-title">📈 Statistiques des Recommandations</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎬 Films Trouvés", len(recommendations))
            with col2:
                st.metric("⭐ Rating Moyen", f"{recommendations['avg_rating'].mean():.2f}/5")
            with col3:
                st.metric("🏆 Meilleur Rating", f"{recommendations['avg_rating'].max():.2f}")
            with col4:
                st.metric("🗳️ Total Votes", int(recommendations['rating_count'].sum()))
        
        else:
            st.warning(f"❌ Aucun film trouvé pour les genres: {', '.join(selected_genres)}")
    
    else:
        st.info("👆 Sélectionnez au moins un genre pour voir les recommandations!")

# ============================================================================
# PAGE 6: STATISTIQUES
# ============================================================================
elif page == '📊 Statistiques':
    st.markdown('<div class="netflix-title">📊 Statistiques Complètes</div>', unsafe_allow_html=True)
    
    # Charger les statistiques
    stats = load_statistics()
    summary = stats.get_summary_statistics()
    
    # Afficher les statistiques clés
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎬 Total Films", f"{summary['total_movies']:,}")
    with col2:
        st.metric("⭐ Total Évaluations", f"{summary['total_ratings']:,}")
    with col3:
        st.metric("👥 Utilisateurs", f"{summary['total_users']:,}")
    with col4:
        st.metric("📊 Note Moyenne", f"{summary['avg_rating']:.2f}/5")
    
    st.markdown('---')
    
    # ========================================================================
    # ONGLETS DE VISUALISATIONS
    # ========================================================================
    
    tab1, tab2, tab3 = st.tabs(["📊 Histogrammes", "🎭 Diagrammes en Secteurs", "📈 Diagrammes d'Aires"])
    
    # ========================================================================
    # TAB 1: HISTOGRAMMES
    # ========================================================================
    with tab1:
        st.markdown('<div class="section-title">📊 Histogrammes</div>', unsafe_allow_html=True)
        
        # Histogramme 1: Distribution des évaluations
        st.markdown('### 📊 Distribution des Évaluations')
        fig1 = stats.histogram_ratings_distribution()
        st.plotly_chart(fig1, use_container_width=True)
        
        # Histogramme 2: Films par année
        st.markdown('### 🎬 Nombre de Films par Année')
        fig2 = stats.histogram_movies_per_year()
        st.plotly_chart(fig2, use_container_width=True)
        
        # Histogramme 3: Top genres
        st.markdown('### 🎭 Genres les Plus Populaires')
        fig3 = stats.histogram_top_genres()
        st.plotly_chart(fig3, use_container_width=True)
        
        # Histogramme 4: Évaluations par film
        st.markdown('### ⭐ Distribution des Évaluations par Film')
        fig4 = stats.histogram_ratings_per_movie()
        st.plotly_chart(fig4, use_container_width=True)
        
        # Histogramme 5: Note moyenne par genre
        st.markdown('### 📈 Note Moyenne par Genre')
        fig5 = stats.histogram_average_rating_by_genre()
        st.plotly_chart(fig5, use_container_width=True)
    
    # ========================================================================
    # TAB 2: DIAGRAMMES EN SECTEURS
    # ========================================================================
    with tab2:
        st.markdown('<div class="section-title">🎭 Diagrammes en Secteurs</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('### 🎭 Distribution des Genres (Top 10)')
            fig6 = stats.pie_chart_genres_distribution()
            st.plotly_chart(fig6, use_container_width=True)
        
        with col2:
            st.markdown('### ⭐ Catégories de Notes')
            fig7 = stats.pie_chart_rating_categories()
            st.plotly_chart(fig7, use_container_width=True)
        
        # Films les mieux notés
        st.markdown('### 🏆 Top 10 Films les Mieux Notés')
        fig8 = stats.pie_chart_top_rated_movies()
        st.plotly_chart(fig8, use_container_width=True)
    
    # ========================================================================
    # TAB 3: DIAGRAMMES D'AIRES
    # ========================================================================
    with tab3:
        st.markdown('<div class="section-title">📈 Diagrammes d\'Aires</div>', unsafe_allow_html=True)
        
        # Diagramme d'aires 1: Évaluations par année
        st.markdown('### 📈 Évolution des Évaluations par Année')
        fig9 = stats.area_chart_ratings_by_year()
        st.plotly_chart(fig9, use_container_width=True)
        
        # Diagramme d'aires 2: Évolution des genres
        st.markdown('### 🎬 Évolution des Genres au Fil du Temps')
        fig10 = stats.area_chart_genre_evolution()
        st.plotly_chart(fig10, use_container_width=True)
        
        # Diagramme d'aires 3: Utilisateurs cumulatifs
        st.markdown('### 👥 Évolution Cumulative des Utilisateurs')
        fig11 = stats.area_chart_cumulative_users()
        st.plotly_chart(fig11, use_container_width=True)
        
        # Diagramme d'aires 4: Évolution note moyenne
        st.markdown('### ⭐ Évolution de la Note Moyenne')
        fig12 = stats.area_chart_average_rating_evolution()
        st.plotly_chart(fig12, use_container_width=True)
        
        # Diagramme supplémentaire: Distribution des évaluations par utilisateur
        st.markdown('### 👥 Distribution des Évaluations par Utilisateur')
        fig13 = stats.area_chart_user_rating_distribution()
        st.plotly_chart(fig13, use_container_width=True)
    
    st.markdown('---')
    
    # ========================================================================
    # RÉSUMÉ DÉTAILLÉ
    # ========================================================================
    
    with st.expander("📋 Statistiques Détaillées"):
        st.markdown(f"""
        ## 📊 Résumé Complet des Statistiques
        
        ### 📈 Données Générales
        - **Total de Films:** {summary['total_movies']:,}
        - **Total d'Évaluations:** {summary['total_ratings']:,}
        - **Total d'Utilisateurs:** {summary['total_users']:,}
        - **Total de Tags:** {summary['total_tags']:,}
        - **Nombre de Genres Uniques:** {summary['genres_count']}
        
        ### ⭐ Statistiques des Évaluations
        - **Note Moyenne:** {summary['avg_rating']:.2f}/5.0
        - **Note Médiane:** {summary['median_rating']:.2f}/5.0
        - **Écart-type:** {summary['std_rating']:.2f}
        - **Note Minimum:** {summary['min_rating']:.1f}
        - **Note Maximum:** {summary['max_rating']:.1f}
        
        ### 📊 Moyennes et Ratios
        - **Moyenne d'Évaluations par Film:** {summary['avg_ratings_per_movie']:.1f}
        - **Moyenne d'Évaluations par Utilisateur:** {summary['avg_ratings_per_user']:.1f}
        """)
    
    st.markdown('---')
    st.markdown('<div style="text-align:center; color:#666; font-size:0.85rem; margin-top:2rem;">MyTflix v1.0 — Système de recommandation avec analyse statistique complète</div>',

               unsafe_allow_html=True)
st.markdown('''
<div style="text-align:center; color:#666; font-size:0.85rem; margin-top:3rem;">
    <p>Données simulées • Intégration ML prochainement</p>
</div>
''', unsafe_allow_html=True)