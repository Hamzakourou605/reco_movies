"""
Module de statistiques et visualisations pour MyTflix
- Histogrammes
- Diagrammes en secteurs (Pie Charts)
- Diagrammes d'aires (Area Charts)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


class MovieStatistics:
    def __init__(self, movies_df, ratings_df, tags_df):
        """
        Initialise la classe avec les données
        
        Args:
            movies_df: DataFrame des films
            ratings_df: DataFrame des évaluations
            tags_df: DataFrame des tags
        """
        self.movies = movies_df
        self.ratings = ratings_df
        self.tags = tags_df
        self._prepare_data()
    
    def _prepare_data(self):
        """Prépare les données pour l'analyse"""
        # Fusion des données
        self.merged_data = self.ratings.merge(self.movies, on='movieId', how='left')
        
        # Extraction des genres (un film peut avoir plusieurs genres)
        self.movies['genre_list'] = self.movies['genres'].str.split('|')
        self.genre_data = self.movies.explode('genre_list')
    
    # ========================================================================
    # HISTOGRAMMES
    # ========================================================================
    
    def histogram_ratings_distribution(self):
        """Histogramme de la distribution des évaluations"""
        fig = px.histogram(
            self.ratings,
            x='rating',
            nbins=20,
            title='📊 Distribution des Évaluations',
            labels={'rating': 'Note', 'count': 'Nombre d\'évaluations'},
            color_discrete_sequence=['#e50914'],
            opacity=0.7
        )
        
        fig.update_layout(
            template='plotly_dark',
            hovermode='x unified',
            xaxis_title='Note donnée',
            yaxis_title='Nombre d\'évaluations',
            font=dict(size=12),
            height=400,
            bargap=0.1
        )
        fig.update_xaxes(dtick=0.5)
        
        return fig
    
    def histogram_movies_per_year(self):
        """Histogramme du nombre de films par année"""
        # Extraire l'année du titre
        self.movies['year'] = self.movies['title'].str.extract(r'\((\d{4})\)', expand=False)
        self.movies['year'] = pd.to_numeric(self.movies['year'], errors='coerce')
        
        movies_by_year = self.movies.dropna(subset=['year']).groupby('year').size().reset_index(name='count')
        
        fig = px.bar(
            movies_by_year,
            x='year',
            y='count',
            title='🎬 Nombre de Films par Année',
            labels={'year': 'Année', 'count': 'Nombre de films'},
            color='count',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            template='plotly_dark',
            hovermode='x unified',
            xaxis_title='Année',
            yaxis_title='Nombre de films',
            font=dict(size=12),
            height=400,
            showlegend=False
        )
        fig.update_xaxes(type='linear')
        
        return fig
    
    def histogram_top_genres(self):
        """Histogramme des genres les plus populaires"""
        genre_counts = self.genre_data['genre_list'].value_counts().head(15)
        
        fig = px.bar(
            x=genre_counts.values,
            y=genre_counts.index,
            orientation='h',
            title='🎭 Top 15 des Genres les Plus Populaires',
            labels={'x': 'Nombre de films', 'y': 'Genre'},
            color=genre_counts.values,
            color_continuous_scale='Hot'
        )
        
        fig.update_layout(
            template='plotly_dark',
            hovermode='y unified',
            xaxis_title='Nombre de films',
            yaxis_title='Genre',
            font=dict(size=11),
            height=500,
            showlegend=False,
            margin=dict(l=150)
        )
        
        return fig
    
    def histogram_ratings_per_movie(self):
        """Histogramme du nombre d'évaluations par film"""
        ratings_count = self.ratings.groupby('movieId').size().reset_index(name='rating_count')
        
        fig = px.histogram(
            ratings_count,
            x='rating_count',
            nbins=50,
            title='⭐ Distribution du Nombre d\'Évaluations par Film',
            labels={'rating_count': 'Nombre d\'évaluations', 'count': 'Nombre de films'},
            color_discrete_sequence=['#e50914'],
            opacity=0.7
        )
        
        fig.update_layout(
            template='plotly_dark',
            hovermode='x unified',
            xaxis_title='Nombre d\'évaluations',
            yaxis_title='Nombre de films',
            font=dict(size=12),
            height=400,
            xaxis_type='log'
        )
        
        return fig
    
    def histogram_average_rating_by_genre(self):
        """Histogramme des notes moyennes par genre"""
        # Fusion des données genre avec les ratings
        genre_ratings = self.genre_data.merge(
            self.ratings[['movieId', 'rating']], 
            on='movieId'
        )
        
        avg_rating_by_genre = genre_ratings.groupby('genre_list')['rating'].mean().sort_values(ascending=False).head(15)
        
        fig = px.bar(
            x=avg_rating_by_genre.index,
            y=avg_rating_by_genre.values,
            title='📈 Note Moyenne par Genre (Top 15)',
            labels={'x': 'Genre', 'y': 'Note moyenne'},
            color=avg_rating_by_genre.values,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            template='plotly_dark',
            hovermode='x unified',
            xaxis_title='Genre',
            yaxis_title='Note moyenne',
            font=dict(size=11),
            height=400,
            xaxis_tickangle=-45,
            margin=dict(b=100)
        )
        
        return fig
    
    # ========================================================================
    # DIAGRAMMES EN SECTEURS (PIE CHARTS)
    # ========================================================================
    
    def pie_chart_genres_distribution(self):
        """Diagramme en secteurs de la distribution des genres"""
        genre_counts = self.genre_data['genre_list'].value_counts().head(10)
        
        fig = px.pie(
            values=genre_counts.values,
            names=genre_counts.index,
            title='🎭 Distribution des Genres (Top 10)',
            color_discrete_sequence=px.colors.sequential.Reds[::-1]
        )
        
        fig.update_layout(
            template='plotly_dark',
            font=dict(size=12),
            height=500,
            hovermode='closest'
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='label+percent',
            textfont=dict(size=10)
        )
        
        return fig
    
    def pie_chart_rating_categories(self):
        """Diagramme en secteurs des catégories de notes"""
        # Catégoriser les notes
        self.ratings['rating_category'] = pd.cut(
            self.ratings['rating'],
            bins=[0, 1, 2, 3, 4, 5],
            labels=['Très mauvais (≤1)', 'Mauvais (1-2)', 'Moyen (2-3)', 'Bon (3-4)', 'Très bon (4-5)'],
            include_lowest=True
        )
        
        category_counts = self.ratings['rating_category'].value_counts()
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='⭐ Distribution des Catégories de Notes',
            color_discrete_sequence=px.colors.sequential.RdYlGn[::-1]
        )
        
        fig.update_layout(
            template='plotly_dark',
            font=dict(size=11),
            height=500,
            hovermode='closest'
        )
        
        fig.update_traces(
            textposition='auto',
            textinfo='label+percent',
            textfont=dict(size=10)
        )
        
        return fig
    
    def pie_chart_top_rated_movies(self):
        """Diagramme en secteurs des films les mieux notés"""
        # Calculer la note moyenne par film
        avg_ratings = self.ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
        # Filtrer les films avec au moins 20 évaluations
        avg_ratings = avg_ratings[avg_ratings['count'] >= 20].sort_values('mean', ascending=False).head(10)
        
        # Merger avec les titres
        top_movies = avg_ratings.merge(self.movies[['movieId', 'title']], on='movieId')
        
        fig = px.pie(
            values=top_movies['count'],
            names=top_movies['title'],
            title='🏆 Top 10 Films les Mieux Notés (≥20 évaluations)',
            color=top_movies['mean'],
            color_continuous_scale='Hot'
        )
        
        fig.update_layout(
            template='plotly_dark',
            font=dict(size=10),
            height=500,
            hovermode='closest'
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='label+percent',
            textfont=dict(size=9)
        )
        
        return fig
    
    # ========================================================================
    # DIAGRAMMES D'AIRES (AREA CHARTS)
    # ========================================================================
    
    def area_chart_ratings_by_year(self):
        """Diagramme d'aires des évaluations par année"""
        # Ajouter timestamp (conversion de timestamp Unix)
        self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        self.ratings['year'] = self.ratings['timestamp'].dt.year
        
        ratings_by_year = self.ratings.groupby('year').size().reset_index(name='count')
        
        fig = px.area(
            ratings_by_year,
            x='year',
            y='count',
            title='📈 Évolution du Nombre d\'Évaluations par Année',
            labels={'year': 'Année', 'count': 'Nombre d\'évaluations'},
            color_discrete_sequence=['#e50914']
        )
        
        fig.update_layout(
            template='plotly_dark',
            hovermode='x unified',
            xaxis_title='Année',
            yaxis_title='Nombre d\'évaluations',
            font=dict(size=12),
            height=400,
            fillna=0
        )
        
        fig.update_traces(fillna='tozeroy')
        
        return fig
    
    def area_chart_genre_evolution(self):
        """Diagramme d'aires de l'évolution des genres au fil du temps"""
        # Merger genres avec ratings
        genre_ratings = self.genre_data.merge(
            self.ratings[['movieId', 'timestamp']], 
            on='movieId'
        )
        genre_ratings['timestamp'] = pd.to_datetime(genre_ratings['timestamp'], unit='s')
        genre_ratings['year'] = genre_ratings['timestamp'].dt.year
        
        # Top 8 genres
        top_genres = self.genre_data['genre_list'].value_counts().head(8).index
        genre_evolution = genre_ratings[genre_ratings['genre_list'].isin(top_genres)].groupby(
            ['year', 'genre_list']
        ).size().reset_index(name='count')
        
        fig = px.area(
            genre_evolution,
            x='year',
            y='count',
            color='genre_list',
            title='🎬 Évolution des Genres au Fil du Temps (Top 8)',
            labels={'year': 'Année', 'count': 'Nombre d\'évaluations', 'genre_list': 'Genre'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(
            template='plotly_dark',
            hovermode='x unified',
            xaxis_title='Année',
            yaxis_title='Nombre d\'évaluations',
            font=dict(size=11),
            height=450
        )
        
        return fig
    
    def area_chart_cumulative_users(self):
        """Diagramme d'aires du nombre cumulatif d'utilisateurs"""
        # Créer des données cumulatives
        self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        self.ratings['year_month'] = self.ratings['timestamp'].dt.to_period('M')
        
        users_by_month = self.ratings.groupby('year_month')['userId'].nunique().reset_index(name='new_users')
        users_by_month['cumulative_users'] = users_by_month['new_users'].cumsum()
        users_by_month['year_month'] = users_by_month['year_month'].astype(str)
        
        fig = px.area(
            users_by_month,
            x='year_month',
            y='cumulative_users',
            title='👥 Évolution Cumulative du Nombre d\'Utilisateurs',
            labels={'year_month': 'Mois', 'cumulative_users': 'Nombre cumulatif d\'utilisateurs'},
            color_discrete_sequence=['#e50914']
        )
        
        fig.update_layout(
            template='plotly_dark',
            hovermode='x unified',
            xaxis_title='Période',
            yaxis_title='Nombre cumulatif d\'utilisateurs',
            font=dict(size=11),
            height=400,
            xaxis_tickangle=-45
        )
        
        # Afficher moins de ticks sur l'axe X pour la lisibilité
        fig.update_xaxes(nticks=15)
        
        return fig
    
    def area_chart_average_rating_evolution(self):
        """Diagramme d'aires de l'évolution de la note moyenne"""
        self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        self.ratings['year'] = self.ratings['timestamp'].dt.year
        
        avg_rating_by_year = self.ratings.groupby('year')['rating'].agg(['mean', 'count']).reset_index()
        # Calculer une moyenne roulante pour lisser les variations
        avg_rating_by_year = avg_rating_by_year.sort_values('year')
        avg_rating_by_year['rolling_mean'] = avg_rating_by_year['mean'].rolling(window=3, min_periods=1).mean()

        fig = go.Figure()

        # Aire pour la moyenne roulante (smoothing)
        fig.add_trace(go.Scatter(
            x=avg_rating_by_year['year'],
            y=avg_rating_by_year['rolling_mean'],
            fill='tozeroy',
            name='Moyenne roulante (3 ans)',
            line=dict(color='#e50914', width=2),
            fillcolor='rgba(229,9,20,0.18)',
            hovertemplate='Année: %{x}<br>Moyenne roulante: %{y:.2f}<extra></extra>'
        ))

        # Points pour la moyenne annuelle
        fig.add_trace(go.Scatter(
            x=avg_rating_by_year['year'],
            y=avg_rating_by_year['mean'],
            mode='markers+lines',
            name='Note annuelle (moyenne)',
            line=dict(color='#ffffff', width=1, dash='dot'),
            marker=dict(size=6, color='#ffffff'),
            hovertemplate='Année: %{x}<br>Note: %{y:.2f}<br>Nb évaluations: %{customdata[0]}<extra></extra>',
            customdata=np.stack((avg_rating_by_year['count'],), axis=-1)
        ))

        fig.update_layout(
            title='⭐ Évolution de la Note Moyenne par Année',
            xaxis_title='Année',
            yaxis_title='Note moyenne',
            template='plotly_dark',
            hovermode='x unified',
            font=dict(size=12),
            height=520,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        return fig

    def area_chart_user_rating_distribution(self):
        """Distribution (aire/barre) du nombre d'évaluations par utilisateur

        Affiche à la fois l'histogramme des nombre d'évaluations par utilisateur
        et la courbe cumulative en pourcentage.
        """
        # Nombre d'évaluations par utilisateur
        user_counts = self.ratings.groupby('userId').size().reset_index(name='ratings_count')
        user_counts = user_counts.sort_values('ratings_count')

        # Histogramme (barres) + courbe cumulative
        counts = user_counts['ratings_count'].values
        bins = min(50, int(np.ceil(len(np.unique(counts)))))

        hist = np.histogram(counts, bins=bins)
        bin_edges = hist[1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_counts = hist[0]
        cumulative = np.cumsum(bin_counts) / np.sum(bin_counts) * 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Bar(
            x=bin_centers,
            y=bin_counts,
            name='Utilisateurs (count)',
            marker_color='#e50914',
            opacity=0.8
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=cumulative,
            mode='lines+markers',
            name='Cumulatif (%)',
            line=dict(color='#00CC96', width=2),
            marker=dict(size=6)
        ), secondary_y=True)

        fig.update_xaxes(title_text='Nombre d\'évaluations par utilisateur')
        fig.update_yaxes(title_text='Nombre d\'utilisateurs', secondary_y=False)
        fig.update_yaxes(title_text='Cumulatif (%)', secondary_y=True)

        fig.update_layout(
            title="👥 Distribution des Évaluations par Utilisateur",
            template='plotly_dark',
            height=480,
            hovermode='x unified',
        )

        return fig
    
    # ========================================================================
    # STATISTIQUES COMBINÉES
    # ========================================================================
    
    def get_summary_statistics(self):
        """Retourne un dictionnaire avec les statistiques clés"""
        stats = {
            'total_movies': len(self.movies),
            'total_ratings': len(self.ratings),
            'total_users': self.ratings['userId'].nunique(),
            'total_tags': len(self.tags),
            'avg_rating': self.ratings['rating'].mean(),
            'median_rating': self.ratings['rating'].median(),
            'std_rating': self.ratings['rating'].std(),
            'min_rating': self.ratings['rating'].min(),
            'max_rating': self.ratings['rating'].max(),
            'avg_ratings_per_movie': len(self.ratings) / len(self.movies),
            'avg_ratings_per_user': len(self.ratings) / self.ratings['userId'].nunique(),
            'genres_count': self.genre_data['genre_list'].nunique()
        }
        return stats
