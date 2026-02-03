# MyTflix — Recommandation de films

Ce dépôt contient le projet MyTflix : un système de recommandation de films (MovieLens) avec visualisations statistiques et une interface Streamlit.

Contenu principal :
- `app.py` — application Streamlit
- `ml_model.py` — moteur de recommandation
- `statistics.py` — visualisations Plotly
- `movies.csv`, `ratings.csv`, `tags.csv` — datasets

Pour la documentation complète et les exemples d'utilisation, voir `README_ML.md`.

Lancer localement :
```bash
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Licence: MIT
# reco_movies