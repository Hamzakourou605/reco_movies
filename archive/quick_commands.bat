@echo off
REM ============================================================================
REM MyTflix - Commandes Rapides
REM Fichier batch pour accès rapide aux fonctionnalités
REM ============================================================================

:menu
cls
echo.
echo ============================================================================
echo              🎬 MyTflix - Commandes Rapides et Utiles 🎬
echo ============================================================================
echo.
echo 📺 LANCER L'APPLICATION
echo   1) Streamlit (Interface web)
echo   2) Tests ML (Vérifier système)
echo   3) Guide Utilisation (Terminal)
echo.
echo 📚 DOCUMENTATION
echo   4) Lire README_ML.md
echo   5) Lire FEATURES_RESUME.md
echo   6) Lire DEVELOPPEMENT_COMPLET.md
echo   7) Lire PROJECT_STRUCTURE.md
echo.
echo 🐍 PYTHON REPL (Recommandations en Terminal)
echo   8) Python Shell Interactif
echo.
echo 🔧 UTILITAIRES
echo   9) Vérifier Modèle ML
echo  10) Installer Dépendances
echo  11) Nettoyer Cache Python
echo.
echo  0) Quitter
echo.
echo ============================================================================

set /p choice=Choisir une option: 

if "%choice%"=="1" (
    cls
    echo.
    echo 🚀 Lancement Streamlit...
    echo    URL: http://localhost:8501
    echo    Appuyez sur Ctrl+C pour arrêter
    echo.
    call venv\Scripts\activate.bat
    streamlit run app.py
    goto menu
)

if "%choice%"=="2" (
    cls
    echo.
    echo 🧪 Exécution des Tests ML...
    echo.
    call venv\Scripts\activate.bat
    python test_ml_recommendations.py
    pause
    goto menu
)

if "%choice%"=="3" (
    cls
    echo.
    echo 📊 Guide Utilisation...
    echo.
    call venv\Scripts\activate.bat
    python GUIDE_ML_RECOMMENDATIONS.py
    pause
    goto menu
)

if "%choice%"=="4" (
    start README_ML.md
    goto menu
)

if "%choice%"=="5" (
    start FEATURES_RESUME.md
    goto menu
)

if "%choice%"=="6" (
    start DEVELOPPEMENT_COMPLET.md
    goto menu
)

if "%choice%"=="7" (
    start PROJECT_STRUCTURE.md
    goto menu
)

if "%choice%"=="8" (
    cls
    echo.
    echo 🐍 Python Shell Interactif
    echo.
    echo # Exemples d'utilisation:
    echo # from ml_model import MovieRecommender
    echo # recommender = MovieRecommender.load('recommender_model.pkl')
    echo # recs = recommender.recommend_by_multiple_genres(['Action'], n=10)
    echo # print(recs[['title', 'avg_rating']])
    echo.
    call venv\Scripts\activate.bat
    python
    goto menu
)

if "%choice%"=="9" (
    cls
    echo.
    echo 🔍 Vérification du Modèle ML...
    echo.
    call venv\Scripts\activate.bat
    python -c "from ml_model import MovieRecommender; r = MovieRecommender.load('recommender_model.pkl'); print(f'✅ Modèle OK!'); print(f'   Films: {len(r.movies)}'); print(f'   Genres: {len(r.get_all_genres())}'); print(f'   Évaluations: {len(r.ratings)}')"
    pause
    goto menu
)

if "%choice%"=="10" (
    cls
    echo.
    echo 📦 Installation des Dépendances...
    echo.
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
    echo.
    echo ✅ Dépendances à jour!
    pause
    goto menu
)

if "%choice%"=="11" (
    cls
    echo.
    echo 🧹 Nettoyage du Cache...
    echo.
    for /d /r . %%d in (__pycache__) do @if exist "%%d" (
        echo Suppression: %%d
        rmdir /s /q "%%d"
    )
    echo.
    echo ✅ Cache nettoyé!
    pause
    goto menu
)

if "%choice%"=="0" (
    echo.
    echo 👋 Au revoir!
    echo.
    exit /b 0
)

echo.
echo ❌ Option invalide!
pause
goto menu
