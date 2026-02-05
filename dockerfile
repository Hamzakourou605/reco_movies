# Utiliser une version légère de Python
FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le reste du code et les CSV
COPY . .

# Exposer le port utilisé par Streamlit
EXPOSE 8000

# Commande pour lancer l'application
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]