# Utiliser l'image officielle legère de Python 
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Permettre l'affichage immédiat des instructions et les messages de log dans les journeaux de knative
ENV PYTHONUNBUFFERED=True

# 🔹 AJOUT ICI (PORT par défaut pour local & Cloud Run)
ENV PORT=8080
EXPOSE 8080

# Copier le code local dans l'image du conteneur.
# Définir le répertoire de travail dans le conteneur à /app
ENV APP_HOME=/app
WORKDIR $APP_HOME

COPY requirements.txt . 

# Installer les dépendances de production.
# Exécuter pip install pour les packages spécifiés dans requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exécuter le service web au démarrage du conteneur.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 App:app
