# Utiliser l'image officielle legère de Python 
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Permettre l'affichage immédiat des instructions et les messages de log dans les journeaux de knative
ENV PYTHONUNBUFFERED=True

# Copier le code local dans l'image du conteneur.
# Définir le répertoire de travail dans le conteneur à /app
ENV APP_HOME=/app
WORKDIR $APP_HOME
COPY requirements.txt . 

# Installer les dépendances de production.
# Exécuter pip install pour les packages spécifiés dans requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exécuter le service web au démarrage du conteneur. Ici, nous utilisons le serveur web gunicorn.
# avec un processus worker et 8 threads.
# pour des environnements avec plusieurs coeurs CPU. augmenter le nombre de workers en conséquence.
# pour qu'il soit egal au nombre de coeurs disponibles. 
# le timeaout est reglé sur 0 pour désactiver les timeouts des workers pour permettre à Cloude Run de gérer les timeouts.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 App:app
