### 1. Projet: Prédiction du risque de défaut de crédit
Ce projet vise à développer un modèle de machine learning capable de prédire le risque de défaut de crédit des clients en utilisant des données historiques. Le modèle sera entraîné sur un ensemble de données comprenant diverses caractéristiques des clients et leurs comportements financiers passés.
### 2. Créer une API pour le modèle (Fast API)
- Utimiser test/test_request.py pour tester l'appel à l'API en local
### 3. Configurer Google cloud
- Créer un nouveau projet
- Activer l'API Cloud Run et l'API Cloud Build
### 4. Installer et Initialiser Google Cloud SDK
-[Installer Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- Initialiser avec gcloud init
### 5. Installer Docker & Docker Hub
- [Installer Docker](https://docs.docker.com/engine/install/)
### 6. Créer le fichier requirements.txt
pipe freeze > requirements.txt
### 7. Dockerfile, requirements.txt, .Dockerignore
-[Guide de containérisation sur Google Cloud] (https://cloud.google.com/run/docs/quickstarts/build-and-deploy#containerizing)

[Code Dockerfile](https://fastapi.tiangolo.com/deployment/docker/)

### 8. Construction et déployement dans le cloud

Ancienne méthode gcr.io
### Définir les variables
- PROJECT_ID=Prédiction_risque_defaut_crédit #ID du projet GCP
- IMAGE_NAME=credit-risk-prediction  # Nom de l'image Docker 
- REGION="your region"               # Remplacer par la région GCP de votre choix

### Soumettre le build de l'image Docker à Google Container Registry
gcloud builds submit --tag gcr.io/${PROJECT_ID}/${IMAGE_NAME}

### Déployer l'image sur Google Cloud Run
gcloudd run deploy --image gcr.io/${PROJECT_ID}/${IMAGE_NAME} --platform managed --region ${REGION}

gcloud build submit --tag gcr.io/testapi-420317/index
gcloud run deploy --image gcr.io/testapi-420317/index --platform managed

