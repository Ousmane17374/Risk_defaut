import streamlit as st
import requests
import pandas as pd

# URL de l'API
API_URL = 'https://dgfsg-src-risk-credit-prediction-api.onrender.com/predire'

def envoyer_pour_prediction(donnees):
    """ Envoie les données à l'API pour obtenir une prédiction. """
    response = requests.post(f"{API_URL}/predire", json=donnees)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    

def main():
    st.title("Prédiction du Risque de Défaut de Crédit")
    st.write("Entrez les informations du client pour prédire le risque de défaut de crédit.")

    # Collecte des données utilisateur
    fichier = st.file_uploader("Limite de crédit", min_value=0)
    if fichier is not None:
        # Chargement des données
        donnees= pd.read_csv(fichier)
        # Affichage des données chargées
        st.write("Données chargées:")
        st.write(donnees)

        if st.button("prédire"):
            # prédire chaque ligne de données chargées
            predictions=[]
            for _, row in donnees.iterrows():
                donnees_api = row.to_dict()
                resultat = envoyer_pour_prediction(donnees_api)
                if resultat:
                    predictions.append(resultat)
                else:
                    st.error("Erreur lors de la prédiction.")
                    return
            # Affichage des résultats
            st.write("prédiction:")
            st.table(predictions)

        else :
            st.write("cliquer sur le bouton prédire pour obtenir les résultats.")


if __name__ == "__main__":
    main()

                 

    