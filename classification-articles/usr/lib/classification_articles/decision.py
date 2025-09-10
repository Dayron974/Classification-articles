from joblib import load
import spacy
import unicodedata
nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])

class Classifier:
	"""
	Gère le traitement du texte et la prédiction en utilisant deux modèles de machine learning.
	
	Attributs :
        modele (Any): Modèle de prédiction de catégories chargé.
        detect_novelty (Any): Modèle de détection d'outliers chargé.
        nlp (Language): Instance de Spacy pour le traitement du texte.
	"""
	def __init__(self, model_path, novelty_model_path):
		"""
		Initialise le TextProcessor avec deux modèles.
        
		Parameters :
		model_path (str): Chemin vers le fichier du modèle de prédiction de catégories à charger.
		novelty_model_path (str): Chemin vers le fichier du modèle de détection de textes n'appartenant à aucune catégorie. 
		"""
		self.modele = self.load_model(model_path)
		self.detect_novelty = self.load_model(novelty_model_path)
		self.nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
		
	def load_model(self, model_path):
		"""
		Charge le modèle de prédiction à partir du chemin fourni.
		
		Parameters :
		model_path (str): Le chemin vers le fichier du modèle.

		Returns:
		Any: Le modèle de prédiction chargé.

		Raises:
		FileNotFoundError: Si le fichier modèle spécifié n'existe pas.
		Exception: Pour toute autre erreur inattendue lors du chargement du modèle.
		"""
		try:
			modele = load(model_path)
			return modele
		# exception levée si le fichier n'existe pas
		except FileNotFoundError:
			print("Erreur : Le modèle 'modele_classification.jolib' n'a pas été trouvé.")
			sys.exit(1)
		# exception générale
		except Exception as e:
			print(f"Erreur inattendue lors du chargement du modèle : {e}.")
			sys.exit(1)
			
	def traiter_texte(self, text):
		"""
		Utilise Spacy pour retirer les mots stop et la ponctuation du texte donné et lemmatiser le texte restant.

		Parameters:
		text (str): Texte à traiter.

		Returns:
		str: Texte traité, avec les mots stop et la ponctuation retirés et les mots restants lemmatisés.
		"""
		# mots sans interet
		custom_stop = ['par', 'sur', 'si', 'je', 'quelles', 'ils', 'elle', 'u', 'ton', 'après', 'avec', 'ses', 'vôtre',
		 'dans', 'jusqu', 'comme', 'mes', 'les', 'son', 'suivant', 'du', 'malgré', 'chez', 'jusque', 'tes', 'selon',
		 'sous', 'laquelle', 'tien', 'mon', 'ce', 'ta', 'voilà', 'et', 'lesquels', 'leurs', 'au', 'quoi', 'à',
		 'ceci', 'des', 'quelle', 'depuis', 'la', 'de', 'permettre', 'sans', 'nos', 'lors', 'leur', 'aux', 'lequel',
		 'parmi', 'en', 'vers', 'ma', 'lesquelles', 'toute', 'le', 'sien', 'ces', 'quels', 'avant', 'sauf',
		 'non', 'possibilité', 'travailler', 'votre', 'pour', 'dont', 'tout', 'toutes', 'hors', 'sa', 'tous',
		 'tu', 'nôtre', 'vos', 'vous', 'un', 'entre', 'quel', 'ça', 'notre', 'revoir', 'mien', 'soit',
		 'pendant', 'elles', 'faire', 'il', 'nous',
		 'cela', 'une', 'qui',  'l', 's', 'd', 'qu'
		]
		doc = nlp(text)
		# lemmatise les tokens du texte
		tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and token.text not in custom_stop]
		return " ".join(tokens)
	
	def retirer_accents(self, texte):
		"""
		Retire les accents du texte donné.

		Parameters:
		texte (str): Texte duquel retirer les accents.

		Returns:
		str: Texte sans accents.
		"""
		texte_sans_accents = ''.join((c for c in unicodedata.normalize('NFD', texte) if unicodedata.category(c) != 'Mn'))
		return texte_sans_accents

	def classer(self, texte):
		"""
		Prédit la catégorie du texte fourni en utilisant les modèles chargés.

		Parameters:
		texte (str): Texte pour lequel faire une prédiction.

		Returns:
		str: Résultat de la prédiction des modèles.
		"""
		# applique les traitements à l'article 
		processed_text = self.retirer_accents(texte)
		processed_text = processed_text.lower()
		processed_text = self.traiter_texte(processed_text)
		
		# Vérifie si le texte est une nouveauté
		oc_svm_result = self.detect_novelty.predict([processed_text])
		
		proba = self.modele.predict_proba([processed_text])[0]
		predicted_class = self.modele.predict([processed_text])[0]
		confidence = max(proba)

		if confidence < 0.75:
			if oc_svm_result == -1:
				return 'Autre'  # textes non reconnus
			else:
				return self.modele.predict([processed_text])[0]  # Confiance non suffisante, mais pas une nouveauté
		else:
			return self.modele.predict([processed_text])[0]  # Confiance suffisante, classe prédite par SVC
	
