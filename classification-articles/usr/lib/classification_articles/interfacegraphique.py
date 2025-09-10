import tkinter as tk
from tkinter import scrolledtext
from decision import Classifier

class AppGUI:
	"""
	Crée l'interface graphique pour l'application de prédiction de texte.
    
	Attributs :
        classifier (Classifier): L'instance de Classifier utilisée pour traiter le texte et faire des prédictions.
        fenetre (Tk): La fenêtre principale de l'application Tkinter.
	"""
	def __init__(self, classifier):
		"""
		Initialise l'interface graphique avec un Classifier.
        
		Paramètres :
		classifier (Classifier): Une instance de Classifier pour le traitement et la prédiction du texte.
		"""
		self.classifier = classifier
		self.setup()
		
	def setup(self):
		"""
		Configure les widgets de l'interface utilisateur et les affiche dans la fenêtre principale.
		"""
		# Fenetre principale
		self.window = tk.Tk()
		self.window.title("Classification d'articles")
		# Config zone de saisie 
		self.zone_texte = scrolledtext.ScrolledText(self.window, height=25, width=150)
		self.zone_texte.pack(fill=tk.BOTH, expand=True) # place le widget dans la fenetre en focntion de la taille de la fenetre
		
		# zone resultat classification
		self.zone_resultat = scrolledtext.ScrolledText(self.window, height=5, width=150, state=tk.DISABLED)
		
		# Bouton declenche prediction
		self.bouton = tk.Button(self.window, text="Classifier", command=self.classify)
		self.bouton.pack()
		
		self.zone_resultat.pack(fill=tk.BOTH, expand=True)# place le widget dans la fenetre en focntion de la taille de la fenetre
			
	def classify(self):
		"""
		Récupère le texte saisi par l'utilisateur, fait une prédiction et affiche le résultat.
		"""
		texte_saisi = self.zone_texte.get("1.0", "end-1c")
		
		if not texte_saisi.isspace() and texte_saisi :
			# Faire une prédiction
			categorie_predite = self.classifier.classer(texte_saisi)
			
			# permet de modifier la zone_resultat uniquement lors de l'appui du bouton
			self.zone_resultat.config(state=tk.NORMAL)
			self.zone_resultat.delete("1.0", tk.END)
			self.zone_resultat.insert(tk.END, 'Notre modèle de classification place votre texte dans la catégorie : ' + categorie_predite + ".")
			self.zone_resultat.config(state=tk.DISABLED)
		else:
			self.zone_resultat.config(state=tk.NORMAL)
			self.zone_resultat.delete("1.0", tk.END)
			self.zone_resultat.insert(tk.END, 'Saisissez du texte pour effectuer une classification.')
			self.zone_resultat.config(state=tk.DISABLED)
	
	def run(self):
		"""
		Lance l'interface graphique. Doit être appelée pour afficher la fenêtre et commencer l'interaction avec l'utilisateur.
		"""
		# demarrer interface
		self.window.mainloop()
	

if __name__ == "__main__":
	chemin_detect_novelty = '/usr/share/classification_articles/novelty_model.joblib'
	chemin_modele = '/usr/share/classification_articles/modele_classification.joblib'
	classifier_ = Classifier(chemin_modele, chemin_detect_novelty)
	app = AppGUI(classifier_)
	app.run()
