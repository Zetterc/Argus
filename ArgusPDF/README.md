# Visualisateur de Données Multi-échelles

Cette application Streamlit permet de visualiser des séries temporelles avec des échelles différentes sur un même graphique.

## Fonctionnalités

- Upload de fichiers Excel
- Visualisation automatique multi-échelles
- Statistiques descriptives
- Assignment manuel ou automatique des séries aux axes
- Interface utilisateur intuitive

## Installation

1. Installez les dépendances :
```bash
pip install -r requirements.txt
```

2. Lancez l'application :
```bash
streamlit run app.py
```

## Utilisation

1. Chargez votre fichier Excel (format .xlsx ou .xls)
2. Les séries sont automatiquement assignées aux axes selon leur ordre de grandeur
3. Utilisez l'option d'assignment manuel si vous souhaitez personnaliser les axes
4. Consultez les statistiques descriptives et le graphique interactif

## Format des données

- Le fichier Excel doit avoir une colonne d'index temporel
- Chaque colonne représente une série de données
- Les données manquantes sont gérées automatiquement
