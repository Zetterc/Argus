import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import warnings
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(layout="wide")

# Réinitialiser la session state pour les nouvelles assignations
if 'reset_done' not in st.session_state:
    for key in list(st.session_state.keys()):
        if key.endswith('_last'):
            del st.session_state[key]
    st.session_state['reset_done'] = True

def load_data(uploaded_file):
    """Charge les données depuis un fichier Excel et les groupe par préfixe"""
    if uploaded_file is not None:
        excel = pd.ExcelFile(uploaded_file)
        sheets = excel.sheet_names
        
        # Grouper les feuilles par préfixe
        prefix_groups = {}
        for sheet in sheets:
            prefix = sheet.split('_')[0]
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(sheet)
            
        # Charger les données pour chaque feuille
        data_dict = {sheet: pd.read_excel(uploaded_file, sheet_name=sheet, index_col=0, parse_dates=True) 
                    for sheet in sheets}
                    
        return data_dict, prefix_groups
    return None, None

def analyze_series(series, name):
    """Analyse une série temporelle et retourne ses caractéristiques"""
    # Nettoyer la série
    series = series.dropna()
    if len(series) < 4:  # Besoin d'un minimum de points
        return None
    
    stats_dict = {}
    
    # 1. Analyse de stationnarité
    try:
        adf_result = adfuller(series, regression='ct')
        stats_dict['stationary'] = adf_result[1] < 0.05  # p-value < 0.05 => stationnaire
    except:
        stats_dict['stationary'] = None
    
    # 2. Analyse de tendance et saisonnalité
    try:
        # Décomposition de la série
        decomposition = seasonal_decompose(series, period=min(len(series)//2, 12), extrapolate_trend='freq')
        
        # Tendance
        trend = decomposition.trend
        trend_clean = trend.dropna()
        if len(trend_clean) > 1:
            # Calculer la pente de la tendance
            x = np.arange(len(trend_clean))
            z = np.polyfit(x, trend_clean, 1)
            stats_dict['trend_slope'] = z[0]
        else:
            stats_dict['trend_slope'] = 0
            
        # Force de la saisonnalité
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        seasonal_strength = 1 - np.nanvar(residual) / np.nanvar(seasonal + residual)
        stats_dict['seasonal_strength'] = seasonal_strength
        
    except:
        stats_dict['trend_slope'] = 0
        stats_dict['seasonal_strength'] = 0
    
    # 3. Analyse statistique
    try:
        # Asymétrie et Kurtosis
        stats_dict['skewness'] = stats.skew(series)
        stats_dict['kurtosis'] = stats.kurtosis(series)
        
        # Volatilité
        returns = series.pct_change().dropna()
        stats_dict['volatility'] = returns.std()
        
        # Magnitude
        stats_dict['magnitude'] = np.log10(abs(series.mean())) if series.mean() != 0 else 0
        
    except:
        stats_dict['skewness'] = 0
        stats_dict['kurtosis'] = 0
        stats_dict['volatility'] = 0
        stats_dict['magnitude'] = 0
    
    return stats_dict

def get_nice_scale(min_val, max_val, include_zero=False):
    """Calcule une échelle 'propre' pour les valeurs données."""
    if min_val == max_val:
        if max_val == 0:
            return -1, 1
        return max_val * 0.9, max_val * 1.1

    # Forcer l'inclusion de zéro si demandé
    if include_zero:
        min_val = min(0, min_val)
        max_val = max(0, max_val)
    
    # Ajouter une marge de 20%
    range_val = max_val - min_val
    min_val = min_val - (range_val * 0.1)
    max_val = max_val + (range_val * 0.1)
    
    # Arrondir à des valeurs "propres"
    def round_to_nice(x, round_up=True):
        abs_x = abs(x)
        sign = 1 if x >= 0 else -1
        
        if abs_x <= 5:
            step = 1
        elif abs_x <= 10:
            step = 2
        elif abs_x <= 100:
            step = 10
        else:
            step = 50
        
        if round_up:
            return sign * (((abs_x // step) + 1) * step)
        else:
            return sign * ((abs_x // step) * step)
    
    return round_to_nice(min_val, False), round_to_nice(max_val, True)

def calculate_scale_ratio(series):
    """Calcule le ratio d'échelle d'une série."""
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 1
    
    abs_max = abs(clean_series).max()
    abs_min = abs(clean_series[clean_series != 0]).min() if any(clean_series != 0) else abs_max
    
    return abs_max / abs_min if abs_min > 0 else 1

def get_column_axis(column_name):
    """Détermine l'axe d'une colonne basé sur son nom."""
    name = str(column_name).lower()
    if "(l)" in name:
        return "left"
    elif "(r)" in name:
        return "right"
    return None  # Pour les colonnes sans indication

def assign_axis(data):
    """
    Assigne les colonnes aux axes en utilisant une combinaison de marqueurs et de plages de valeurs.
    1. Utilise d'abord les marqueurs (L) et (R) dans les noms de colonnes
    2. Pour les colonnes non marquées, utilise l'analyse des plages de valeurs
    """
    left_cols = []
    right_cols = []
    unassigned_cols = []
    
    # Première étape : assigner les colonnes marquées
    for col in data.columns:
        axis = get_column_axis(col)
        if axis == "left":
            left_cols.append(col)
        elif axis == "right":
            right_cols.append(col)
        else:
            unassigned_cols.append(col)
    
    # Si toutes les colonnes sont assignées, on a fini
    if not unassigned_cols:
        return left_cols, right_cols
    
    # Deuxième étape : analyser les plages de valeurs pour les colonnes non assignées
    ranges = {}
    for col in unassigned_cols:
        series = data[col].dropna()
        if len(series) == 0:
            continue
        ranges[col] = {
            'min': series.min(),
            'max': series.max(),
            'range': series.max() - series.min(),
            'mean': series.mean()
        }
    
    if not ranges:
        # Si pas de données valides, distribuer équitablement
        mid = len(unassigned_cols) // 2
        return left_cols + unassigned_cols[:mid], right_cols + unassigned_cols[mid:]
    
    # Trouver des groupes naturels basés sur les plages de valeurs
    sorted_by_mean = sorted(ranges.items(), key=lambda x: x[1]['mean'])
    
    if len(sorted_by_mean) == 1:
        # S'il n'y a qu'une colonne, la mettre du côté qui en a le moins
        col = sorted_by_mean[0][0]
        if len(left_cols) <= len(right_cols):
            left_cols.append(col)
        else:
            right_cols.append(col)
    else:
        # Trouver le plus grand écart entre les moyennes
        max_gap = 0
        gap_idx = 0
        for i in range(len(sorted_by_mean)-1):
            current_mean = sorted_by_mean[i][1]['mean']
            next_mean = sorted_by_mean[i+1][1]['mean']
            gap = abs(next_mean - current_mean)
            if gap > max_gap:
                max_gap = gap
                gap_idx = i + 1
        
        # Assigner les colonnes en fonction de l'écart trouvé
        for i, (col, _) in enumerate(sorted_by_mean):
            if i < gap_idx:
                left_cols.append(col)
            else:
                right_cols.append(col)
    
    return left_cols, right_cols

def is_bar_column(column_name):
    """Vérifie si une colonne doit être affichée en barres."""
    return "(bar)" in str(column_name).lower()

def create_multi_axis_plot(data, primary_cols, secondary_cols, title=None, 
                      primary_bars=None, secondary_bars=None,
                      primary_stack=False, secondary_stack=False):
    """Crée un graphique avec deux axes Y."""
    fig = go.Figure()
    
    # Initialiser les barres si non spécifiées
    if primary_bars is None:
        primary_bars = []
    if secondary_bars is None:
        secondary_bars = []
    
    def create_waterfall_bars(bars, yaxis):
        """Crée des barres en cascade où chaque barre commence où la précédente se termine"""
        if not bars:
            return
            
        # Pour chaque point temporel, trier les barres par valeur
        sorted_bars_by_time = {}
        for idx in data.index:
            values = [(col, data.loc[idx, col]) for col in bars]
            sorted_bars_by_time[idx] = sorted(values, key=lambda x: x[1])
        
        # Créer les barres en cascade
        for col in bars:
            base = []
            values = []
            
            for idx in data.index:
                sorted_vals = sorted_bars_by_time[idx]
                col_pos = next(i for i, (c, _) in enumerate(sorted_vals) if c == col)
                
                if col_pos == 0:
                    # Première barre : commence à 0
                    base.append(0)
                    values.append(data.loc[idx, col])
                else:
                    # Autres barres : commencent où la précédente finit
                    prev_val = sorted_vals[col_pos - 1][1]
                    base.append(prev_val)
                    values.append(data.loc[idx, col] - prev_val)
            
            fig.add_trace(go.Bar(
                name=f"{col} ({yaxis})",
                x=data.index,
                y=values,
                base=base,
                yaxis='y' if yaxis == 'gauche' else 'y2',
                offsetgroup=0 if yaxis == 'gauche' else 1
            ))
    
    # Ajouter les séries de l'axe principal (gauche)
    if primary_stack and len(primary_bars) > 1:
        create_waterfall_bars(primary_bars, 'gauche')
    else:
        for col in primary_cols:
            if col in primary_bars:
                fig.add_trace(go.Bar(
                    name=f"{col} (gauche)",
                    x=data.index,
                    y=data[col],
                    yaxis='y',
                    offsetgroup=0
                ))
            else:
                fig.add_trace(go.Scatter(
                    name=f"{col} (gauche)",
                    x=data.index,
                    y=data[col],
                    yaxis='y',
                    mode='lines+markers'
                ))
    
    # Ajouter les séries de l'axe secondaire (droit)
    if secondary_stack and len(secondary_bars) > 1:
        create_waterfall_bars(secondary_bars, 'droit')
    else:
        for col in secondary_cols:
            if col in secondary_bars:
                fig.add_trace(go.Bar(
                    name=f"{col} (droit)",
                    x=data.index,
                    y=data[col],
                    yaxis='y2',
                    offsetgroup=1
                ))
            else:
                fig.add_trace(go.Scatter(
                    name=f"{col} (droit)",
                    x=data.index,
                    y=data[col],
                    yaxis='y2',
                    mode='lines+markers'
                ))
    
    # Mise à jour de la mise en page
    fig.update_layout(
        template="plotly",  # Pour avoir les mêmes couleurs que dans Streamlit
        title=title,
        xaxis=dict(
            title='Date',
            showgrid=True
        ),
        yaxis=dict(
            title='Axe principal',
            titlefont=dict(color="#1f77b4"),
            tickfont=dict(color="#1f77b4"),
            showgrid=True,
            side='left'
        ),
        yaxis2=dict(
            title='Axe secondaire',
            titlefont=dict(color="#ff7f0e"),
            tickfont=dict(color="#ff7f0e"),
            anchor="x",
            overlaying="y",
            side="right",
            showgrid=True
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",    # Changé de "bottom" à "top"
            y=-0.1,           # Déplacé sous le graphique
            xanchor="center", # Centré horizontalement
            x=0.5            # Au milieu
        ),
        barmode='relative',
        margin=dict(t=30, b=50),  # Réduit la marge du haut, augmenté celle du bas pour la légende
        height=650,              # Hauteur du graphique en pixels
        width=1000              # Largeur du graphique en pixels
    )
    
    return fig

def calculate_trend(series):
    """Calcule la tendance d'une série (croissante/décroissante/stable)"""
    # Ignorer les valeurs manquantes
    series = series.dropna()
    if len(series) < 2:
        return "N/A"
        
    # Régression linéaire simple
    x = np.arange(len(series))
    slope = np.polyfit(x, series, 1)[0]
    
    # Déterminer la tendance
    if abs(slope) < 0.001:  # Seuil pour "stable"
        return "→ Stable"
    elif slope > 0:
        return "↗ Croissante"
    else:
        return "↘ Décroissante"

def display_statistics(data, sheet_name):
    """Affiche les statistiques descriptives des séries"""
    with st.expander("📊 Statistiques descriptives"):
        # Créer un DataFrame pour les statistiques
        stats_dict = {}
        for col in data.columns:
            series = data[col].dropna()
            if len(series) == 0:
                continue
                
            # Calculer les variations
            variation = series.iloc[-1] - series.iloc[0] if len(series) > 1 else None
            var_pct = (series.iloc[-1] / series.iloc[0] - 1) if len(series) > 1 and series.iloc[0] != 0 else None
            
            stats_dict[col] = {
                'Dernier point': series.iloc[-1] if len(series) > 0 else None,
                'Minimum': series.min(),
                'Maximum': series.max(),
                'Moyenne': series.mean(),
                'Variation absolue': variation,
                'Variation %': var_pct,
                'Tendance': calculate_trend(series)
            }
        
        # Convertir en DataFrame et transposer pour avoir les colonnes en lignes
        stats_df = pd.DataFrame(stats_dict)
        
        # Formater les nombres
        for col in stats_df.columns:
            for idx in stats_df.index:
                if idx != 'Tendance':  # Ne pas formater la colonne tendance
                    val = stats_df.loc[idx, col]
                    if pd.isna(val):
                        stats_df.loc[idx, col] = "N/A"
                    elif isinstance(val, (int, float)):
                        if abs(val) >= 1000000:
                            stats_df.loc[idx, col] = f"{val:,.0f}"
                        elif abs(val) >= 1:
                            stats_df.loc[idx, col] = f"{val:,.2f}"
                        elif idx == 'Variation %':
                            stats_df.loc[idx, col] = f"{val:.2%}"
                        else:
                            stats_df.loc[idx, col] = f"{val:.2f}"
        
        # Afficher le DataFrame
        st.dataframe(stats_df, use_container_width=True)

def export_to_pdf(data, primary_cols, secondary_cols, title=None, 
                 primary_bars=None, secondary_bars=None,
                 primary_stack=False, secondary_stack=False,
                 comment=None):
    """Export le graphique en PDF."""
    # Créer le nom de fichier avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"graphique_{timestamp}.pdf"
    output_pdf = os.path.join(os.path.dirname(__file__), pdf_filename)
    
    # Créer une nouvelle figure avec un espace pour le commentaire
    fig = plt.figure(figsize=(15, 10))  # Augmenter la hauteur pour le commentaire
    
    # Créer deux sous-plots : un pour le graphique, un pour le commentaire
    gs = plt.GridSpec(2, 1, height_ratios=[4, 1])
    
    # Subplot pour le graphique
    ax_plot = fig.add_subplot(gs[0])
    ax2 = ax_plot.twinx()
    
    # Calculer la largeur optimale des barres
    n_points = len(data)
    bar_width = min(50.0, 500/n_points)
    
    def create_waterfall_bars(ax, bars, data):
        if not bars:
            return
        sorted_bars_by_time = {}
        for idx in data.index:
            values = [(col, data.loc[idx, col]) for col in bars]
            sorted_bars_by_time[idx] = sorted(values, key=lambda x: x[1])
        
        for col in bars:
            bases = []
            values = []
            for idx in data.index:
                sorted_vals = sorted_bars_by_time[idx]
                col_pos = next(i for i, (c, _) in enumerate(sorted_vals) if c == col)
                if col_pos == 0:
                    bases.append(0)
                    values.append(data.loc[idx, col])
                else:
                    prev_val = sorted_vals[col_pos - 1][1]
                    bases.append(prev_val)
                    values.append(data.loc[idx, col] - prev_val)
            ax.bar(data.index, values, bottom=bases, label=f"{col}", 
                  alpha=0.7, width=bar_width)
    
    # Tracer les séries
    if primary_stack and len(primary_bars) > 1:
        create_waterfall_bars(ax_plot, primary_bars, data)
    else:
        for col in primary_cols:
            legend_label = f"{col} (gauche)"
            if col in primary_bars:
                ax_plot.bar(data.index, data[col], label=legend_label, 
                           alpha=0.7, width=bar_width)
            else:
                ax_plot.plot(data.index, data[col], marker='o', label=legend_label)
    
    if secondary_stack and len(secondary_bars) > 1:
        create_waterfall_bars(ax2, secondary_bars, data)
    else:
        for col in secondary_cols:
            legend_label = f"{col} (droite)"
            if col in secondary_bars:
                ax2.bar(data.index, data[col], label=legend_label, 
                       alpha=0.7, width=bar_width)
            else:
                ax2.plot(data.index, data[col], marker='s', label=legend_label, linestyle='--')
    
    # Configurer les légendes et labels
    ax_plot.set_xlabel('Date')
    
    # Combiner les légendes des deux axes
    lines1, labels1 = ax_plot.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Créer une seule légende combinée à l'extérieur
    if lines1 or lines2:
        ax_plot.legend(lines1 + lines2, labels1 + labels2, 
                    bbox_to_anchor=(1.15, 1), loc='upper left')
    
    if title:
        plt.title(title)
    
    # Ajouter le commentaire si présent
    if comment:
        ax_comment = fig.add_subplot(gs[1])
        ax_comment.axis('off')  # Cacher les axes
        ax_comment.text(0, 0.5, f"Commentaires :\n{comment}", 
                       wrap=True, va='center', fontsize=10,
                       bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    
    # Ajuster la mise en page
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    # Sauvegarder en PDF
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_pdf

def export_all_to_pdf(data_dict, prefix_groups):
    """Export tous les graphiques dans un seul PDF multi-pages."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"tous_les_graphiques_{timestamp}.pdf"
    output_pdf = os.path.join(os.path.dirname(__file__), pdf_filename)
    
    with PdfPages(output_pdf) as pdf:
        for prefix, sheet_names in prefix_groups.items():
            for sheet_name in sheet_names:
                data = data_dict[sheet_name]
                
                # Utiliser les paramètres actuels du graphique depuis session_state
                primary_cols = st.session_state.get(f"primary_{sheet_name}", [])
                secondary_cols = st.session_state.get(f"secondary_{sheet_name}", [])
                
                # Récupérer les barres sélectionnées
                primary_bars = []
                secondary_bars = []
                for col in primary_cols:
                    if st.session_state.get(f"bar_primary_{sheet_name}_{col}", False):
                        primary_bars.append(col)
                for col in secondary_cols:
                    if st.session_state.get(f"bar_secondary_{sheet_name}_{col}", False):
                        secondary_bars.append(col)
                
                # Récupérer le mode d'empilement
                primary_stack = st.session_state.get(f"stack_primary_{sheet_name}", "Groupées") == "En cascade"
                secondary_stack = st.session_state.get(f"stack_secondary_{sheet_name}", "Groupées") == "En cascade"
                
                # Récupérer le commentaire
                comment = st.session_state.get(f"comment_{sheet_name}", "")
                
                # Créer une nouvelle figure avec espace pour le commentaire
                fig = plt.figure(figsize=(15, 10))
                
                # Créer deux sous-plots : un pour le graphique, un pour le commentaire
                gs = plt.GridSpec(2, 1, height_ratios=[4, 1])
                
                # Subplot pour le graphique
                ax1 = plt.subplot(gs[0])
                ax2 = ax1.twinx()
                
                # Calculer la largeur optimale des barres
                n_points = len(data)
                bar_width = min(50.0, 500/n_points)
                
                def create_waterfall_bars(ax, bars, data):
                    if not bars:
                        return
                    sorted_bars_by_time = {}
                    for idx in data.index:
                        values = [(col, data.loc[idx, col]) for col in bars]
                        sorted_bars_by_time[idx] = sorted(values, key=lambda x: x[1])
                    
                    for col in bars:
                        bases = []
                        values = []
                        for idx in data.index:
                            sorted_vals = sorted_bars_by_time[idx]
                            col_pos = next(i for i, (c, _) in enumerate(sorted_vals) if c == col)
                            if col_pos == 0:
                                bases.append(0)
                                values.append(data.loc[idx, col])
                            else:
                                prev_val = sorted_vals[col_pos - 1][1]
                                bases.append(prev_val)
                                values.append(data.loc[idx, col] - prev_val)
                        ax.bar(data.index, values, bottom=bases, label=f"{col}", 
                              alpha=0.7, width=bar_width)
                
                # Tracer les séries
                if primary_stack and len(primary_bars) > 1:
                    create_waterfall_bars(ax1, primary_bars, data)
                else:
                    for col in primary_cols:
                        legend_label = f"{col} (gauche)"
                        if col in primary_bars:
                            ax1.bar(data.index, data[col], label=legend_label, 
                                   alpha=0.7, width=bar_width)
                        else:
                            ax1.plot(data.index, data[col], marker='o', label=legend_label)
                
                if secondary_stack and len(secondary_bars) > 1:
                    create_waterfall_bars(ax2, secondary_bars, data)
                else:
                    for col in secondary_cols:
                        legend_label = f"{col} (droite)"
                        if col in secondary_bars:
                            ax2.bar(data.index, data[col], label=legend_label, 
                                   alpha=0.7, width=bar_width)
                        else:
                            ax2.plot(data.index, data[col], marker='s', label=legend_label, linestyle='--')
                
                # Configuration finale
                ax1.set_xlabel('Date')
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                if lines1 or lines2:
                    ax1.legend(lines1 + lines2, labels1 + labels2, 
                             bbox_to_anchor=(1.15, 1), loc='upper left')
                
                plt.title(f"Données de {sheet_name}")
                
                # Ajouter le commentaire si présent
                if comment:
                    ax_comment = plt.subplot(gs[1])
                    ax_comment.axis('off')  # Cacher les axes
                    ax_comment.text(0, 0.5, f"Commentaires :\n{comment}", 
                                wrap=True, va='center', fontsize=10,
                                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
                
                plt.tight_layout()
                plt.subplots_adjust(right=0.85)
                
                # Sauvegarder la page
                pdf.savefig(bbox_inches='tight')
                plt.close()
    
    return output_pdf

def export_selected_to_pdf(data_dict, selected_sheets):
    """Export les graphiques sélectionnés dans un seul PDF multi-pages."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"selection_graphiques_{timestamp}.pdf"
    output_pdf = os.path.join(os.path.dirname(__file__), pdf_filename)
    
    with PdfPages(output_pdf) as pdf:
        for sheet_name in selected_sheets:
            data = data_dict[sheet_name]
            
            # Utiliser les paramètres actuels du graphique depuis session_state
            primary_cols = st.session_state.get(f"primary_{sheet_name}", [])
            secondary_cols = st.session_state.get(f"secondary_{sheet_name}", [])
            
            # Récupérer les barres sélectionnées
            primary_bars = []
            secondary_bars = []
            for col in primary_cols:
                if st.session_state.get(f"bar_primary_{sheet_name}_{col}", False):
                    primary_bars.append(col)
            for col in secondary_cols:
                if st.session_state.get(f"bar_secondary_{sheet_name}_{col}", False):
                    secondary_bars.append(col)
            
            # Récupérer le mode d'empilement
            primary_stack = st.session_state.get(f"stack_primary_{sheet_name}", "Groupées") == "En cascade"
            secondary_stack = st.session_state.get(f"stack_secondary_{sheet_name}", "Groupées") == "En cascade"
            
            # Récupérer le commentaire
            comment = st.session_state.get(f"comment_{sheet_name}", "")
            
            # Créer une nouvelle figure avec espace pour le commentaire
            fig = plt.figure(figsize=(15, 10))
            
            # Créer deux sous-plots : un pour le graphique, un pour le commentaire
            gs = plt.GridSpec(2, 1, height_ratios=[4, 1])
            
            # Subplot pour le graphique
            ax1 = plt.subplot(gs[0])
            ax2 = ax1.twinx()
            
            # Calculer la largeur optimale des barres
            n_points = len(data)
            bar_width = min(50.0, 500/n_points)
            
            def create_waterfall_bars(ax, bars, data):
                if not bars:
                    return
                sorted_bars_by_time = {}
                for idx in data.index:
                    values = [(col, data.loc[idx, col]) for col in bars]
                    sorted_bars_by_time[idx] = sorted(values, key=lambda x: x[1])
                
                for col in bars:
                    bases = []
                    values = []
                    for idx in data.index:
                        sorted_vals = sorted_bars_by_time[idx]
                        col_pos = next(i for i, (c, _) in enumerate(sorted_vals) if c == col)
                        if col_pos == 0:
                            bases.append(0)
                            values.append(data.loc[idx, col])
                        else:
                            prev_val = sorted_vals[col_pos - 1][1]
                            bases.append(prev_val)
                            values.append(data.loc[idx, col] - prev_val)
                    ax.bar(data.index, values, bottom=bases, label=f"{col}", 
                          alpha=0.7, width=bar_width)
            
            # Tracer les séries
            if primary_stack and len(primary_bars) > 1:
                create_waterfall_bars(ax1, primary_bars, data)
            else:
                for col in primary_cols:
                    legend_label = f"{col} (gauche)"
                    if col in primary_bars:
                        ax1.bar(data.index, data[col], label=legend_label, 
                               alpha=0.7, width=bar_width)
                    else:
                        ax1.plot(data.index, data[col], marker='o', label=legend_label)
            
            if secondary_stack and len(secondary_bars) > 1:
                create_waterfall_bars(ax2, secondary_bars, data)
            else:
                for col in secondary_cols:
                    legend_label = f"{col} (droite)"
                    if col in secondary_bars:
                        ax2.bar(data.index, data[col], label=legend_label, 
                               alpha=0.7, width=bar_width)
                    else:
                        ax2.plot(data.index, data[col], marker='s', label=legend_label, linestyle='--')
            
            # Configuration finale
            ax1.set_xlabel('Date')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            if lines1 or lines2:
                ax1.legend(lines1 + lines2, labels1 + labels2, 
                         bbox_to_anchor=(1.15, 1), loc='upper left')
            
            plt.title(f"Données de {sheet_name}")
            
            # Ajouter le commentaire si présent
            if comment:
                ax_comment = plt.subplot(gs[1])
                ax_comment.axis('off')  # Cacher les axes
                ax_comment.text(0, 0.5, f"Commentaires :\n{comment}", 
                            wrap=True, va='center', fontsize=10,
                            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
            
            plt.tight_layout()
            plt.subplots_adjust(right=0.85)
            
            # Sauvegarder la page
            pdf.savefig(bbox_inches='tight')
            plt.close()
    
    return output_pdf

def main():
    st.title("📊 Argus 2.0")
    
    # Sidebar pour les contrôles
    with st.sidebar:
        uploaded_file = st.file_uploader("Choisir un fichier Excel", type="xlsx")
        
        if uploaded_file is not None:
            # Charger les données
            data_dict, prefix_groups = load_data(uploaded_file)
            
            if data_dict:
                # Options d'export
                st.subheader("Export des graphiques")
                
                # Option 1: Export PDF global
                if st.button("📑 Exporter tous les graphiques en PDF"):
                    with st.spinner("Création du PDF en cours..."):
                        pdf_path = export_all_to_pdf(data_dict, prefix_groups)
                        if os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as pdf_file:
                                st.download_button(
                                    label="📥 Télécharger le PDF complet",
                                    data=pdf_file,
                                    file_name=os.path.basename(pdf_path),
                                    mime="application/pdf",
                                    key="download_all_pdf"
                                )
                
                st.markdown("---")
                
                # Option 2: Export sélectif
                st.write("Export sélectif :")
                selected_sheets = []
                for prefix, sheets in prefix_groups.items():
                    st.write(f"**{prefix}**")
                    for sheet in sheets:
                        if st.checkbox(sheet, key=f"select_{sheet}"):
                            selected_sheets.append(sheet)
                
                if selected_sheets:
                    if st.button("📑 Exporter la sélection en PDF"):
                        with st.spinner("Création du PDF en cours..."):
                            pdf_path = export_selected_to_pdf(data_dict, selected_sheets)
                            if os.path.exists(pdf_path):
                                with open(pdf_path, "rb") as pdf_file:
                                    st.download_button(
                                        label="📥 Télécharger la sélection en PDF",
                                        data=pdf_file,
                                        file_name=os.path.basename(pdf_path),
                                        mime="application/pdf",
                                        key="download_selected_pdf"
                                    )
    
    # Contenu principal
    if uploaded_file is not None and data_dict:
        # Créer un onglet pour chaque préfixe
        tabs = st.tabs(list(prefix_groups.keys()))
        
        for tab, (prefix, sheet_names) in zip(tabs, prefix_groups.items()):
            with tab:
                st.header(f"Données {prefix}")
                
                # Pour chaque feuille dans ce préfixe
                for sheet_name in sheet_names:
                    data = data_dict[sheet_name]
                    st.subheader(sheet_name)
                    
                    # Initialiser les clés de session_state si nécessaire
                    if f"primary_{sheet_name}_last" not in st.session_state:
                        auto_primary, auto_secondary = assign_axis(data)
                        st.session_state[f"primary_{sheet_name}_last"] = auto_primary
                        st.session_state[f"secondary_{sheet_name}_last"] = auto_secondary
                    
                    # Conteneur pour l'assignation manuelle
                    with st.expander("🎯 Assignation manuelle des axes"):
                        data_columns = data.columns.tolist()
                        
                        # Initialiser les variables par défaut
                        primary_bars = []
                        secondary_bars = []
                        primary_stack = False
                        secondary_stack = False
                        
                        # Sélection de l'axe principal
                        st.subheader("Axe principal (gauche)")
                        primary_cols = st.multiselect(
                            "Séries (axe gauche)",
                            options=data_columns,
                            default=st.session_state[f"primary_{sheet_name}_last"],
                            key=f"primary_{sheet_name}",
                            format_func=lambda x: f"{x} (gauche)"
                        )
                        
                        # Sélection des barres pour l'axe principal
                        if primary_cols:
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write("Type d'affichage :")
                                cols = st.columns(len(primary_cols))
                                for i, col in enumerate(primary_cols):
                                    with cols[i]:
                                        if st.checkbox("Barres", 
                                                     key=f"bar_primary_{sheet_name}_{col}",
                                                     value=is_bar_column(col)):
                                            primary_bars.append(col)
                            
                            # Option d'empilement pour l'axe principal
                            with col2:
                                if primary_bars and len(primary_bars) > 1:
                                    primary_stack = st.radio(
                                        "Mode des barres (gauche)",
                                        ["Groupées", "En cascade"],
                                        key=f"stack_primary_{sheet_name}"
                                    ) == "En cascade"
                        
                        # Sélection de l'axe secondaire
                        remaining_cols = [col for col in data_columns if col not in primary_cols]
                        st.subheader("Axe secondaire (droit)")
                        secondary_cols = st.multiselect(
                            "Séries (axe droit)",
                            options=remaining_cols,
                            default=[col for col in st.session_state[f"secondary_{sheet_name}_last"] if col in remaining_cols],
                            key=f"secondary_{sheet_name}",
                            format_func=lambda x: f"{x} (droit)"
                        )
                        
                        # Sélection des barres pour l'axe secondaire
                        if secondary_cols:
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write("Type d'affichage :")
                                cols = st.columns(len(secondary_cols))
                                for i, col in enumerate(secondary_cols):
                                    with cols[i]:
                                        if st.checkbox("Barres", 
                                                     key=f"bar_secondary_{sheet_name}_{col}",
                                                     value=is_bar_column(col)):
                                            secondary_bars.append(col)
                            
                            # Option d'empilement pour l'axe secondaire
                            with col2:
                                if secondary_bars and len(secondary_bars) > 1:
                                    secondary_stack = st.radio(
                                        "Mode des barres (droit)",
                                        ["Groupées", "En cascade"],
                                        key=f"stack_secondary_{sheet_name}"
                                    ) == "En cascade"
                    
                    # Stocker les nouvelles sélections
                    st.session_state[f"primary_{sheet_name}_last"] = primary_cols
                    st.session_state[f"secondary_{sheet_name}_last"] = secondary_cols
                    
                    # Créer et afficher le graphique dans une colonne de 70% de largeur
                    left_col, right_col = st.columns([0.7, 0.3])  # 70% gauche, 30% droite
                    with left_col:
                        fig = create_multi_axis_plot(
                            data,
                            primary_cols, secondary_cols,
                            title=sheet_name,
                            primary_bars=primary_bars,
                            secondary_bars=secondary_bars,
                            primary_stack=primary_stack,
                            secondary_stack=secondary_stack
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Bouton d'export PDF
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if st.button("Exporter en PDF", key=f"export_{sheet_name}"):
                                with st.spinner("Création du PDF en cours..."):
                                    # Récupérer le commentaire pour l'export
                                    comment = st.session_state.get(f"comment_{sheet_name}", "")
                                    pdf_path = export_to_pdf(data, primary_cols, secondary_cols, title=sheet_name, 
                                                        primary_bars=primary_bars, secondary_bars=secondary_bars,
                                                        primary_stack=primary_stack, secondary_stack=secondary_stack,
                                                        comment=comment)
                                    st.session_state[f"pdf_path_{sheet_name}"] = pdf_path
                                    st.success(f"Graphique exporté en PDF")
                        
                        # Bouton de téléchargement si un PDF a été généré
                        with col2:
                            if f"pdf_path_{sheet_name}" in st.session_state:
                                pdf_path = st.session_state[f"pdf_path_{sheet_name}"]
                                if os.path.exists(pdf_path):
                                    with open(pdf_path, "rb") as pdf_file:
                                        st.download_button(
                                            label="📥 Télécharger le PDF",
                                            data=pdf_file,
                                            file_name=os.path.basename(pdf_path),
                                            mime="application/pdf",
                                            key=f"download_{sheet_name}"
                                        )
                        
                        # Afficher les statistiques
                        display_statistics(data, sheet_name)
                    
                    # Ajouter la zone de commentaire dans la colonne de droite
                    with right_col:
                        with st.expander("💭 Commentaires"):
                            comment = st.text_area(
                                "Ajouter un commentaire pour ce graphique",
                                value=st.session_state.get(f"comment_{sheet_name}", ""),
                                key=f"comment_{sheet_name}",
                                height=300
                            )
                    
                    st.divider()
    else:
        st.info("👈 Commencez par charger un fichier Excel dans la barre latérale")

if __name__ == "__main__":
    main()
