"""
src/visualize.py
----------------
Visualization utilities for clinical NLP pipeline.
Author: Aruna Kunche
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords


def plot_specialty_distribution(specialty_series, top_n: int = 15,
                                 save_path: str = None):
    """Bar chart of top N medical specialties by note count."""
    counts = specialty_series.value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=sns.color_palette('Blues_d', top_n))
    ax.set_xlabel('Number of Notes', fontsize=12)
    ax.set_title(f'Top {top_n} Medical Specialties', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()


def plot_note_length_distribution(length_series, save_path: str = None):
    """Histogram of clinical note word counts."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(length_series, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(length_series.median(), color='red', linestyle='--',
               label=f'Median: {length_series.median():.0f}')
    ax.axvline(length_series.mean(), color='orange', linestyle='--',
               label=f'Mean: {length_series.mean():.0f}')
    ax.set_xlabel('Words per Note', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Clinical Note Lengths', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()


def plot_wordclouds(df, specialty_col: str, text_col: str,
                    specialties: list, save_path: str = None):
    """Word cloud grid for selected specialties."""
    n = len(specialties)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).flatten()
    sw = set(stopwords.words('english'))
    for i, spec in enumerate(specialties):
        text = ' '.join(df[df[specialty_col] == spec][text_col].tolist())
        wc = WordCloud(width=400, height=260, background_color='white',
                       colormap='Blues', max_words=80, stopwords=sw).generate(text)
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].axis('off')
        axes[i].set_title(spec[:30], fontsize=10, fontweight='bold')
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Most Frequent Words by Specialty', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()


def plot_model_comparison(results: dict, save_path: str = None):
    """Side-by-side bar chart comparing model accuracy and F1."""
    names = list(results.keys())
    accs  = [results[n]['accuracy']    for n in names]
    f1s   = [results[n]['weighted_f1'] for n in names]

    x = np.arange(len(names))
    w = 0.32
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, accs, w, label='Accuracy',    color='steelblue',     edgecolor='white')
    b2 = ax.bar(x + w/2, f1s,  w, label='Weighted F1', color='lightsteelblue', edgecolor='white')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_title('Model Comparison: Accuracy vs Weighted F1', fontsize=13, fontweight='bold')
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    for bar in list(b1) + list(b2):
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels: list,
                           save_path: str = None):
    """Dual confusion matrix: raw counts + normalized."""
    cm      = np.array([[v for v in row]
                         for row in __import__('sklearn.metrics', fromlist=['confusion_matrix'])
                         .confusion_matrix(y_true, y_pred, labels=labels)])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    short   = [l[:18] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short, yticklabels=short,
                ax=axes[0], linewidths=0.3)
    axes[0].set_title('Raw Counts', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
    axes[0].tick_params(axis='x', rotation=40, labelsize=8)
    axes[0].tick_params(axis='y', rotation=0,  labelsize=8)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=short, yticklabels=short,
                ax=axes[1], linewidths=0.3, vmin=0, vmax=1)
    axes[1].set_title('Normalized (Recall per Class)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
    axes[1].tick_params(axis='x', rotation=40, labelsize=8)
    axes[1].tick_params(axis='y', rotation=0,  labelsize=8)

    plt.suptitle('Confusion Matrix — TF-IDF + Linear SVC',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()


def plot_top_terms(pipeline, n_terms: int = 12, save_path: str = None):
    """Top TF-IDF coefficient terms per specialty (Logistic Regression only)."""
    tfidf    = pipeline.named_steps['tfidf']
    clf      = pipeline.named_steps['clf']
    vocab    = np.array(tfidf.get_feature_names_out())
    classes  = clf.classes_
    n_cols   = 5
    n_rows   = int(np.ceil(len(classes) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 4))
    axes = axes.flatten()
    for i, (spec, ax) in enumerate(zip(classes, axes)):
        idx     = clf.coef_[i].argsort()[-n_terms:][::-1]
        terms   = vocab[idx]
        weights = clf.coef_[i][idx]
        ax.barh(terms[::-1], weights[::-1],
                color=sns.color_palette('Blues_d', n_terms))
        ax.set_title(spec[:28], fontsize=9, fontweight='bold')
        ax.tick_params(axis='y', labelsize=7.5)
        ax.tick_params(axis='x', labelsize=7)
    for j in range(len(classes), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Top Predictive Terms per Specialty (LR Coefficients)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()
