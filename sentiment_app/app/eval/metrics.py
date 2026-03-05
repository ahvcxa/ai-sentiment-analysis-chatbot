#accuracy, precision, recall, F1 score gibi metrikleri hesaplar
from typing import List, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix as sklearn_cm
)

def calculate_metrics(y_true, y_pred):
    """
    Temel metrikleri hesaplar.
    
    Args:
        y_true: Gerçek etiketler ["positive", "negative", "neutral"]
        y_pred: Tahmin edilen etiketler
    
    Returns:
        Dict: accuracy, precision, recall, f1_score
    """
    # Accuracy hesapla
    acc = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 hesapla (macro average - her sınıfa eşit ağırlık)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, 
        y_pred, 
        average='macro',
        zero_division=0
    )
    
    return {
        'accuracy': round(acc, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4)
    }
def confusion_matrix(y_true, y_pred):
    """
    Confusion matrix oluşturur.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
    
    Returns:
        numpy.ndarray: 3x3 confusion matrix
        Satırlar: Gerçek etiketler [negative, neutral, positive]
        Sütunlar: Tahmin edilen etiketler [negative, neutral, positive]
    """
    labels = ['negative', 'neutral', 'positive']
    cm = sklearn_cm(y_true, y_pred, labels=labels)
    return cm
def print_confusion_matrix(matrix):
    """
    Confusion matrix'i güzel bir formatta yazdırır.
    
    Args:
        matrix: 3x3 confusion matrix (numpy array)
    """
    labels = ['negative', 'neutral', 'positive']
    
    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    print("Satırlar: Gerçek etiketler | Sütunlar: Tahmin edilen etiketler\n")
    
    # Header
    header = "               |"
    for label in labels:
        header += f" {label:^15} |"
    print(header)
    print("-" * len(header))
    
    # Rows
    for i, true_label in enumerate(labels):
        row = f" {true_label:^13} |"
        for j in range(len(labels)):
            row += f" {matrix[i][j]:^15} |"
        print(row)
    
    print("="*70 + "\n")
def analyze_errors(y_true, y_pred, texts=None, scores=None):
    """
    Hatalı tahminleri analiz eder.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        texts: Yorumların metinleri (opsiyonel)
        scores: Tahmin skorları (opsiyonel)
    
    Returns:
        List[Dict]: Hatalı tahminlerin listesi
    """
    errors = []
    
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            error_info = {
                'index': i,
                'true_label': y_true[i],
                'predicted_label': y_pred[i]
            }
            
            if texts is not None and i < len(texts):
                error_info['text'] = texts[i]
            
            if scores is not None and i < len(scores):
                error_info['score'] = round(scores[i], 2)
            
            errors.append(error_info)
    
    return errors