"""Test verisi üzerinde model performansını değerlendirir"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple

from ..core.classifier import predict_sentiment
from ..core.lexicon import load_lexicon
from .metrics import (
    calculate_metrics,
    confusion_matrix,
    print_confusion_matrix,
    analyze_errors
)


def polarity_to_label(polarity: int) -> str:
    """
    Polarity değerini label'a çevirir.
    
    Args:
        polarity: 0=negative, 1=neutral, 2=positive
    
    Returns:
        str: "negative", "neutral", veya "positive"
    """
    mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return mapping.get(polarity, 'neutral')


def load_test_data(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Test CSV dosyasını yükler.
    
    Args:
        csv_path: Test CSV dosyasının yolu (Sentence, Polarity kolonları olmalı)
    
    Returns:
        (texts, true_labels) tuple
    """
    df = pd.read_csv(csv_path)
    
    if 'Sentence' not in df.columns or 'Polarity' not in df.columns:
        raise ValueError("CSV'de 'Sentence' ve 'Polarity' kolonları olmalı!")
    
    texts = df['Sentence'].tolist()
    true_labels = [polarity_to_label(p) for p in df['Polarity'].tolist()]
    
    return texts, true_labels


def evaluate_model(test_csv_path: str, verbose: bool = True) -> dict:
    """
    Modeli test verisi üzerinde değerlendirir.
    
    Args:
        test_csv_path: Test CSV dosyasının yolu
        verbose: Detaylı çıktı göster
    
    Returns:
        dict: Değerlendirme sonuçları
    """
    # Lexicon'u yükle
    lex_mgr = load_lexicon()
    
    # Test verisini yükle
    texts, y_true = load_test_data(test_csv_path)
    
    if verbose:
        print(f"\n📊 Test verisi yüklendi: {len(texts)} örnek\n")
    
    # Tahminleri yap
    y_pred = []
    scores = []
    
    for text in texts:
        label, score, _ = predict_sentiment(text, lex_mgr)
        y_pred.append(label)
        scores.append(score)
    
    # Metrikleri hesapla
    overall_metrics = calculate_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    errors = analyze_errors(y_true, y_pred, texts, scores)
    
    # Sonuçları yazdır
    if verbose:
        print("\n" + "="*70)
        print("GENEL METRİKLER")
        print("="*70)
        for metric, value in overall_metrics.items():
            print(f"{metric.upper():15}: {value:.4f} ({value*100:.2f}%)")
        
        print_confusion_matrix(cm)
        
        # Hata analizi
        if errors:
            print("\n" + "="*70)
            print(f"HATA ANALİZİ (Toplam {len(errors)} hata)")
            print("="*70)
            
            for i, error in enumerate(errors):  # Tüm hataları göster
                print(f"\n[{i+1}] Yorum: {error['text']}")
                print(f"    Gerçek: {error['true_label']:8} | Tahmin: {error['predicted_label']:8} | Skor: {error['score']}")
                print("-" * 70)
    
    return {
        'overall_metrics': overall_metrics,
        'confusion_matrix': cm.tolist(),
        'errors': errors,
        'total_samples': len(texts),
        'correct_predictions': len(texts) - len(errors),
        'error_count': len(errors)
    }


def main():
    """Değerlendirmeyi çalıştır"""
    base_dir = Path(__file__).resolve().parents[1]
    test_csv = base_dir / "eval" / "test_reviews.csv"
    
    print("\n" + "#"*70)
    print("#" + " "*20 + "MODEL DEĞERLENDİRME" + " "*20 + "#")
    print("#"*70)
    
    results = evaluate_model(str(test_csv), verbose=True)
    
    print("\n" + "="*70)
    print("ÖZET")
    print("="*70)
    print(f"Toplam test örneği: {results['total_samples']}")
    print(f"Doğru tahmin: {results['correct_predictions']}")
    print(f"Hatalı tahmin: {results['error_count']}")
    print(f"Accuracy: {results['overall_metrics']['accuracy']:.4f} ({results['overall_metrics']['accuracy']*100:.2f}%)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main() 