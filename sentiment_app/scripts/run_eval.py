"""Test ve değerlendirme scriptini çalıştırır"""

import sys
from pathlib import Path

# Parent dizinleri path'e ekle
base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

from app.eval.evaluate import main

if __name__ == "__main__":
    print("\n🚀 Model değerlendirmesi başlatılıyor...\n")
    main()
    print("\n✅ Değerlendirme tamamlandı!\n")