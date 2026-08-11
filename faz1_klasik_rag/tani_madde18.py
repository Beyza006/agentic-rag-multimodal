# -*- coding: utf-8 -*-
"""
TANI SCRIPTI: MADDE 18'in ChromaDB'deki TUM parcalarini (chunk) tek tek
gosterir - boylece "d bendinin sonu hangi parcada, eksik mi" sorusunu
kesin olarak cevaplayabiliriz.

Kullanim:
    python tani_madde18.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util
spec = importlib.util.spec_from_file_location("retriever", "05_retriever.py")
retriever = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retriever)

koleksiyon = retriever.koleksiyonu_ac()

sonuc = koleksiyon.get(
    where={"madde_no": "MADDE 18"},
    include=["documents", "metadatas"]
)

print(f"MADDE 18 icin ChromaDB'de toplam {len(sonuc['documents'])} parca (chunk) bulundu.\n")

# chunk_index'e gore sirala (dogru okuma sirasi icin)
parcalar = sorted(zip(sonuc["documents"], sonuc["metadatas"]), key=lambda x: x[1]["chunk_index"])

for i, (doc, meta) in enumerate(parcalar):
    print(f"{'='*70}")
    print(f"PARCA {i+1}/{len(parcalar)} | chunk_index={meta['chunk_index']} | sayfa_no={meta['sayfa_no']}")
    print(f"{'='*70}")
    print(doc)
    print()

print("\n\n--- ÖZET KONTROL ---")
tum_metin = " ".join(doc for doc, meta in parcalar)
print("'idari para cezası verilir' ifadesi HERHANGİ bir parçada var mı:",
      "idari para cezası verilir" in tum_metin)
print("'50.000 Türk lirasından' ifadesi HERHANGİ bir parçada var mı:",
      "50.000 Türk lirasından" in tum_metin)
