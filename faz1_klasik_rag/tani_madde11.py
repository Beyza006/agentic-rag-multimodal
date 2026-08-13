# -*- coding: utf-8 -*-
"""
TANI SCRIPTI: "İlgili kişinin KVKK kapsamındaki hakları nelerdir?" sorusu
icin ChromaDB'nin GERCEKTEN neyi getirdigini (hangi maddeler, hangi
benzerlik skorlariyla) gosterir - boylece MADDE 11'in neden retrieval'da
zayif cikmis olabilecegini anlayabiliriz.

Kullanim:
    python tani_madde11.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util
spec = importlib.util.spec_from_file_location("retriever", "05_retriever.py")
retriever = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retriever)

SORU = "İlgili kişinin KVKK kapsamındaki hakları nelerdir?"

print(f"Soru: \"{SORU}\"\n")

koleksiyon = retriever.koleksiyonu_ac()
model = retriever.modeli_yukle()
sonuc = retriever.ara(model, koleksiyon, SORU, top_k=8)  # top_k=8, normalde 5 kullaniliyor - daha genis bir pencereden bakalim

print("=== RETRIEVAL SONUCU (benzerlik skoruna gore siralı) ===\n")
for i, (doc, meta, mesafe) in enumerate(zip(
    sonuc["documents"][0], sonuc["metadatas"][0], sonuc["distances"][0]
)):
    benzerlik = 1 - mesafe
    print(f"[{i+1}] {meta['madde_no']} (sayfa {meta['sayfa_no']}) - benzerlik: {benzerlik:.3f}")
    print(f"    {doc[:150]}...")
    print()

# MADDE 11'in TAM metnini de dogrudan ChromaDB'den cekip, kendi
# EMBEDDING'inin soruya olan benzerligini ayrica hesaplayalim - boylece
# "MADDE 11 hic mi retrieval'a girmiyor, yoksa top-5/top-8 disinda mi
# kaliyor" sorusunu net cevaplayabiliriz.
print("\n=== MADDE 11'in TUM PARCALARI (dogrudan ChromaDB'den) ===\n")
madde11_kayitlar = koleksiyon.get(where={"madde_no": "MADDE 11"}, include=["documents", "metadatas"])
print(f"MADDE 11 icin ChromaDB'de {len(madde11_kayitlar['documents'])} parca bulundu.\n")
for doc, meta in zip(madde11_kayitlar["documents"], madde11_kayitlar["metadatas"]):
    print(f"chunk_index={meta['chunk_index']} (sayfa {meta['sayfa_no']}):")
    print(f"  {doc[:300]}")
    print()

# MADDE 11'in ilk parcasinin soruya olan HAM benzerligini manuel hesapla
if madde11_kayitlar["documents"]:
    import numpy as np
    ilk_parca = madde11_kayitlar["documents"][0]
    sorgu_emb = model.encode([retriever.sorgu_formatla(SORU)], normalize_embeddings=True)[0]
    dokuman_emb = model.encode([ilk_parca], normalize_embeddings=True)[0]
    manuel_benzerlik = float(np.dot(sorgu_emb, dokuman_emb))
    print(f"\n=== MADDE 11'in (ilk parca) soruya HAM cosine benzerligi: {manuel_benzerlik:.3f} ===")
