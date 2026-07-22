# -*- coding: utf-8 -*-
"""
Task 1.3 - Embedding Olusturma
-------------------------------
Bu script, Task 1.2'de urettigimiz chunk'lari okuyup her birini bir
embedding vektorune cevirir.

Kullanilan model: ytu-ce-cosmos/turkish-e5-large
- Yildiz Teknik Universitesi COSMOS AI Arastirma Grubu tarafindan gelistirilmis
- intfloat/multilingual-e5-large-instruct modelinin çesitli Turkce veri
  kumeleriyle fine-tune edilmis versiyonu
- TR-MTEB (Turkce Embedding Benchmark) sonuclarina gore acik kaynak modeller
  arasinda en iyi performansi gosteren modellerden biri (retrieval skoru: 77.0)
- Boyut: ~1.1GB

ONEMLI - Kullanim deseni bu modelde FARKLIDIR (multilingual-e5-small'dan farkli):
Bu model, "instruct" tabanli bir E5 modelinden turetildigi icin:
- DOKUMAN/CHUNK metinleri: hicbir on ek eklenmeden, ham haliyle embed edilir.
- SORGU (query) metinleri: "Instruct: {gorev aciklamasi}\\nQuery: {soru}"
  formatinda bir on ek gerektirir (bu, Task 1.5 - Retriever'da uygulanacak,
  cunku o asamada kullanicinin sorusunu embed edecegiz).
Bu script sadece CHUNK'lari (dokuman parcalarini) embed ettigi icin, burada
hicbir on ek KULLANILMIYOR - bu bilerek yapilmis bir tercih, hata degil.

Kullanim:
    python 03_embedder.py ../data/processed/chunks_madde_bazli.json
"""

import sys
import os
import json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL_ADI = "ytu-ce-cosmos/turkish-e5-large"


def chunklari_yukle(json_yolu: str) -> list[dict]:
    if not os.path.isfile(json_yolu):
        raise FileNotFoundError(f"Dosya bulunamadi: {json_yolu}")
    with open(json_yolu, "r", encoding="utf-8") as f:
        return json.load(f)


def modeli_yukle():
    """
    sentence-transformers kutuphanesini ve modeli yukler. Import'u
    fonksiyon icinde yapiyoruz ki kutuphane kurulu degilse kullaniciya
    daha anlasilir bir hata verebilelim.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers kutuphanesi kurulu degil. "
            "'pip install -r requirements.txt' komutunu calistirdiginizdan emin olun."
        )

    print(f"Model yukleniyor: {MODEL_ADI}")
    print("(Ilk calistirmada model internetten indirilecek, ~1.1GB, biraz surebilir)")
    model = SentenceTransformer(MODEL_ADI)
    return model


def embeddingleri_olustur(model, chunklar: list[dict]):
    """
    Her chunk icin embedding vektoru uretir.

    turkish-e5-large icin dokuman/chunk metinlerine herhangi bir on ek
    EKLENMEZ (bu, multilingual-e5-small'daki "passage: " on ekinden farkli
    bir kullanim desenidir - bkz. dosya basindaki aciklama).

    normalize_embeddings=True: vektorleri birim uzunluga getirir, boylece
    Task 1.4/1.5'te cosine similarity hesaplamasi basit bir ic carpimla
    yapilabilir (ChromaDB bu sekilde daha verimli calisir).
    """
    metinler = [c["metin"] for c in chunklar]
    vektorler = model.encode(
        metinler,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return vektorler


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python 03_embedder.py <chunk_json_yolu>")
        print("Ornek:    python 03_embedder.py ../data/processed/chunks_madde_bazli.json")
        sys.exit(1)

    json_yolu = sys.argv[1]

    try:
        chunklar = chunklari_yukle(json_yolu)
    except FileNotFoundError as hata:
        print(f"\n\u274c HATA: {hata}")
        sys.exit(1)

    print(f"{len(chunklar)} chunk yuklendi.\n")

    try:
        model = modeli_yukle()
    except ImportError as hata:
        print(f"\n\u274c HATA: {hata}")
        sys.exit(1)

    vektorler = embeddingleri_olustur(model, chunklar)

    print(f"\nEmbedding boyutu (vektor uzunlugu): {vektorler.shape[1]}")
    print(f"Toplam embedding sayisi: {vektorler.shape[0]}")

    # Her chunk'a kendi embedding'ini ekle
    for chunk, vektor in zip(chunklar, vektorler):
        chunk["embedding"] = vektor.tolist()

    kok, uzanti = os.path.splitext(json_yolu)
    cikti_yolu = kok + "_embedded.json"

    with open(cikti_yolu, "w", encoding="utf-8") as f:
        # indent kullanmiyoruz: embedding vektorleri (384 sayi/chunk) ile
        # dosya zaten buyuk olacak, indent bosuna yer kaplar
        json.dump(chunklar, f, ensure_ascii=False)

    print(f"\n\u2705 Embedding'li chunk'lar kaydedildi: {cikti_yolu}")

    print("\n--- Ornek: ilk chunk'in embedding'inden ilk 5 sayi ---")
    print(chunklar[0]["madde_no"], "->", chunklar[0]["embedding"][:5])

    print("\nBir sonraki adim: ChromaDB'ye kayit (Task 1.4)")
