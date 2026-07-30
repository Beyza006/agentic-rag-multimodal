# -*- coding: utf-8 -*-
"""
Task 1.5 - Retriever
----------------------
Bu script, kullanicinin dogal dilde bir soru sormasini, bu soruyu embedding'e
cevirmesini ve ChromaDB'de en alakali chunk'lari bulup getirmesini saglar.

ONEMLI - Sorgu (query) formati:
turkish-e5-large, intfloat/multilingual-e5-large-instruct modelinden
turetildigi icin SORGULAR icin ozel bir format gerektirir:

    "Instruct: {gorev_tanimi}\nQuery: {kullanicinin_sorusu}"

Bu, Task 1.3'te DOKUMANLARI (chunk'lari) embed ederken hicbir on ek
kullanmadigimizdan FARKLIDIR - o asimetriktir: dokuman duz metin, sorgu
ozel formatli. Modelin kendi model kartinda onerilen gorev tanimini
kullaniyoruz (ytu-ce-cosmos/turkish-e5-large, Hugging Face).

Kullanim:
    python 05_retriever.py "Kişisel veriler ne zaman açık rıza olmadan işlenebilir?"
"""

import sys
import os

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL_ADI = "ytu-ce-cosmos/turkish-e5-large"
CHROMA_DB_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "chroma_db")
KOLEKSIYON_ADI = "kvkk_madde_bazli"

# Modelin kendi model kartinda onerdigi gorev tanimi (Turkce siparise
# ozel hazirlanmis, degistirmeden kullanmak en iyi sonucu verir)
GOREV_TANIMI = (
    "Given a Turkish search query, retrieve relevant passages "
    "written in Turkish that best answer the query"
)


def sorgu_formatla(soru: str) -> str:
    """Sorguyu modelin bekledigi 'Instruct: ...\\nQuery: ...' formatina cevirir."""
    return f"Instruct: {GOREV_TANIMI}\nQuery: {soru}"


def modeli_yukle():
    from sentence_transformers import SentenceTransformer
    print(f"Model yukleniyor: {MODEL_ADI} (daha once indirilmisse cache'den gelir)...")
    return SentenceTransformer(MODEL_ADI)


def koleksiyonu_ac():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_YOLU)
    try:
        return client.get_collection(KOLEKSIYON_ADI)
    except Exception as hata:
        raise RuntimeError(
            f"'{KOLEKSIYON_ADI}' koleksiyonu bulunamadi. Once "
            f"04_chromadb_writer.py'yi calistirdiginizdan emin olun. Detay: {hata}"
        )


def ara(model, koleksiyon, soru: str, top_k: int = 5) -> dict:
    """
    Kullanicinin sorusunu embed edip ChromaDB'de arama yapar.

    Donen deger: ChromaDB'nin query() sonucunun kendisi (documents,
    metadatas, distances iceren sozluk).
    """
    formatli_sorgu = sorgu_formatla(soru)
    sorgu_embedding = model.encode(
        [formatli_sorgu],
        normalize_embeddings=True
    )[0].tolist()

    sonuc = koleksiyon.query(
        query_embeddings=[sorgu_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return sonuc


def sonuclari_yazdir(soru: str, sonuc: dict):
    print(f"\nSoru: \"{soru}\"")
    print("=" * 60)

    dokumanlar = sonuc["documents"][0]
    metadatalar = sonuc["metadatas"][0]
    mesafeler = sonuc["distances"][0]

    if not dokumanlar:
        print("Hicbir sonuc bulunamadi.")
        return

    for i, (doc, meta, mesafe) in enumerate(zip(dokumanlar, metadatalar, mesafeler)):
        # ChromaDB cosine UZAKLIGI (distance) donduruyor, benzerlik degil.
        # cosine_distance = 1 - cosine_similarity oldugu icin benzerligi
        # geri hesaplamak icin: benzerlik = 1 - mesafe
        benzerlik = 1 - mesafe
        print(f"\n[{i+1}] {meta['madde_no']} (sayfa {meta['sayfa_no']}) - benzerlik: {benzerlik:.3f}")
        print(f"    {doc[:250]}{'...' if len(doc) > 250 else ''}")

    print("\n" + "=" * 60)
    print(f"En alakali sonuc: {metadatalar[0]['madde_no']} (sayfa {metadatalar[0]['sayfa_no']})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python 05_retriever.py \"<soru>\"")
        print("Ornek:    python 05_retriever.py \"Kişisel veriler ne zaman açık rıza olmadan işlenebilir?\"")
        sys.exit(1)

    soru = sys.argv[1]

    try:
        koleksiyon = koleksiyonu_ac()
    except RuntimeError as hata:
        print(f"\n\u274c HATA: {hata}")
        sys.exit(1)

    model = modeli_yukle()
    sonuc = ara(model, koleksiyon, soru, top_k=5)
    sonuclari_yazdir(soru, sonuc)

    print("\nBir sonraki adim: LLM ile cevap uretme (Task 1.6)")
