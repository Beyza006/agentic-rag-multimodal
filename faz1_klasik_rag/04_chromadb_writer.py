# -*- coding: utf-8 -*-
"""
Task 1.4 - ChromaDB'ye Kayit
------------------------------
Bu script, Task 1.3'te urettigimiz embedding'li chunk'lari (madde_no,
chunk_index, sayfa_no, metin, embedding icere JSON) kalici bir ChromaDB
veritabanina yazar.

Neden ChromaDB'ye yaziyoruz (JSON'da tutmak yerine)?
- ChromaDB, embedding'ler uzerinde HIZLI benzerlik aramasi (similarity
  search) yapmak icin ozel olarak tasarlanmis bir vector database'dir.
  JSON dosyasinda arama yapmak icin her seferinde TUM vektorleri elle
  karsilastirmamiz gerekirdi (yavas); ChromaDB bunu optimize edilmis
  indeksleme (HNSW algoritmasi) ile cok daha hizli yapar.
- Kalicidir (persistent): Program kapansa bile veriler diskte kalir,
  her calistirmada yeniden embedding uretmemize gerek kalmaz.

Kullanim:
    python 04_chromadb_writer.py ../data/processed/chunks_madde_bazli_embedded.json
"""

import sys
import os
import json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ChromaDB veritabaninin diskte saklanacagi klasor
CHROMA_DB_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "chroma_db")
KOLEKSIYON_ADI = "kvkk_madde_bazli"


def embedded_chunklari_yukle(json_yolu: str) -> list[dict]:
    if not os.path.isfile(json_yolu):
        raise FileNotFoundError(f"Dosya bulunamadi: {json_yolu}")
    with open(json_yolu, "r", encoding="utf-8") as f:
        chunklar = json.load(f)

    # Temel dogrulama: her chunk'ta embedding var mi?
    for c in chunklar:
        if "embedding" not in c or not c["embedding"]:
            raise ValueError(
                f"'{c.get('madde_no', '?')}' chunk'inda embedding eksik. "
                f"Once 03_embedder.py'yi calistirdiginizdan emin olun."
            )
    return chunklar


def chunk_id_olustur(madde_no: str, chunk_index: int) -> str:
    """
    ChromaDB her kayit icin benzersiz bir ID ister. 'MADDE 5' ve
    chunk_index=0 -> 'MADDE_5_0' gibi ID'ler uretiyoruz. Bosluklari
    alt cizgiye ceviriyoruz cunku bazi ID formatlarinda bosluk sorun
    cikarabilir.
    """
    temiz_ad = madde_no.replace(" ", "_")
    return f"{temiz_ad}_{chunk_index}"


def chromadb_ye_yaz(chunklar: list[dict]):
    import chromadb

    # PersistentClient: veriler diske yazilir, program kapansa da kalir
    client = chromadb.PersistentClient(path=CHROMA_DB_YOLU)

    # Koleksiyon zaten varsa uzerine yazmamak icin once siliyoruz
    # (script'i birden fazla kez calistirinca "zaten var" hatasi almamak icin).
    #
    # Koleksiyon sıfırlama işlemi sırasında olası bağlantı kilitlenmelerini (lock) önlemek için güvenli sıfırlama metodu (reset) kullanılmıştır.
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python 04_chromadb_writer.py <embedded_json_yolu>")
        print("Ornek:    python 04_chromadb_writer.py ../data/processed/chunks_madde_bazli_embedded.json")
        sys.exit(1)

    json_yolu = sys.argv[1]

    try:
        chunklar = embedded_chunklari_yukle(json_yolu)
    except (FileNotFoundError, ValueError) as hata:
        print(f"\n\u274c HATA: {hata}")
        sys.exit(1)

    print(f"{len(chunklar)} embedding'li chunk yuklendi.")
    print(f"ChromaDB'ye yaziliyor: {os.path.normpath(CHROMA_DB_YOLU)}")

    koleksiyon = chromadb_ye_yaz(chunklar)

    print(f"\n\u2705 Basariyla yazildi.")
    print(f"Koleksiyon adi: {KOLEKSIYON_ADI}")
    print(f"Koleksiyondaki toplam kayit sayisi: {koleksiyon.count()}")

    # Basit bir dogrulama: veritabanini YENIDEN acip (persistent oldugunu
    # kanitlamak icin) birkac kaydi geri okuyalim
    print("\n--- Dogrulama: veritabanini yeniden acip ornek kayit okuyoruz ---")
    import chromadb
    dogrulama_client = chromadb.PersistentClient(path=CHROMA_DB_YOLU)
    dogrulama_koleksiyon = dogrulama_client.get_collection(KOLEKSIYON_ADI)
    ornek = dogrulama_koleksiyon.get(limit=2, include=["documents", "metadatas"])
    for i, (doc, meta) in enumerate(zip(ornek["documents"], ornek["metadatas"])):
        print(f"  [{i}] {meta['madde_no']} (sayfa {meta['sayfa_no']}): {doc[:80]}...")

    print("\nBir sonraki adim: Retriever (Task 1.5)")
