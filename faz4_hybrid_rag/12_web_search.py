# -*- coding: utf-8 -*-
"""
Task 4.1 — Web Search Tool
----------------------------
Bu script, agent'in ihtiyaç duyduğunda internetten bilgi çekebilmesi
için bir WEB ARAÇ (tool) katmanı sağlar.

NEDEN GEREKLİ: Faz 1-2'deki agent sadece ChromaDB içindeki
dökümanlara bakabiliyor. Doküman yetersiz kaldığında "bilgi bulunamadı"
diyor. Bu tool, o boşluğu kapatır: doküman bilgisi yetersizse web'e
başvurulur.

ARAÇ SEÇİMİ (otomatik — sırasıyla dener):
  1) Tavily  → .env dosyasında TAVILY_API_KEY dolu ise kullanılır.
              RAG projeleri için özel olarak tasarlanmış, temiz özetler
              döndürür, LangChain/LangGraph ile resmi entegrasyon var.
  2) DuckDuckGo → API key gerekmez, tamamen ücretsiz. Tavily key
              yoksa otomatik olarak bu kullanılır.

Kullanım (doğrudan test):
    python 12_web_search.py "KVKK 2024 değişiklikleri"
    python 12_web_search.py "Kişisel veri ihlali cezaları Türkiye"
"""

import sys
import os

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# .env dosyasındaki değişkenleri yükle (python-dotenv)
# Proje kök klasöründeki .env'i okumak için iki üst klasöre çıkıyoruz:
#   bu dosya → faz4_hybrid_rag/ → agentic-rag-project/ (kök)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_proje_koku = os.path.join(_current_dir, "..")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_proje_koku, ".env"))
except ImportError:
    # python-dotenv yüklü değilse devam et (os.environ'dan okumayı dener)
    pass

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

# Hangi aracı kullanacağımızı bir kez belirleyip sabit tutuyoruz
# (her aramada tekrar kontrol etmemek için).
_KULLANILAN_ARAC = "tavily" if TAVILY_API_KEY else "duckduckgo"


# ============================================================
# TAVILY ile arama
# ============================================================

def _tavily_ile_ara(sorgu: str, maks_sonuc: int = 5) -> list:
    """
    Tavily Search API'yi kullanarak web araması yapar.

    Tavily neden tercih edilir:
    - RAG sistemleri için özel olarak geliştirilmiş
    - Ham HTML yerine temiz, özetlenmiş snippet'ler döndürür
    - LangChain/LangGraph ile resmi entegrasyon
    - Her arama 1 kredi (aylık 1.000 ücretsiz kredi)

    Dönen format: [{"title": ..., "url": ..., "snippet": ...}, ...]
    """
    try:
        from tavily import TavilyClient
    except ImportError:
        raise RuntimeError(
            "tavily-python paketi yüklü değil. "
            "Kurmak için: pip install tavily-python"
        )

    istemci = TavilyClient(api_key=TAVILY_API_KEY)

    try:
        yanit = istemci.search(
            query=sorgu,
            max_results=maks_sonuc,
            search_depth="advanced",  # "basic" yuzeysel/kelime bazli eslesme
                                        # yapiyordu (orn. "2026 yilinda
                                        # yapay zeka" sorgusunda "2026"ya
                                        # asiri agirlik verip alakasiz
                                        # takvim sayfalari donduruyordu).
                                        # "advanced" anlamsal siralama
                                        # yapiyor, 2 kredi harciyor ama
                                        # ucretsiz kotamiz (1.000/ay) bu
                                        # proje olcegi icin zaten bol.
            include_answer=False,   # Biz kendi LLM'imizle cevap üretiyoruz
        )
    except Exception as hata:
        raise RuntimeError(f"Tavily araması başarısız oldu: {hata}")

    sonuclar = []
    for ogre in yanit.get("results", []):
        sonuclar.append({
            "baslik":  ogre.get("title", ""),
            "url":     ogre.get("url", ""),
            "snippet": ogre.get("content", ""),
        })

    return sonuclar


# ============================================================
# DUCKDUCKGO ile arama (API key gerektirmez)
# ============================================================

def _duckduckgo_ile_ara(sorgu: str, maks_sonuc: int = 5) -> list:
    """
    DuckDuckGo Search API'yi kullanarak web araması yapar.
    Tamamen ücretsizdir, API key gerektirmez.

    Tavily'den farkı: ham arama sonuçları döner (özetlenmemiş),
    Türkçe sorgularda bazen İngilizce sonuçlar gelebilir.

    Dönen format: [{"title": ..., "url": ..., "snippet": ...}, ...]
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # eski paket adı (fallback)
        except ImportError:
            raise RuntimeError(
                "ddgs paketi yüklü değil. "
                "Kurmak için: pip install ddgs"
            )

    sonuclar = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(sorgu, max_results=maks_sonuc):
                sonuclar.append({
                    "baslik":  r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as hata:
        raise RuntimeError(f"DuckDuckGo araması başarısız oldu: {hata}")

    return sonuclar


# ============================================================
# ANA FONKSİYON — Agent bu fonksiyonu çağırır
# ============================================================

def web_de_ara(sorgu: str, maks_sonuc: int = 5) -> list:
    """
    Web araması yapar. Hangi aracın kullanıldığı .env'deki
    TAVILY_API_KEY varlığına göre otomatik belirlenir:
      - Key varsa  → Tavily (daha kaliteli)
      - Key yoksa  → DuckDuckGo (ücretsiz)

    Dönen format (her iki araç için aynı):
    [
        {
            "baslik":  "Sayfa başlığı",
            "url":     "https://...",
            "snippet": "Sayfanın ilgili metin özeti"
        },
        ...
    ]

    Bu ortak format sayesinde 13_hybrid_agent.py, hangi aracın
    kullanıldığına bakmaksızın aynı şekilde sonuçları işleyebilir.
    """
    if _KULLANILAN_ARAC == "tavily":
        return _tavily_ile_ara(sorgu, maks_sonuc)
    else:
        return _duckduckgo_ile_ara(sorgu, maks_sonuc)


def sonuclari_baglama_donustur(sonuclar: list) -> str:
    """
    web_de_ara'dan dönen sonuç listesini, LLM'e bağlam olarak
    gönderilebilecek düz bir metne dönüştürür.

    Her sonuç için:
        [1] Sayfa Başlığı (URL)
        İçerik özeti...

    Bu format, RAG pipeline'ındaki "context" ile aynı mantıkta —
    LLM bu metni okuyarak soruyu cevaplar.
    """
    if not sonuclar:
        return "(Web araması sonuç döndürmedi.)"

    parcalar = []
    for i, sonuc in enumerate(sonuclar, start=1):
        parcalar.append(
            f"[{i}] {sonuc['baslik']} ({sonuc['url']})\n"
            f"{sonuc['snippet']}"
        )

    return "\n\n".join(parcalar)


# ============================================================
# Doğrudan çalıştırma — test amaçlı
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python 12_web_search.py \"<arama sorgusu>\"")
        print("Örnek:    python 12_web_search.py \"KVKK 2024 değişiklikleri\"")
        sys.exit(1)

    sorgu = sys.argv[1]

    print(f"Kullanılan arama aracı: {_KULLANILAN_ARAC.upper()}")
    print(f"Sorgu: {sorgu}")
    print("Arama yapılıyor...\n")

    try:
        sonuclar = web_de_ara(sorgu)
    except RuntimeError as hata:
        print(f"\n❌ HATA: {hata}")
        sys.exit(1)

    if not sonuclar:
        print("Sonuç bulunamadı.")
        sys.exit(0)

    print("=" * 60)
    print(f"BULUNAN SONUÇLAR ({len(sonuclar)} adet):")
    print("=" * 60)
    for i, s in enumerate(sonuclar, start=1):
        print(f"\n[{i}] {s['baslik']}")
        print(f"    URL: {s['url']}")
        print(f"    Özet: {s['snippet'][:200]}{'...' if len(s['snippet']) > 200 else ''}")

    print(f"\n🌐 Task 4.1 — Web arama tamamlandı! ({_KULLANILAN_ARAC.upper()} kullanıldı)")
