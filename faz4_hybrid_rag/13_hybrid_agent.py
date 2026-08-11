# -*- coding: utf-8 -*-
"""
Task 4.2 — Hybrid RAG Agent (Web Search Entegrasyonu)
-------------------------------------------------------
Bu script, Faz 2'deki agent'ı (07_agent.py) bir adım ileri taşır:
doküman (ChromaDB) yetersiz kaldığında otomatik olarak web aramasına
geçer — "Hybrid RAG" böyle çalışır.

FAZ 2 vs FAZ 4 FARKI:
  Faz 2: soru → [KVKK ile ilgili mi?] → rag VEYA doğrudan_cevap
  Faz 4: soru → rag → [bilgi yeterli mi?]
                          ├─ yeterli  → cevap ver
                          └─ yetersiz → web ara → cevap ver

YENİ LANGGRAPH AKIŞI (GÜNCEL - gerçek test sırasında düzeltildi):
  BAŞLANGIÇ
      │
      ▼
  [rag_dugumu]           ChromaDB'de arama yapar (HER soru için, konu
      │                  kısıtlaması YOK - "karar_dugumu"nun sert
      │                  KVKK/alakasız filtresi artık devre dışı,
      │                  bkz. hybrid_agent_olustur() içindeki not)
      ▼
[yeterlilik_dugumu]   ChromaDB yeterli bilgi içeriyor mu?
  ├─ yeterli  → [cevap_uret_dugumu] → SON
  └─ yetersiz → [web_dugumu]
                      │
                      ▼
               [hibrit_cevap_dugumu]  Web sonuçlarıyla cevap ver
                      │
                      ▼
                    SON

NOT: "karar_dugumu" ve "dogrudan_cevap_dugumu" fonksiyonları kodda hâlâ
duruyor (silinmedi, ileride farklı bir senaryoda geri getirilebilir)
ama grafiğe bağlı DEĞİLLER - şu an hiç çalışmıyorlar.

Kullanım:
    python 13_hybrid_agent.py "Kişisel veri ihlalinin cezası nedir?"
    python 13_hybrid_agent.py "2024 sonrası KVKK değişiklikleri nelerdir?"
"""

import sys
import os
import re
import importlib.util
from typing import TypedDict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from langgraph.graph import StateGraph, END

_current_dir = os.path.dirname(os.path.abspath(__file__))
_faz1_dir = os.path.join(_current_dir, "..", "faz1_klasik_rag")
_faz2_dir = os.path.join(_current_dir, "..", "faz2_agentic_rag")


# ============================================================
# Modülleri dosya yoluyla yükleme (rakamla başlayan dosyalar için)
# ============================================================

def _modul_yukle(dosya_yolu: str, modul_adi: str):
    spec = importlib.util.spec_from_file_location(modul_adi, dosya_yolu)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


_retriever  = _modul_yukle(os.path.join(_faz1_dir, "05_retriever.py"), "retriever_module")
_llm_answer = _modul_yukle(os.path.join(_faz1_dir, "06_llm_answer.py"), "llm_answer_module")
_faz2_agent = _modul_yukle(os.path.join(_faz2_dir, "07_agent.py"), "agent_module")

# Faz 4 yeni tool'u: web araması
_web_search = _modul_yukle(os.path.join(_current_dir, "12_web_search.py"), "web_search_module")


# ============================================================
# STATE — Faz 2'ye ek alanlar eklendi
# ============================================================

class HybridAgentState(TypedDict):
    # Faz 2'den devralınanlar
    soru: str
    arac_karari: str          # "rag" veya "dogrudan"
    cevap: str
    kaynaklar: list
    gecmis: list              # Session Memory
    arama_sorgusu: str
    ham_sonuc: dict

    # Faz 4 yeni alanlar
    rag_yeterli: bool         # ChromaDB yeterli bilgi döndürdü mü?
    web_sonuclari: list       # Web araması sonuçları (dict listesi)
    kaynak_turu: str          # "rag" | "web" | "hibrit" — sunumda gösterilecek


# ============================================================
# NODE 1: Karar düğümü (Faz 2'den aynen alındı)
# ============================================================

def karar_dugumu(state: HybridAgentState) -> dict:
    """
    Sorunun KVKK ile ilgili olup olmadığını LLM ile sınıflandırır.
    Faz 2'deki mantığın aynısı — Session Memory de dahil.
    """
    soru   = state["soru"]
    gecmis = state.get("gecmis", [])

    baglam_metni = ""
    if gecmis:
        son_soru, son_cevap = gecmis[-1]
        baglam_metni = (
            f"\nÖNCEKİ KONUŞMA BAĞLAMI:\n"
            f"Önceki soru: {son_soru}\n"
            f"Önceki cevabın özeti: {son_cevap[:200]}\n"
        )

    prompt = (
        f"Aşağıdaki soru, Kişisel Verilerin Korunması Kanunu (KVKK) veya "
        f"kişisel veri koruma konusuyla mı ilgili, yoksa tamamen ALAKASIZ bir konu mu?\n"
        f"{baglam_metni}"
        f"Yeni soru: {soru}\n\n"
        f"Eğer yeni soru önceki konuşmanın DEVAMI niteliğindeyse, İLGİLİ say.\n"
        f"SADECE tek bir kelime yaz: \"ILGILI\" veya \"ALAKASIZ\". Başka hiçbir şey yazma."
    )

    yanit = _llm_answer.ollama_ile_cevap_uret(prompt).strip().upper()
    karar = "dogrudan" if "ALAKASIZ" in yanit else "rag"

    durum_metni = "KVKK ile ilgili" if karar == "rag" else "alakasız"
    print(f"[Karar] Soru '{durum_metni}' → '{karar}' düğümüne yönlendiriliyor.")

    return {"arac_karari": karar}


def yonlendirme_fonksiyonu(state: HybridAgentState) -> str:
    return state["arac_karari"]


# ============================================================
# NODE 2: RAG düğümü — ChromaDB'den bilgi çeker
# ============================================================

def rag_dugumu(state: HybridAgentState) -> dict:
    """
    Faz 1'in retriever'ını çalıştırır. Faz 2'deki rag_dugumu'nden
    TEK FARKI: reranking + cevap üretimi YAPMAZ, sadece ham sonuçları
    döndürür — yeterlilik kararını bir sonraki düğüm verir.

    Bu ayrımın nedeni: eğer ChromaDB yetersizse, cevap üretmek
    için harcanan LLM çağrısından tasarruf etmiş oluruz.
    """
    soru   = state["soru"]
    gecmis = state.get("gecmis", [])

    # Faz 2'deki query rewriting: takip sorularını bağımsız hale getir
    arama_sorgusu = _faz2_agent.sorguyu_baglamla_zenginlestir(soru, gecmis)
    if arama_sorgusu != soru:
        print(f"[Memory] Arama sorgusu zenginleştirildi: '{arama_sorgusu}'")

    koleksiyon     = _llm_answer._retriever.koleksiyonu_ac()
    embedding_modeli = _llm_answer._retriever.modeli_yukle()
    sonuc = _llm_answer._retriever.ara(
        embedding_modeli, koleksiyon, arama_sorgusu, top_k=_llm_answer.TOP_K
    )

    return {
        "arama_sorgusu": arama_sorgusu,
        "ham_sonuc":     sonuc,
    }


# ============================================================
# NODE 3: Yeterlilik düğümü — "ChromaDB yetti mi?"
# ============================================================

def yeterlilik_dugumu(state: HybridAgentState) -> dict:
    """
    ChromaDB'den dönen sonuçların soruyu gerçekten yanıtlamaya yetip
    yetmediğini iki aşamada kontrol eder:

    AŞAMA 1 — SAYISAL KONTROL (hızlı):
      Hiç sonuç yoksa veya çok azsa direkt web'e geç.

    AŞAMA 2 — LLM KONTROLÜ (akıllı):
      ChromaDB her zaman top-k sonuç döndürür — "5 sonuç var"
      olması sorunun cevabının orada olduğu anlamına gelmez.
      (Örneğin "yapay zeka yasal düzenlemeleri" sorusuna KVKK
      genel maddeleri benzerlik skoru alabilir ama aslında
      hiç alakalı değildir.)

      Bu yüzden LLM'e "bu bağlam soruyu cevaplıyor mu?" diye
      soruyoruz — evet ise RAG, hayır ise web araması.
    """
    ham_sonuc = state.get("ham_sonuc", {})
    soru      = state["soru"]
    docs      = ham_sonuc.get("documents", [[]])[0]

    # --- AŞAMA 1: Sayısal kontrol ---
    if len(docs) == 0:
        print("[Yeterlilik] ❌ Hiç sonuç bulunamadı → web araması tetikleniyor.")
        return {"rag_yeterli": False}

    # --- AŞAMA 2: LLM tabanlı anlam kontrolü ---
    # İlk 4 dokümanın genişletilmiş özetini bağlam olarak ver (daha doğru karar için)
    baglam_ozeti = "\n---\n".join(d[:1000] for d in docs[:4])

    prompt = f"""Aşağıdaki BAĞLAM bilgisi, bir kullanıcının sorusunu
cevaplamak için bir veri tabanından getirildi.

Soru: {soru}

Bağlam:
{baglam_ozeti}

Bu bağlam, soruyu DOĞRUDAN ve YETERLİ ŞEKİLDE cevaplıyor mu?

Kurallar:
- Bağlamda soruyla doğrudan ilgili spesifik bilgi varsa → EVET
- Bağlam genel/alakasız bilgiler içeriyor, soruya doğrudan yanıt yoksa → HAYIR
- Bağlam "bu konuda bilgi yok" diyorsa veya tamamen farklı konudaysa → HAYIR

SADECE tek kelime yaz: "EVET" veya "HAYIR". Başka hiçbir şey yazma."""

    yanit = _llm_answer.ollama_ile_cevap_uret(prompt).strip().upper()
    yeterli = "EVET" in yanit

    if yeterli:
        print(f"[Yeterlilik] ✅ LLM: Bağlam yeterli ({len(docs)} sonuç) → cevap üretiliyor.")
    else:
        print(f"[Yeterlilik] ❌ LLM: Bağlam yetersiz/alakasız → web araması tetikleniyor.")

    return {"rag_yeterli": yeterli}


def yeterlilik_sonrasi_yonlendirme(state: HybridAgentState) -> str:
    """
    CONDITIONAL EDGE: Yeterlilik düğümünden çıkışta nereye gideceğimizi
    belirler.
    """
    return "cevap_uret" if state["rag_yeterli"] else "web"


# ============================================================
# NODE 4a: Cevap üret düğümü — ChromaDB yeterliyse
# ============================================================

def cevap_uret_dugumu(state: HybridAgentState) -> dict:
    """
    ChromaDB yeterliyse: Faz 2'deki reranking + cevap üretim
    mantığını çalıştırır.
    """
    arama_sorgusu = state["arama_sorgusu"]
    sonuc         = state["ham_sonuc"]

    koleksiyon = _llm_answer._retriever.koleksiyonu_ac()

    # Faz 2'deki reranking: embedding sırasını LLM ile düzelt
    en_alakali_madde = _faz2_agent.en_uygun_maddeyi_sec(arama_sorgusu, sonuc)
    sonuc = _faz2_agent.madde_parcalarini_genislet_hedefli(koleksiyon, sonuc, en_alakali_madde)
    cevap = _faz2_agent.belirtilen_madde_icin_cevap_uret(arama_sorgusu, sonuc, en_alakali_madde)

    kaynaklar = []
    for meta in sonuc["metadatas"][0]:
        kaynak_metni = f"{meta['madde_no']} (sayfa {meta['sayfa_no']})"
        if kaynak_metni not in kaynaklar:
            kaynaklar.append(kaynak_metni)

    return {
        "cevap":       cevap,
        "kaynaklar":   kaynaklar,
        "kaynak_turu": "rag",
    }


# ============================================================
# NODE 4b: Web düğümü — ChromaDB yetersizse
# ============================================================

def web_dugumu(state: HybridAgentState) -> dict:
    """
    ChromaDB yetersiz kaldığında web araması yapar.
    12_web_search.py'deki web_de_ara fonksiyonunu çağırır.
    
    ÖNEMLİ DÜZELTME: Doğal dilli uzun sorular (ör. "2026 yılında yapay zeka...")
    arama motorlarını (özellikle Tavily) yanıltabiliyor ve sadece "2026"
    gibi ilk kelimeye odaklanmalarına yol açabiliyordu.
    Bunu çözmek için: Aramaya göndermeden önce LLM ile "Anahtar Kelime" (Keyword)
    optimizasyonu yapıyoruz.
    """
    orijinal_sorgu = state["arama_sorgusu"]
    
    # --- Arama Optimizasyonu (Keyword Extraction) ---
    prompt = f"""Aşağıdaki soruyu bir arama motorunda (Google vb.) aratmak için 
en uygun 3-4 kelimelik anahtar kelimelere dönüştür.

Soru: {orijinal_sorgu}

ÖNEMLİ KURALLAR:
- Sadece anahtar kelimeleri yaz, başka hiçbir açıklama ekleme.
- Soru işaretleri veya dolgu kelimeleri (nelerdir, nedir, nasıl) KULLANMA.
- Konunun özüne odaklan (ör. "2026 yapay zeka güncel gelişmeler").

Arama Sorgusu:"""

    optimize_sorgu = _llm_answer.ollama_ile_cevap_uret(prompt).strip()
    # LLM bazen tırnak içine alabilir, onları temizleyelim
    optimize_sorgu = optimize_sorgu.replace('"', '').replace("'", "")
    
    print(f"[Web] Orijinal soru: '{orijinal_sorgu}'")
    print(f"[Web] Optimize arama sorgusu: '{optimize_sorgu}' ({_web_search._KULLANILAN_ARAC.upper()} ile)...")

    try:
        sonuclar = _web_search.web_de_ara(optimize_sorgu, maks_sonuc=5)
        print(f"[Web] {len(sonuclar)} web sonucu bulundu.")
    except RuntimeError as hata:
        print(f"[Web] ⚠️ Web araması başarısız: {hata}")
        sonuclar = []

    # Aramayı başarılı yapsa da yapmasa da state'e geri dönüyoruz
    # Not: LLM'in cevabı üretirken sorunun Orijinal halini hatırlaması
    # için arama_sorgusu'nu değiştirmiyoruz, o zaten State'de "soru" olarak var.
    return {"web_sonuclari": sonuclar}


# ============================================================
# NODE 5: Hibrit cevap düğümü — Web sonuçlarını LLM ile yanıtla
# ============================================================

def hibrit_cevap_dugumu(state: HybridAgentState) -> dict:
    """
    Web aramasından dönen sonuçları bağlam olarak kullanarak
    LLM ile nihai cevabı üretir.

    NOT (duzeltme): Bu duguma graf yapisinda HER ZAMAN "yeterlilik_dugumu"
    tarafindan dokuman icerigi "yetersiz/alakasiz" bulundugunda gelinir.
    Bu yuzden o ayni (az once reddedilen) dokuman icerigini nihai cevaba
    "kaynak" olarak tekrar eklemek TUTARSIZ olurdu - LLM'in kendi
    "bu alakasiz" kararini gormezden gelmis oluruz. Bu nedenle burada
    SADECE web sonuclari kullaniliyor, ChromaDB icerigi dahil edilmiyor.
    """
    soru          = state["soru"]
    web_sonuclari = state.get("web_sonuclari", [])

    if not web_sonuclari:
        return {
            "cevap":       "Ne dokümanlarda ne de web'de bu soruya yönelik bilgi bulunamadı.",
            "kaynaklar":   [],
            "kaynak_turu": "bulunamadi",
        }

    web_baglam = _web_search.sonuclari_baglama_donustur(web_sonuclari)

    prompt = f"""Aşağıdaki web kaynaklarına (snippet) dayanarak soruyu cevapla.

ÇOK ÖNEMLİ KURALLAR (HAYATİ ÖNEM TAŞIR):
1. SADECE aşağıdaki kaynak metinlerinde AÇIKÇA GEÇEN bilgileri kullan.
2. Hava durumu, derece, fiyat, tarih gibi sayısal verileri ASLA UYDURMA (Halüsinasyon yapma). Eğer kaynaklarda net bir sayı/derece yazmıyorsa "Verilen kaynaklarda bu bilgi bulunmamaktadır" de.
3. Kaynaklar genel konulardan bahsediyor ama sorunun tam cevabını içermiyorsa, tahminde bulunmak yerine "Kaynaklar yetersiz" olduğunu açıkça belirt.
4. ÇOK ÖNEMLİ: Cevabına KESİNLİKLE "Yukarıdaki kaynaklara göre" veya "Verilen kaynaklara göre" gibi yön belirten cümlelerle BAŞLAMA.

Kaynaklar:
{web_baglam}

Soru: {soru}

Nihai Cevap:"""

    cevap = _llm_answer.ollama_ile_cevap_uret(prompt).strip()

    # GARANTILI TEMIZLEME: Prompt'a "kaynaklara gore diye baslama" talimati
    # eklenmis olsa bile kucuk model bunu bazen gormezden gelip yine de o
    # sekilde basliyor (gercek testte gorüldü). Bu yuzden prompt'a GUVENMEK
    # YERINE, ciktinin basindaki bu kaliplasmis ifadeleri KODDAN kesin
    # olarak siliyoruz - "arayuzde kaynaklar cevabin ALTINDA oldugu icin
    # 'yukaridaki/verilen kaynaklara gore' demek yaniltici.
    cevap = re.sub(
        r'^\s*(?:Yukarıdaki|Aşağıdaki|Verilen|Bu)\s+kaynaklar[ae]?\s+(?:göre|dayanarak|dayanılarak)[,;:]?\s*',
        "",
        cevap,
        flags=re.IGNORECASE,
    )
    # Bazen bu ifade cumle icinde "Kaynaklara gore, ..." seklinde de gelir
    cevap = re.sub(r'^\s*Kaynaklar[ae]?\s+göre[,;:]?\s*', "", cevap, flags=re.IGNORECASE)
    # Ilk harfi buyut (temizlik sonrasi kucuk harfle baslamis olabilir)
    cevap = cevap.strip()
    if cevap:
        cevap = cevap[0].upper() + cevap[1:]

    web_kaynaklari = [
        f"{s['baslik']} — {s['url']}"
        for s in web_sonuclari
        if s.get("url")
    ]

    return {
        "cevap":       cevap,
        "kaynaklar":   web_kaynaklari,
        "kaynak_turu": "web",
    }


# ============================================================
# NODE (Faz 2'den devralınan): Doğrudan cevap düğümü
# ============================================================

def dogrudan_cevap_dugumu(state: HybridAgentState) -> dict:
    return {
        "cevap": (
            "Bu sistem şu an sadece Kişisel Verilerin Korunması Kanunu (KVKK) "
            "ile ilgili sorulara cevap verebiliyor. Başka bir konuda "
            "yardımcı olamıyorum."
        ),
        "kaynaklar":   [],
        "kaynak_turu": "yok",
    }


# ============================================================
# GRAPH — Düğüm ve kenarları birleştiriyoruz
# ============================================================

def hybrid_agent_olustur():
    graph = StateGraph(HybridAgentState)

    # Düğümler
    graph.add_node("karar",         karar_dugumu)
    graph.add_node("rag",           rag_dugumu)
    graph.add_node("yeterlilik",    yeterlilik_dugumu)
    graph.add_node("cevap_uret",    cevap_uret_dugumu)
    graph.add_node("web",           web_dugumu)
    graph.add_node("hibrit_cevap",  hibrit_cevap_dugumu)
    graph.add_node("dogrudan",      dogrudan_cevap_dugumu)

    # NOT (duzeltme, gercek test sirasinda bulundu): Baslangic noktasi
    # artik DOGRUDAN "rag" - "karar" dugumunun sert "KVKK ile ilgili mi/
    # degil mi" siniflandirmasi ATLANIYOR.
    #
    # Nedeni: Faz 2'de sistem SADECE KVKK sorularina cevap veren bir
    # asistandi, bu yuzden "alakasiz" sorulari en bastan reddetmek
    # mantikliydi. Ama Faz 4'te sistem artik "dokuman yetersiz kalirsa
    # web'e bak" diyen DAHA GENIS bir hybrid sistem - roadmap'in kendi
    # tanimina gore ("Dokumanin yetersiz kaldigi durumlarda otomatik
    # olarak web'e basvuran hybrid RAG sistemi") bu davranis herhangi
    # bir konu kisitlamasi icermiyor. Eski akista, "2026'da yapay zeka
    # gelismeleri" gibi KVKK-disi bir soru "karar_dugumu"nde en bastan
    # reddediliyordu, web aramasina HIC ulasamiyordu.
    #
    # Simdi HER soru once RAG'a gidiyor; "yeterlilik_dugumu" (zaten var
    # olan, daha akilli bir kontrol) dokumanin o soruyu cevaplayip
    # cevaplamadigina karar veriyor - yetersizse otomatik web'e dusuyor.
    # "karar_dugumu" ve "dogrudan_cevap_dugumu" fonksiyonlari SILINMEDI
    # (ileride farkli bir kullanim senaryosunda geri getirilebilir),
    # sadece graf akisindan CIKARILDI.
    graph.set_entry_point("rag")

    graph.add_edge("rag", "yeterlilik")

    graph.add_conditional_edges(
        "yeterlilik",
        yeterlilik_sonrasi_yonlendirme,
        {"cevap_uret": "cevap_uret", "web": "web"}
    )

    graph.add_edge("cevap_uret",   END)
    graph.add_edge("web",          "hibrit_cevap")
    graph.add_edge("hibrit_cevap", END)
    graph.add_edge("dogrudan",     END)

    return graph.compile()


# ============================================================
# Doğrudan çalıştırma — sohbet modu
# ============================================================

if __name__ == "__main__":
    agent  = hybrid_agent_olustur()
    gecmis = []   # Session Memory

    print("=" * 60)
    print("Hybrid RAG Agent — Faz 4")
    print(f"Web arama aracı: {_web_search._KULLANILAN_ARAC.upper()}")
    print("Çıkmak için 'q' yazın.")
    print("=" * 60)

    ilk_soru = sys.argv[1] if len(sys.argv) >= 2 else None

    while True:
        if ilk_soru is not None:
            soru    = ilk_soru
            ilk_soru = None
        else:
            soru = input("\nSoru: ").strip()

        if soru.lower() in ("q", "çıkış", "cikis", "exit"):
            print("Görüşürüz!")
            break
        if not soru:
            continue

        baslangic_state = {
            "soru":          soru,
            "arac_karari":   "",
            "cevap":         "",
            "kaynaklar":     [],
            "gecmis":        gecmis,
            "arama_sorgusu": "",
            "ham_sonuc":     {},
            "rag_yeterli":   True,
            "web_sonuclari": [],
            "kaynak_turu":   "",
        }

        sonuc_state = agent.invoke(baslangic_state)

        print("\n" + "=" * 60)
        print(f"SORU: {soru}")
        print("=" * 60)
        print(f"\nCEVAP:\n{sonuc_state['cevap']}")

        if sonuc_state["kaynaklar"]:
            print("\n" + "-" * 60)
            kaynak_turu = sonuc_state.get("kaynak_turu", "")
            etiket = {
                "rag":       "📄 Doküman Kaynakları",
                "web":       "🌐 Web Kaynakları",
                "hibrit":    "📄🌐 Hibrit Kaynaklar (Doküman + Web)",
                "bulunamadi": "⚠️  Kaynak",
            }.get(kaynak_turu, "Kaynaklar")

            print(f"{etiket}:")
            for k in sonuc_state["kaynaklar"]:
                print(f"  - {k}")

        gecmis.append((soru, sonuc_state["cevap"]))
        print(f"\n🤖 Task 4.2 — Hybrid RAG yanıtı tamamlandı! "
              f"(kaynak: {sonuc_state.get('kaynak_turu', '?').upper()})")
