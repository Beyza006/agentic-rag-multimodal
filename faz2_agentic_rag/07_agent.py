# -*- coding: utf-8 -*-
"""
Task 2.3 - LangGraph ile Agent Workflow
------------------------------------------
Bu script, Faz 1'de kurdugumuz SABIT workflow'u (PDF -> chunk -> embed ->
ara -> cevap uret, her zaman ayni sirada calisan) bir ADIM ileri tasiyip
gercek bir AGENT'a donusturur.

Faz 1 ile Faz 2 arasindaki fark (Task 2.1'de ogrendigimiz kavram):
- Faz 1: WORKFLOW - her soru icin ayni sabit adimlar calisir.
- Faz 2: AGENT - sistem, soruya bakip HANGI aracin cagrilacagina
  CALISMA ZAMANINDA (runtime) kendisi karar verir.

Bu ilk agent'ta SADECE 2 dugum var (basit tutmak icin):
  1) karar_dugumu: soru KVKK ile mi ilgili, alakasiz mi - LLM'e sorarak
     karar verir (bu, Task 2.1'de ogrendigimiz "Tool Calling" mantiginin
     basitlestirilmis bir versiyonu - LLM burada "hangi arac" kararini
     yapisal bir sekilde veriyor).
  2) rag_dugumu VEYA dogrudan_cevap_dugumu: karara gore CALISTIRILAN dugum.

LangGraph kavramlari (Task 2.3):
- State: dugumler arasinda paylasilan, okunup yazilabilen ortak veri yapisi.
- Node: bir is birimini temsil eden fonksiyon (State al, State guncelle).
- Edge: iki dugum arasindaki sabit gecis.
- Conditional Edge: bir dugumden CIKISTA, State'teki bir degere gore HANGI
  dugume gidilecegine karar veren dinamik gecis - bu, agent'in "karar
  verme" yetenegini saglayan asil mekanizma.

Kullanim:
    python 07_agent.py "Kişisel veriler ne zaman açık rıza olmadan işlenebilir?"
    python 07_agent.py "Bugün hava nasıl?"
"""

import sys
import os
import importlib.util
from typing import TypedDict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from langgraph.graph import StateGraph, END


# --- Faz 1'deki modulleri import ediyoruz (dosya adlari rakamla basladigi
# icin importlib ile dosya yolundan yukluyoruz, ayni Faz 1'deki gibi) ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_faz1_dir = os.path.join(_current_dir, "..", "faz1_klasik_rag")


def _modul_yukle(dosya_adi, modul_adi):
    yol = os.path.join(_faz1_dir, dosya_adi)
    spec = importlib.util.spec_from_file_location(modul_adi, yol)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


_retriever = _modul_yukle("05_retriever.py", "retriever_module")
_llm_answer = _modul_yukle("06_llm_answer.py", "llm_answer_module")


# ============================================================
# STATE: Agent'in dugumler arasinda tasidigi ortak veri yapisi
# ============================================================

class AgentState(TypedDict):
    soru: str            # kullanicinin girdisi
    arac_karari: str      # "rag" veya "dogrudan" - karar dugumunun cikardigi sonuc
    cevap: str            # nihai cevap
    kaynaklar: list       # kullanilan madde/sayfa referanslari (varsa)
    gecmis: list          # [(soru, cevap), ...] - Task 2.4: SESSION MEMORY
                          # Bu oturum boyunca sorulan onceki soru-cevaplari
                          # tutar. Programi kapatinca kaybolur (kalici degil) -
                          # bu yuzden "Session Memory", "Persistent Memory" degil.


# ============================================================
# NODE 1: Karar dugumu - "Tool Calling" mantiginin basitlestirilmis hali
# ============================================================

def karar_dugumu(state: AgentState) -> dict:
    """
    Soruyu LLM'e gonderip "KVKK ile ilgili mi, alakasiz mi" diye siniflandirir.

    TASK 2.4 GUNCELLEMESI: Artik SADECE mevcut soruya degil, gecmis
    (Session Memory) varsa ONA da bakarak karar veriyor. Boylece "peki
    cezasi nedir?" gibi, tek basina belirsiz duran ama bir onceki KVKK
    sorusunun DEVAMI olan bir soru da dogru siniflandirilabiliyor -
    hafizasiz bir agent bunu yanlislikla "ALAKASIZ" sanabilirdi.
    """
    soru = state["soru"]
    gecmis = state.get("gecmis", [])

    baglam_metni = ""
    if gecmis:
        son_soru, son_cevap = gecmis[-1]
        baglam_metni = f"""
ONCEKI KONUSMA BAGLAMI (varsa yeni soru buna gore degerlendirilmeli):
Onceki soru: {son_soru}
Onceki cevabin ozeti: {son_cevap[:200]}
"""

    prompt = f"""Aşağıdaki soru, Kişisel Verilerin Korunması Kanunu (KVKK) veya
kişisel veri koruma konusuyla mı ilgili, yoksa tamamen ALAKASIZ bir konu mu
(örn. hava durumu, yemek tarifi, spor vb.)?
{baglam_metni}
Yeni soru: {soru}

ÖNEMLİ: Eğer yeni soru, önceki konuşmanın DEVAMI niteliğindeyse (örn.
"peki cezası nedir?", "bu durumda ne olur?" gibi, önceki KVKK konusuna
atıfta bulunan takip sorularıysa), bunu da İLGİLİ say.

SADECE tek bir kelime yaz: "ILGILI" veya "ALAKASIZ". Başka hiçbir şey yazma."""

    yanit = _llm_answer.ollama_ile_cevap_uret(prompt).strip().upper()

    if "ALAKASIZ" in yanit:
        karar = "dogrudan"
    else:
        karar = "rag"

    print(f"[Agent karari] Soru '{'KVKK ile ilgili' if karar == 'rag' else 'alakasiz'}' olarak siniflandirildi -> '{karar}' dugumune yonlendiriliyor.")

    return {"arac_karari": karar}


def yonlendirme_fonksiyonu(state: AgentState) -> str:
    """
    LangGraph'in CONDITIONAL EDGE mekanizmasi bu fonksiyonu cagirir ve
    donen string'e gore bir sonraki dugumu secer (asagidaki add_conditional_edges
    cagrisindaki sozluk eslesmesine bakar).
    """
    return state["arac_karari"]


# ============================================================
# NODE 2a: RAG dugumu - Faz 1'in tum pipeline'ini calistirir
# ============================================================

def sorguyu_baglamla_zenginlestir(soru: str, gecmis: list) -> str:
    """
    TASK 2.4: Eger kullanicinin sorusu bir onceki konusmaya atifta
    bulunuyorsa (orn. "peki cezasi nedir?" - "ceza" kelimesi tek basina
    hangi konudaki cezayi kastettigini belli etmez), bu soruyu ARAMA
    ICIN daha isabetli, BAGIMSIZ bir sorguya donusturuyoruz. Bu,
    gercek RAG sistemlerinde "conversational query rewriting" (query
    condensation) olarak bilinen, yaygin kullanilan bir tekniktir.

    Eger gecmis yoksa, soru oldugu gibi kullanilir - bu adim sadece
    takip sorularinda devreye girer.
    """
    if not gecmis:
        return soru

    son_soru, _ = gecmis[-1]
    prompt = f"""Önceki soru: {son_soru}
Yeni soru: {soru}

Yeni soru, önceki sorunun devamı/takibi niteliğindeyse, bu iki soruyu
birleştirerek TEK BAŞINA anlaşılır, bağımsız bir arama sorgusu oluştur
(örn. önceki soru "açık rıza olmadan ne zaman işlenir" ve yeni soru
"peki cezası nedir" ise, birleşik sorgu "açık rızasız veri işlemenin
cezası nedir" gibi olmalı). Eğer yeni soru zaten kendi başına
anlaşılırsa, sadece yeni soruyu aynen döndür.

SADECE oluşturulan sorguyu yaz, başka hiçbir açıklama ekleme."""

    return _llm_answer.ollama_ile_cevap_uret(prompt).strip()


def rag_dugumu(state: AgentState) -> dict:
    """
    Faz 1'de kurdugumuz retriever + generation pipeline'ini oldugu gibi
    kullanir - Task 2.3'un amaci Faz 1'i YENIDEN YAZMAK degil, onu bir
    "arac" (tool) olarak agent'a BAGLAMAK.
    """
    soru = state["soru"]
    gecmis = state.get("gecmis", [])

    # TASK 2.4: Takip sorularinda, aramayi gecmis baglamla zenginlestir
    arama_sorgusu = sorguyu_baglamla_zenginlestir(soru, gecmis)
    if arama_sorgusu != soru:
        print(f"[Memory] Takip sorusu tespit edildi, arama sorgusu zenginlestirildi: '{arama_sorgusu}'")

    koleksiyon = _llm_answer._retriever.koleksiyonu_ac()
    embedding_modeli = _llm_answer._retriever.modeli_yukle()
    sonuc = _llm_answer._retriever.ara(embedding_modeli, koleksiyon, arama_sorgusu, top_k=_llm_answer.TOP_K)

    if not sonuc["documents"][0]:
        return {
            "cevap": "Bu soruyla ilgili dokümanlarda bir bilgi bulunamadı.",
            "kaynaklar": []
        }

    sonuc = _llm_answer.madde_parcalarini_genislet(koleksiyon, sonuc)

    en_alakali_madde_no = sonuc["metadatas"][0][0]["madde_no"]
    ilgili_kayitlar = sorted(
        [
            (doc, meta) for doc, meta in zip(sonuc["documents"][0], sonuc["metadatas"][0])
            if meta["madde_no"] == en_alakali_madde_no
        ],
        key=lambda x: x[1]["chunk_index"]
    )

    # Bkz. 06_llm_answer.py'deki ayni duzeltmenin aciklamasi: alt-chunklar
    # arasindaki kasitli ORTUSME nedeniyle ham metni dogrudan birlestirmek
    # yerine, her chunk'tan bentleri ayri cikarip harf bazinda dedup ediyoruz.
    TURKCE_ALFABE = "abcçdefgğhıijklmnoöprsştuüvyz"

    def alfabe_sira_no(harf):
        try:
            return TURKCE_ALFABE.index(harf)
        except ValueError:
            return 999

    tum_bentler_sozluk = {}
    for doc, meta in ilgili_kayitlar:
        for harf, icerik in _llm_answer.bentleri_cikar(doc):
            if harf not in tum_bentler_sozluk or len(icerik) > len(tum_bentler_sozluk[harf]):
                tum_bentler_sozluk[harf] = icerik

    bentler = sorted(tum_bentler_sozluk.items(), key=lambda x: alfabe_sira_no(x[0]))

    if len(bentler) >= 2:
        cevap = _llm_answer.kod_tabanli_nihai_cevap_olustur(arama_sorgusu, en_alakali_madde_no, bentler)
    else:
        liste_prompti = _llm_answer.eksiksiz_liste_prompti_olustur(arama_sorgusu, sonuc)
        eksiksiz_liste = _llm_answer.ollama_ile_cevap_uret(liste_prompti)
        akici_prompt = _llm_answer.akici_hale_getir_prompti_olustur(arama_sorgusu, eksiksiz_liste)
        cevap = _llm_answer.ollama_ile_cevap_uret(akici_prompt)

    cevap = _llm_answer.cevabi_temizle(cevap)

    kaynaklar = [
        f"{meta['madde_no']} (sayfa {meta['sayfa_no']})"
        for meta in sonuc["metadatas"][0]
    ]

    return {"cevap": cevap, "kaynaklar": kaynaklar}


# ============================================================
# NODE 2b: Dogrudan cevap dugumu - RAG'a hic gerek olmadigi durumlar icin
# ============================================================

def dogrudan_cevap_dugumu(state: AgentState) -> dict:
    """
    Soru KVKK ile alakasizsa, hicbir arama/embedding islemi yapmadan
    (gereksiz maliyetten kacinarak) kullaniciyi bilgilendiren sabit bir
    cevap doner. Bu, agent'in "her zaman RAG calistirma" yerine "gerekliyse
    calistir" mantiginin somut faydasini gosteriyor.
    """
    return {
        "cevap": (
            "Bu sistem şu an sadece Kişisel Verilerin Korunması Kanunu (KVKK) "
            "ile ilgili sorulara cevap verebiliyor. Başka bir konuda "
            "yardımcı olamıyorum."
        ),
        "kaynaklar": []
    }


# ============================================================
# GRAPH: State/Node/Edge/Conditional Edge'leri birlestirip agent'i kuruyoruz
# ============================================================

def agent_olustur():
    graph = StateGraph(AgentState)

    graph.add_node("karar", karar_dugumu)
    graph.add_node("rag", rag_dugumu)
    graph.add_node("dogrudan", dogrudan_cevap_dugumu)

    graph.set_entry_point("karar")

    # CONDITIONAL EDGE: karar dugumunden cikista, yonlendirme_fonksiyonu'nun
    # dondurdugu degere gore "rag" ya da "dogrudan" dugumune gidilir.
    graph.add_conditional_edges(
        "karar",
        yonlendirme_fonksiyonu,
        {"rag": "rag", "dogrudan": "dogrudan"}
    )

    graph.add_edge("rag", END)
    graph.add_edge("dogrudan", END)

    return graph.compile()


if __name__ == "__main__":
    agent = agent_olustur()
    gecmis = []  # Task 2.4: Session Memory - program calistigi surece hatirlanir

    print("=" * 60)
    print("KVKK Agent - Sohbet Modu (Task 2.4: Session Memory ile)")
    print("Cikmak icin 'q' veya 'çıkış' yazabilirsiniz.")
    print("=" * 60)

    # Eger komut satirindan dogrudan bir soru verildiyse (eski kullanim
    # sekliyle geriye-uyumluluk), onu ilk soru olarak isle, sonra sohbete
    # devam et.
    ilk_soru = sys.argv[1] if len(sys.argv) >= 2 else None

    while True:
        if ilk_soru is not None:
            soru = ilk_soru
            ilk_soru = None
        else:
            soru = input("\nSoru: ").strip()

        if soru.lower() in ("q", "çıkış", "cikis", "exit"):
            print("Görüşürüz!")
            break
        if not soru:
            continue

        baslangic_state = {
            "soru": soru,
            "arac_karari": "",
            "cevap": "",
            "kaynaklar": [],
            "gecmis": gecmis
        }
        sonuc_state = agent.invoke(baslangic_state)

        print("\n" + "=" * 60)
        print(f"SORU: {soru}")
        print("=" * 60)
        print(f"\nCEVAP:\n{sonuc_state['cevap']}")

        if sonuc_state["kaynaklar"]:
            print("\n" + "-" * 60)
            print("Kullanilan kaynaklar:")
            for k in sonuc_state["kaynaklar"]:
                print(f"  - {k}")

        # Task 2.4: Bu tur, bir sonraki turda hatirlanmak uzere gecmise eklenir
        gecmis.append((soru, sonuc_state["cevap"]))

        print("\n\U0001F916 Task 2.4 - Memory ile agent yaniti tamamlandi!")
