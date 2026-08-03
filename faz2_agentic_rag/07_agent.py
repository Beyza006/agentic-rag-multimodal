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


# ============================================================
# NODE 1: Karar dugumu - "Tool Calling" mantiginin basitlestirilmis hali
# ============================================================

def karar_dugumu(state: AgentState) -> dict:
    """
    Soruyu LLM'e gonderip "KVKK ile ilgili mi, alakasiz mi" diye siniflandirir.
    Bu, gercek tool-calling sistemlerindeki "LLM hangi araci secmeli"
    kararinin basitlestirilmis bir versiyonu - ileride (Faz 4) buraya
    "web_search" gibi baska secenekler de eklenebilir.
    """
    soru = state["soru"]
    prompt = f"""Aşağıdaki soru, Kişisel Verilerin Korunması Kanunu (KVKK) veya
kişisel veri koruma konusuyla mı ilgili, yoksa tamamen ALAKASIZ bir konu mu
(örn. hava durumu, yemek tarifi, spor vb.)?

Soru: {soru}

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

def rag_dugumu(state: AgentState) -> dict:
    """
    Faz 1'de kurdugumuz retriever + generation pipeline'ini oldugu gibi
    kullanir - Task 2.3'un amaci Faz 1'i YENIDEN YAZMAK degil, onu bir
    "arac" (tool) olarak agent'a BAGLAMAK.
    """
    soru = state["soru"]

    koleksiyon = _llm_answer._retriever.koleksiyonu_ac()
    embedding_modeli = _llm_answer._retriever.modeli_yukle()
    sonuc = _llm_answer._retriever.ara(embedding_modeli, koleksiyon, soru, top_k=_llm_answer.TOP_K)

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
        cevap = _llm_answer.kod_tabanli_nihai_cevap_olustur(soru, en_alakali_madde_no, bentler)
    else:
        liste_prompti = _llm_answer.eksiksiz_liste_prompti_olustur(soru, sonuc)
        eksiksiz_liste = _llm_answer.ollama_ile_cevap_uret(liste_prompti)
        akici_prompt = _llm_answer.akici_hale_getir_prompti_olustur(soru, eksiksiz_liste)
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
    if len(sys.argv) < 2:
        print("Kullanim: python 07_agent.py \"<soru>\"")
        print("Ornek:    python 07_agent.py \"Kişisel veriler ne zaman açık rıza olmadan işlenebilir?\"")
        sys.exit(1)

    soru = sys.argv[1]

    agent = agent_olustur()

    baslangic_state = {"soru": soru, "arac_karari": "", "cevap": "", "kaynaklar": []}
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

    print("\n\U0001F916 Task 2.3 - Agent workflow tamamlandi!")
