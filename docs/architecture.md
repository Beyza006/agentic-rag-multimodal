# Task 0.5 — Sistem Mimarisi

## 1. Kullanıcı İsteği Akışı (uçtan uca)

1. Kullanıcı arayüze (Gradio) bir girdi sağlar: metin sorusu + isteğe bağlı dosya (PDF/Word/görsel/video)
2. Dosya varsa, türüne göre ön işleme yapılır:
   - PDF/Word → **Document Loader** (PyMuPDF/python-docx) → **Text Extraction** → Chunking → Embedding → ChromaDB → **Retriever**
   - Görsel → doğrudan VLM'e iletilecek şekilde hazırlanır
   - Video → **Frame Extraction** (OpenCV) → **Frame Selection** (soruyla ilgili karelerin seçilmesi — tüm frame'ler gönderilmez)
3. Kullanıcının sorusu ve hazırlanan bağlam **Agent**'a (LangGraph orkestrasyonu) iletilir
4. Agent (Planner → Tool Selector → Tool Executor) hangi aracı/araçları kullanacağına karar verir
5. Seçilen araç(lar) çalıştırılır, sonuçlar toplanır; Memory bu süreçte Agent'a bağlam sağlar
6. LLM, toplanan bağlamı kullanarak nihai yanıtı (kaynak göstererek) üretir
7. Yanıt + kullanılan araç/kaynak bilgisi Gradio arayüzünde kullanıcıya gösterilir

## 2. Agent Workflow (karar noktaları)

```
Soru geldi
  ├─ Doküman/metin sorusu mu? → RAG Tool (Retriever sonucu + LLM)
  ├─ Görsel/grafik sorusu mu? → Vision Tool (VLM)
  ├─ Video ile ilgili mi?
  │     ├─ Tek an sorusu → seçili tek frame + VLM
  │     └─ Zaman/olay sorusu → Frame Selection ile seçilen çoklu frame + VLM + LLM (Temporal Reasoning)
  ├─ Güncel/harici bilgi mi gerekiyor? → Web Search Tool
  └─ Doküman bulundu mu? Hayır / benzerlik (similarity) skoru düşük mü? → Web Search'e fallback (Hybrid RAG)
```

Bu karar mekanizması ReAct (Reasoning + Acting) mantığıyla çalışır. Agent kendi içinde
üç alt bileşenden oluşur:

```
Agent
  ↓
Planner        (soruyu analiz eder, hangi adımların gerektiğine karar verir)
  ↓
Tool Selector   (uygun aracı/araçları seçer: RAG / VLM / Web Search)
  ↓
Tool Executor   (seçilen aracı çalıştırır, sonucu toplar)
  ↓
LLM             (toplanan bağlamla nihai yanıtı sentezler)
```

**Not — Doküman yeterliliği kararı:** Agent aslında "doküman yetersiz" diye öznel bir
yargıya varmaz; retriever'dan dönen sonuçların similarity skoru bir eşik değerin (threshold)
altında kalırsa ya da hiç sonuç dönmezse, bu durum Web Search'e fallback için tetikleyici olur.

**Not — MCP:** İlk prototipte standart Tool Calling kullanılacaktır. MCP (Model Context
Protocol) desteği, agent mimarisi olgunlaştıktan sonra ilerleyen aşamalarda eklenebilir.

## 3. Kullanılan Bileşenler

| Katman | Teknoloji | Not |
|---|---|---|
| Arayüz | Gradio | Sohbet + dosya yükleme + streaming + log paneli |
| Document Processing | LangChain Document Loader / PyMuPDF / python-docx | PDF ve Word dosyalarının okunması ve metin çıkarımı |
| Orkestrasyon | LangGraph | State/Node/Edge tabanlı agent workflow (Planner/Tool Selector/Tool Executor) |
| Araç çağırma | Tool Calling (standart) | MCP ilerleyen aşamada değerlendirilecek |
| LLM | Bulut API (Anthropic/OpenAI) veya Ollama (yerel) | Karşılaştırmalı değerlendirilecek |
| VLM | Qwen2.5-VL / LLaVA — API veya Ollama üzerinden | Model seçimi prototip aşamasında |
| Vector DB + Retriever | ChromaDB + Retriever (similarity/MMR search) | Faz 0.3'te DB kararı verildi |
| Embedding modeli | BAAI/bge-m3 veya Sentence Transformers | Faz 1'de netleşecek |
| Web Search | Tavily / SerpAPI | Hybrid RAG fallback için |
| Video işleme | OpenCV (Frame Extraction + Frame Selection) | Tüm frame değil, ilgili frame(ler) seçilir |
| Memory | Session / Persistent / User Profile / Retrieval Memory | Agent tarafından yönetilir (Faz 2.4) |
| Değerlendirme | RAGAS / DeepEval | Faz 6 |
| Deployment | FastAPI + Gradio + Docker Compose + Ollama (opsiyonel) | Faz 7, bkz. aşağıdaki hiyerarşi |

**Deployment hiyerarşisi (Faz 7):**
```
FastAPI (Backend)
   ↓
Gradio (Frontend)
   ↓
Docker Compose  (tüm servisleri paketleyip birlikte ayağa kaldıran orkestrasyon katmanı)
   ↓
Ollama (Opsiyonel, yerel LLM/VLM için)
```
Not: Bu sıralama fonksiyonel bir bağımlılık zinciri değil, konteynerler arası ilişkiyi
gösterir — Docker Compose, FastAPI/Gradio/Ollama container'larını birlikte yönetir.

## 4. Mermaid Diyagramı

```mermaid
flowchart TD
    U[Kullanıcı: Soru + Dosya] --> P{Dosya Türü?}
    P -->|PDF/Word| DL[Document Loader] --> TXE[Text Extraction] --> CH[Chunking] --> EMB[Embedding] --> VDB[(ChromaDB)] --> RET[Retriever]
    P -->|Görsel| VI[Görsel Ön İşleme]
    P -->|Video| FE[Frame Extraction] --> FS[Frame Selection]
    P -->|Yok, sadece metin| AG

    RET --> AG[Agent: Planner]
    VI --> AG
    FS --> AG

    MEM[(Memory: Session/Persistent)] --> AG

    AG --> TS[Tool Selector]
    TS --> D{Karar}
    D -->|Doküman sorusu| RAG[RAG Tool: Retriever Sonucu]
    D -->|Görsel/Grafik sorusu| VLM[Vision Tool: VLM]
    D -->|Video - tek an| VLM
    D -->|Video - zaman/olay| TR[Temporal Reasoning: Seçili Frameler + VLM]
    D -->|Doküman bulunamadı / confidence düşük| WS[Web Search Tool]

    RAG --> TEX[Tool Executor]
    VLM --> TEX
    TR --> TEX
    WS --> TEX

    TEX --> MEM
    TEX --> LLM[LLM: Yanıt Sentezi]
    LLM --> R[Yanıt + Kaynak + Kullanılan Araç]
    R --> UI[Gradio Arayüzü]
```
