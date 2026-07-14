# Agentic RAG + Multimodal Video QA — Staj Proje Yol Haritası

> TSE stajı kapsamında hazırlanan, Jetson AI Lab görevinden ilham alan ve donanımsız
> (PC/API tabanlı) olarak uyarlanmış birleşik proje planı.
>
> **Etiketler:** 🟢 Çekirdek (mutlaka bitmeli) — 🟡 Önemli (zaman kalırsa) — ⚪ Stretch/Opsiyonel (gelecek çalışma)

---

## Proje Senaryosu (Use Case)

**Problem Tanımı:**
Kullanıcı PDF, Word, görsel veya video yükleyebilir. Sistem, sorunun niteliğini analiz
ederek uygun aracı (RAG, Web Search, Vision-Language Model veya bunların kombinasyonu)
otomatik seçer. Gerekli bilgileri toplar, çok adımlı akıl yürütme gerçekleştirir ve
kaynak göstererek güvenilir yanıt üretir.

Bu paragraf projenin amacını ilk bakışta anlatır; rapor ve sunum yazarken buraya geri dön.

---

## Faz Bazlı Teslim Çıktıları (özet)

| Faz | Teslim Çıktısı |
|---|---|
| Faz 0 | Araştırma notları + sistem mimarisi (Mermaid diyagramı + teknoloji listesi) |
| Faz 1 | Çalışan PDF tabanlı RAG (yükle → soru sor → grounded cevap) |
| Faz 2 | Agentic RAG: tool calling ile çalışan, doğru aracı otomatik seçen agent |
| Faz 3 | Multimodal demo: görsel soru-cevap + video frame QA (Gradio üzerinde) |
| Faz 4 | Hybrid RAG: doküman yetersizse web search'e geçen agent |
| Faz 5 | Kullanılabilir Gradio arayüzü: sohbet + yükleme + streaming + kaynak gösterme |
| Faz 6 | Değerlendirme raporu: RAGAS/DeepEval metrikleriyle sistem performansı |
| Faz 7 | Docker (Compose) ile tek komutla ayağa kalkan, çalıştırılabilir proje |

> Her fazın sonunda yukarıdaki çıktı elinde yoksa faz "tamamlandı" sayılmaz — bir sonrakine geçmeden önce kontrol et.

---

## Faz 0 — Temel Araştırma, Literatür ve Mimari Tasarım

### Task 0.1 — LLM Temellerini Tekrar Et 🟢
- [ ] Transformer
- [ ] Attention Mechanism
- [ ] Token / Context Window
- [ ] Prompt Engineering
- [ ] Fine-tuning / Inference
- **Kaynaklar:** Attention Is All You Need, Illustrated Transformer (Jay Alammar), Hugging Face NLP Course
- **Çıktı:** 2-3 sayfalık özet + kavram notları

### Task 0.2 — Embedding Kavramı 🟢
- [ ] Embedding nedir?
- [ ] Semantic / Similarity Search
- [ ] Cosine Similarity
- [ ] Vector Search
- **Kaynaklar:** Pinecone Learn, ChromaDB Docs, Sentence Transformers
- **Çıktı:** Embedding mantığını açıklayan kısa not

### Task 0.3 — Vector Database 🟢
- [ ] ChromaDB
- [ ] FAISS
- [ ] Milvus
- [ ] Weaviate
- **Çıktı:** Avantaj/dezavantaj/kullanım alanı karşılaştırma tablosu

### Task 0.4 — Multimodal & Benchmarking Kavramları 🟢 *(Jetson AI Lab arşivinden, donanımsız)*
- [ ] Text (LLM)
- [ ] Text + Vision (VLM)
- [ ] Gen AI Benchmarking
- [ ] Vision Transformers (ViT)
- [ ] Image Generation
- [ ] RAG & Vector Database (tekrar/derinleştirme)
- [ ] Audio
- **Kaynak:** jetson-ai-lab.com/archive/index.html
- **Çıktı:** Kısa notlar + sözlü/yazılı mini anlatım (en fazla 2 gün hedefle)

### Task 0.5 — Sistem Mimarisi Tasarımı 🟢 *(koda geçmeden önce)*
Araştırma bitince doğrudan kodlamaya geçmek yerine sistemi kağıt üzerinde tasarla:
- [ ] Kullanıcı isteğinin uçtan uca akışı
- [ ] Agent workflow (karar noktaları)
- [ ] Kullanılacak tool'lar
- [ ] Vector DB
- [ ] LLM
- [ ] VLM
- [ ] Web Search
- [ ] Memory katmanları
- **Çıktı:**
  - Mermaid diyagramı (sistem akışı)
  - Genel mimari çizimi
  - Kullanılacak teknolojilerin listesi
- **Not:** Bu çıktı ileride rapor/sunum yazarken doğrudan kullanılır — zaman kaybı değil, yatırım.

---

## Faz 1 — Klasik RAG 🟢

| Task | Konu | Araştırılacaklar |
|---|---|---|
| 1.1 | PDF yükleme | PyMuPDF, Unstructured, LangChain Document Loader — *taranmış PDF varsa OCR gerekecek, bkz. Faz 3.5* |
| 1.2 | Chunking | Recursive Character Splitter, Semantic Chunking, Chunk Size/Overlap |
| 1.3 | Embedding oluşturma | BAAI/bge-m3, E5, Instructor, Sentence Transformers |
| 1.4 | Vector DB'ye kayıt | ChromaDB, koleksiyon oluşturma, metadata |
| 1.5 | Retriever | Similarity Search, Top-k, MMR Search |
| 1.6 | LLM ile cevap üretme | Prompt Template, Context Injection, Hallucination, Grounded Generation |

**Faz çıktısı:** PDF yükleyip soru sorulabilen basit bir RAG prototipi.
**Not:** Evaluation bu fazda yapılmaz — önce çalışan sistem, ölçüm en sonda (bkz. Faz 6).

---

## Faz 2 — Agentic RAG 🟢

| Task | Konu | Araştırılacaklar |
|---|---|---|
| 2.1 | Agent kavramı | Workflow, Planning, Tool Calling — LangGraph / OpenAI Agents / Anthropic Tool Use |
| 2.2 | Tool Calling & MCP 🟡 | LLM'nin araç çağırma mantığı — PDF Search, Web Search, Calculator, Python, Vision Model; **MCP (Model Context Protocol)**, Tool Registry, Function Calling, Structured Output — güncel agent sistemlerinin standart yaklaşımı |
| 2.3 | LangGraph | State, Node, Edge, Conditional Edge → agent workflow kurma |
| 2.4 | Memory (genişletilmiş) 🟡 | Session Memory, Persistent Memory, User Profile Memory, Retrieval Memory (basit "Conversation Memory" yerine bu ayrımı kullan) |
| 2.5 | Multi-step Reasoning | ReAct, Plan and Execute, Reflection, Self Correction |

**Faz çıktısı:** Soruya göre doğru aracı otomatik seçen agent yapısı.

---

## Faz 3 — Multimodal 🟢 *(Jetson görevinden uyarlanmış, donanımsız)*

### Task 3.1 — Vision Language Models 🟢
- [ ] Qwen2.5-VL (ör. 3B) / Qwen3-VL (ör. 2B) — küçük/quantized sürüm ya da API
- [ ] LLaVA
- **Not:** GPU yetersizse API üzerinden VLM çağırmak (Anthropic/OpenAI/Qwen API) tercih edilebilir — model performansı değerlendirme kriteri değil, odak agent/RAG mantığında.

### Task 3.2 — Image Understanding 🟢
- [ ] Resim yükleme
- [ ] Görsel açıklama üretme
- [ ] Görsel üzerinden soru-cevap

### Task 3.3 — PDF İçindeki Görseller 🟡
- [ ] Metin ve görselleri birlikte yorumlama

### Task 3.4 — Video + Frame + VLM (Gradio) 🟢 ⭐ ana demo
Basit hâli:
```
Video → Frame Extraction → VLM → Yanıt
```
Daha gerçekçi hâli (Temporal Reasoning) 🟡:
```
Video → Frame Extraction → Frame Selection → VLM (çoklu frame) → LLM (özet/akıl yürütme) → Yanıt
```
Neden gerekli: "Adam kapıyı açmadan önce ne yaptı?" gibi sorular tek frame ile cevaplanamaz;
zaman içinde ardışık birkaç frame (5-10) gerekir. Önce basit sürümü bitir, zaman kalırsa
temporal sürüme geç.

- [ ] Kullanıcı arayüze video yükleyebilmeli
- [ ] Video oynatılırken kullanıcı prompt girebilmeli
- [ ] Soru gönderildiğinde ilgili anın frame'i (veya birkaç frame'i) çıkarılmalı (OpenCV)
- [ ] Frame(ler) + soru VLM'e gönderilmeli
- [ ] (Opsiyonel) Birden fazla frame varsa LLM ile özetleyip tek yanıt üretilmeli
- [ ] Yanıt arayüzde gösterilmeli

### Task 3.5 — OCR 🟡 *(taranmış PDF/görsel içindeki metin için)*
- [ ] PaddleOCR
- [ ] EasyOCR
- **Bağlantı:** Faz 1.1'deki PDF yüklemede taranmış sayfa tespit edilirse bu adım devreye girer.

**Faz çıktısı:** Video yükleyip belirli bir andaki kare(ler)le ilgili soru sorulabilen Gradio demosu.

---

## Faz 4 — Web Search / Hybrid RAG 🟢

- [ ] Task 4.1 — Web Search Tool: Tavily, SerpAPI, DuckDuckGo Search
- [ ] Task 4.2 — Hybrid RAG: Agent'ın "doküman yeterli mi / internet gerekli mi" kararı vermesi
- [ ] Task 4.3 — Knowledge Graph ⚪ *(opsiyonel/gelecek çalışma — şimdilik şart değil)*
  - İleride: Hybrid RAG → Knowledge Graph → Vector Search birlikte kullanılabilir.

**Faz çıktısı:** Dokümanın yetersiz kaldığı durumlarda otomatik olarak web'e başvuran hybrid RAG sistemi.

---

## Faz 5 — Arayüz (Gradio) 🟢

- [ ] Task 5.1 — Sohbet ekranı, PDF yükleme, görsel yükleme, cevap ekranı
- [ ] Task 5.2 — Streaming (cevap yazılırken ekrana akması)
- [ ] Task 5.3 — Kaynak gösterme: sayfa numarası, dosya adı, kullanılan araç
- [ ] Task 5.4 — Log paneli 🟡: Agent hangi tool'u çağırdı, kaç saniye sürdü, hangi doküman kullanıldı — ayrı bir panelde gösterilirse sunumda etkileyici olur.

**Faz çıktısı:** Uçtan uca kullanılabilir bir Gradio arayüzü — sohbet, dosya yükleme, streaming, kaynak gösterme bir arada.

---

## Faz 6 — Değerlendirme (Evaluation) 🟢 *(sistem tamamen çalışır hâle geldikten sonra)*

- [ ] Task 6.1 — RAG Evaluation: Faithfulness, Context Precision, Context Recall, Answer Relevancy
- [ ] Task 6.2 — Benchmark araçları: RAGAS, DeepEval, LangSmith

**Faz çıktısı:** Faithfulness/Context Precision/Recall/Answer Relevancy metrikleriyle hazırlanmış kısa bir değerlendirme raporu.

---

## Faz 7 — Deployment 🟡

- [ ] Docker
- [ ] Docker Compose
- [ ] Ollama (yerel model çalıştırma seçeneği)
- [ ] FastAPI (backend servis katmanı)
- [ ] Gradio (frontend, containerize edilmiş hâli)

**Neden önemli:** Gerçek projelerde deployment aşaması küçümsenmemeli; TSE'ye teslim ederken
"çalışan, paketlenmiş bir sistem" göstermek çok daha güçlü bir izlenim bırakır.

**Faz çıktısı:** `docker compose up` ile tek komutla ayağa kalkan, bağımsız çalıştırılabilir proje.

---

## Jetson Görevinden Ne Alındı / Ne Elendi

| Jetson Dokümanı Adımı | Durum |
|---|---|
| Aşama 1: Jetson kiti kurulumu, donanım üzerinde çalıştırma | ❌ Elendi (donanım yok) |
| İnceleme & Anlatım (Jetson AI Lab arşiv konuları) | ✅ Alındı → Faz 0 / Task 0.4 |
| Aşama 2: Gradio + video + frame + VLM uygulaması | ✅ Alındı → Faz 3 / Task 3.4, model API/lokal küçük modele uyarlandı, temporal reasoning ile genişletildi |

---

## Öncelik Sıralaması (zaman kısıtlıysa)

1. 🟢 Use Case + Faz 0 + Faz 1 + Faz 2 + Faz 3 (basit video demo) + Faz 5.1-5.3 → **minimum sunulabilir ürün**
2. 🟡 Faz 2 MCP/Memory genişletmesi, Faz 3 temporal reasoning + OCR, Faz 5.4 log paneli, Faz 7 deployment → **zaman kalırsa**
3. ⚪ Faz 4.3 Knowledge Graph → **gelecek çalışma olarak raporda bahsedilebilir, uygulanması şart değil**

---

## Sonuç — Proje Teslim Edildiğinde Sistem Şunları Yapabilecek:

- LLM tabanlı sohbet
- PDF tabanlı RAG (gerekirse OCR destekli)
- Vektör veritabanı entegrasyonu
- Agent mimarisi ve otomatik araç seçimi (MCP/Tool Calling)
- Web Search entegrasyonu (hybrid RAG)
- Görsel analiz (VLM) — statik görsel + PDF içi görseller
- Video üzerinde frame bazlı, zamana duyarlı VLM soru-cevap (Gradio)
- Çok adımlı akıl yürütme (ReAct / Plan-Execute / Reflection)
- Kaynak göstererek cevap üretme + agent log paneli
- RAG performans değerlendirmesi (RAGAS/DeepEval)
- Docker/FastAPI ile paketlenmiş, dağıtılabilir sistem
