# Agentic RAG + Multimodal

TSE stajı kapsamında geliştirilen, LLM/RAG/Agent mimarisini kullanan çok katmanlı yapay zeka asistanı projesi.

## Kullanılan Modeller

| Model | Görev |
|---|---|
| `gemma2:9b` (Ollama) | LLM — cevap üretme ve reranking |
| `qwen2.5vl:7b` (Ollama) | Vision LM — görsel ve video analizi |
| `ytu-ce-cosmos/turkish-e5-large` | Embedding — Türkçe semantik arama |

## Proje Mimarisi

Proje 7 bağımsız faza ayrılmıştır. Her faz bir öncekinin üzerine inşa edilir.

```
faz1 (Klasik RAG) → faz2 (Agentic RAG) → faz3 (Multimodal)
                                        ↘ faz4 (Hybrid + Web)
                                                ↓
                              faz5 (Gradio Arayüz)
                                                ↓
                              faz6 (Değerlendirme)
                                                ↓
                              faz7 (Deployment)
```

## Klasör Yapısı

```
agentic-rag-project/
│
├── docs/
│   ├── roadmap.md              # Proje yol haritası
│   ├── architecture.md         # Mimari tasarım notları
│   └── research-notes.md       # Araştırma notları
│
├── data/
│   ├── documents/              # PDF, Word gibi doküman test dosyaları
│   ├── images/                 # Görsel test dosyaları
│   ├── videos/                 # Video test dosyaları
│   ├── processed/              # Chunk ve embedding JSON çıktıları
│   └── ocr_cikti/              # PaddleOCR metin çıktıları (.md)
│
├── faz1_klasik_rag/            # PDF yükleme, chunking, embedding, retriever, LLM cevap
├── faz2_agentic_rag/           # LangGraph agent, LLM reranking, self-reflection
├── faz3_multimodal/            # Görsel QA, PDF görsel analizi, video analizi, OCR
├── faz4_hybrid_rag/            # Web search entegrasyonu (Tavily/DuckDuckGo)
├── faz5_arayuz/                # Gradio arayüzü
├── faz6_evaluation/            # Otomatik değerlendirme sistemi
├── faz7_deployment/            # Docker/FastAPI deployment
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                        # API anahtarları (git'e dahil edilmez)
```

## Faz Detayları

### Faz 1 — Klasik RAG
`01_pdf_loader.py` → `02_chunker.py` → `03_embedder.py` → `04_chromadb_writer.py` → `05_retriever.py` → `06_llm_answer.py`

PDF dokümanı madde bazlı parçalara ayırır, `ytu-ce-cosmos/turkish-e5-large` ile embed eder ve ChromaDB'ye yazar. Kullanıcı sorusunu en alakalı maddeye yönlendirip `gemma2:9b` ile cevap üretir.

### Faz 2 — Agentic RAG
`07_agent.py`

LangGraph ile yönetilen otonom ajan mimarisi. LLM tabanlı reranking, self-reflection ve fallback mekanizmaları içerir.

### Faz 3 — Multimodal
`08_vision.py`, `09_pdf_gorsel.py`, `10_video.py`, `11_ocr.py`

`qwen2.5vl:7b` ile görsel ve video analizi. PaddleOCR ile tablo/metin içeren PDF sayfalarından metin çıkarma.

### Faz 4 — Hybrid RAG
`12_web_search.py`, `13_hybrid_agent.py`

Yerel vektör tabanında cevap bulunamazsa sistemi otomatik olarak Tavily/DuckDuckGo web aramasına yönlendiren otonom karar mekanizması.

### Faz 5 — Arayüz
`14_gradio_app.py`

Tüm fazları tek bir Gradio arayüzünde birleştiren kullanıcı dostu web arayüzü.

### Faz 6 — Değerlendirme
`15_degerlendirme.py`

Otomatik test sistemi. Her modül için beklenen çıktılarla karşılaştırmalı doğruluk ölçümü.

### Faz 7 — Deployment
`Dockerfile` + `docker-compose.yml`

Tüm sistem Docker ile containerize edilmiştir. Tek komutla çalıştırılabilir.

## Kurulum

**Gereksinimler:**
- Python 3.10+
- [Ollama](https://ollama.com) (yerel LLM sunucusu)

```bash
# 1. Modelleri indir
ollama pull gemma2:9b
ollama pull qwen2.5vl:7b

# 2. Repoyu klonla
git clone https://github.com/Beyza006/agentic-rag-multimodal.git
cd agentic-rag-multimodal

# 3. Sanal ortam oluştur
python -m venv venv
venv\Scripts\activate  # Windows

# 4. Bağımlılıkları yükle
pip install -r requirements.txt

# 5. Ortam değişkenlerini ayarla
cp .env.example .env  # .env dosyasını doldurun
```

## Docker ile Çalıştırma

```bash
docker-compose up --build
```

## Not: Veri Dosyaları

`.gitignore` içinde `*.pdf`, `*.mp4`, `*.json` gibi veri dosyaları hariç tutulmuştur. `data/` klasör yapısı GitHub'da görünür ancak içindeki gerçek dosyalar görünmez — bu dosyaları her makinede ayrıca eklemeniz gerekir.

Detaylı yol haritası için bkz. [docs/roadmap.md](docs/roadmap.md)
