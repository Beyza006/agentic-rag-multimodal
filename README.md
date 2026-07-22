# Agentic RAG + Multimodal Video QA

TSE stajı kapsamında geliştirilen, LLM/RAG/Agent mimarisini kullanan yapay zeka asistanı projesi.

Detaylı yol haritası için bkz. [docs/roadmap.md](docs/roadmap.md)

## Klasör Yapısı
- `docs/` — araştırma notları, mimari tasarım, roadmap
- `data/` — tüm fazların ortak kullandığı test verileri
  - `data/documents/` — PDF, Word gibi doküman test dosyaları (örn. ornek.pdf)
  - `data/images/` — görsel test dosyaları
  - `data/videos/` — video test dosyaları
- `faz1_klasik_rag/` — PDF yükleme, chunking, embedding, retriever
- `faz2_agentic_rag/` — agent, tool calling, LangGraph, memory
- `faz3_multimodal/` — VLM, image/video QA
- `faz4_hybrid_rag/` — web search entegrasyonu
- `faz5_arayuz/` — Gradio arayüzü
- `faz6_evaluation/` — RAGAS/DeepEval değerlendirme
- `faz7_deployment/` — Docker/FastAPI deployment

## Kurulum
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Not: Veri dosyaları
`.gitignore` içinde `*.pdf`, `*.mp4` gibi veri dosyaları hariç tutulmuştur
(repoyu hafif tutmak için). `data/` klasör yapısı GitHub'da görünür ancak
içindeki gerçek PDF/görsel/video dosyaları görünmez — bu dosyaları her
makinede ayrıca eklemeniz gerekir.
