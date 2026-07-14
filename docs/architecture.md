# Task 0.5 — Sistem Mimarisi

## Kullanıcı İsteği Akışı
(buraya adım adım akış yazılacak)

## Agent Workflow
(karar noktaları, tool seçimi mantığı)

## Kullanılan Bileşenler
- LLM:
- VLM:
- Vector DB:
- Web Search:
- Memory:

## Mermaid Diyagramı
```mermaid
flowchart TD
    A[Kullanıcı Sorusu] --> B{Agent Karar Verir}
    B -->|Doküman| C[RAG]
    B -->|Görsel/Video| D[VLM]
    B -->|Güncel bilgi| E[Web Search]
    C --> F[Yanıt]
    D --> F
    E --> F
```
