<div align="center">
  <img src="assets/images/icon.png" alt="Sahai Logo" width="120" height="120" style="border-radius: 20px;" />
  <h1>Sahai</h1>
  <p><strong>Next-Generation Multimodal AI Companion & Meeting Assistant</strong></p>
</div>

---

**Sahai** is an advanced, offline-first mobile application built with React Native and Expo that acts as your ultimate AI companion. With a beautifully designed interface, Sahai effortlessly handles multi-modal inputs, local document context (RAG), and real-time meeting assistance.

## ✨ Crazy Features

### 🎙️ Live Meeting Engine
Never miss a beat in your meetings again.
- **Intelligent Question Detection:** Actively listens during live meetings and automatically detects when a question is asked.
- **Separate Answers:** Isolates questions and provides separate, concise answers in real-time, functioning as an invisible co-pilot.

### 🧠 Advanced RAG & Document Context
Turn your files into an instant knowledge base.
- **Multiple Files Support:** Upload up to 50 documents, including PDFs, text files, and even **code files**.
- **Offline-First Retrieval:** Uses a highly efficient on-device chunking and BM25 relevance-scoring system.
- **Context Injection:** Seamlessly utilizes your uploaded files as context for deeply accurate and grounded chat responses.

### 🎥 Intelligent Video Processing
Analyze video inputs on the fly.
- **Frame Segregation:** Processes video by extracting two frames per second for AI vision analysis.
- **Toggleable Workflow:** Video processing features can be easily toggled on or off to conserve resources.

### 🌌 Multimodal Input Processing
Why type when you can show and tell?
- **All-in-One Chat:** Seamlessly handles **audio, video, images, and text** all within a single chat interface.
- **Vision Integration:** Uses Llama Vision to extract text and visual context from images, automatically answering prioritized questions from the visual data.

### ⚡ Content Summarization & Caching
Built for speed and efficiency.
- **Summarization:** Instantly summarizes long conversations or extracted contexts.
- **Smart Caching:** Employs targeted memory caching for documents and images to prevent stale data leaks and optimize performance without unnecessary re-processing.

## 🛠️ Technology Stack
- **Framework:** React Native / Expo
- **AI Models:** Groq (Llama 3.3, Llama 4 Vision) & Google Gemini (2.0 Flash)
- **Local Storage:** On-device persistent JSON chunk indexing & file storage
- **Retrieval:** Custom Local BM25 Scoring Algorithm

---

<div align="center">
  <sub>Built for ultimate productivity.</sub>
</div>
