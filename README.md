# Mentora — Your voice-powered, source-backed study buddy.

Mentora is a modern AI tutoring chat that blends four modes—Teach, Learn (with screen share), Quiz, and RAG (PDF-grounded answers)—into one sleek experience. It features JWT auth, a realtime WebSocket pipeline, optional text-to-speech responses, on-device speech recognition, and safe Mermaid + syntax-highlighted Markdown rendering.

Built with:

- **Backend**: Python (aiohttp), Google Generative AI (Gemini), MongoDB (Motor), Milvus/Zilliz (with FAISS fallback), edge-tts.
- **Frontend**: React + Vite, Tailwind-esque styling, lucide-react icons, react-markdown, mermaid, Prism.js.

---

## 📽️ Walkthrough

### Teach Mode + Conversation History

[teachmode](https://github.com/user-attachments/assets/5f9fb69e-716d-40d9-95c7-da19028da8ca)

---

### Learn Mode

[learnmode](https://github.com/user-attachments/assets/c4187725-2633-4bb7-9d5b-76c54ff11f05)

---

### Quiz Mode

[quizmode](https://github.com/user-attachments/assets/422636d9-0e93-45ac-bedb-7d748b77c326)

---

### RAG Mode

[ragmode](https://github.com/user-attachments/assets/9534c0cc-69b1-4e11-aec1-e7ef98b2a94b)

---

## 🚀 Features

### 🔐 Auth & Sessions

- Register/login with **JWT**; 6-hour session lifetime.
- All protected routes guarded by middleware.
- WebSocket requires `?token=...`.

### 🧠 Four Conversation Modes

- **Teach**: step-by-step explanations with optional Mermaid diagrams.
- **Learn (Vision)**: share your screen; each captured frame is sent as a JPEG to answer “what’s on screen?” type questions.
- **Quiz**: one-shot 5-question MCQ generator (returns structured JSON + a pretty card UI).
- **RAG**: PDF-grounded answers powered by **LangChain** + **GoogleEmbeddings**; stores vectors in **Milvus/Zilliz** with **FAISS** fallback.

### 🗣️ Voice Mode (TTS + ASR)

- **ASR**: Browser **Web Speech API** (no server key required).
- **TTS**: **edge-tts** (Microsoft neural voices). Assistant can speak replies; autoplay with smart ASR gating to avoid echo.

### 🧾 Nice Markdown, Safer Diagrams

- **react-markdown** pipeline with a hardened Mermaid renderer.
- Automatic sanitization and graceful fallback to code blocks on parse errors.
- **Prism.js** highlighting for JS/TS/TSX/JSX/Python/Java/C#/CSS/JSON.

### 💬 Chat UX

- Copy to clipboard, “Add to Notes” hook, local history cache in `localStorage`.
- Reconnect logic and user-friendly status toasts.
- Screen-share indicator and voice mode status pill.

---

## 🧰 Tech Stack

**Backend**

- Python 3.10+
- aiohttp, aiohttp_cors
- google-generativeai (Gemini 1.5 Flash)
- LangChain (community loaders, GoogleEmbeddings)
- Milvus/Zilliz (pymilvus) + FAISS (in-memory fallback)
- MongoDB (motor)
- edge-tts, PyJWT, bcrypt
- OpenCV, Pillow, NumPy
- dotenv, logging

**Frontend**

- React + Vite
- lucide-react
- react-markdown
- mermaid
- prismjs
- Tailwind-style utility classes (works with any utility CSS approach)

---

## ⚙️ Environment Variables

Create `backend/.env`:

```
# Google Generative AI
GEMINI_API_KEY=YOUR_GEMINI_KEY

# Zilliz / Milvus (Vectors for RAG). If unavailable, code falls back to FAISS.
ZILLIZ_CLOUD_URI=YOUR_ZILLIZ_URI
ZILLIZ_CLOUD_API_KEY=YOUR_ZILLIZ_API_KEY

# Mongo & Auth
MONGODB_URI=mongodb+srv://user:pass@cluster/dbname
JWT_SECRET=supersecretjwtstring

# Optional
PORT=8765
TTS_PROVIDER=edge
```

Create `frontend/.env` (Vite):

```
VITE_API_BASE_URL=http://localhost:8765
VITE_WS_BASE=ws://localhost:8765
```

---

## 🛠️ Setup & Run

### Backend

Install & run:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place your RAG PDF next to server.py: lecs105.pdf

python server.py
# Server logs:
# http://0.0.0.0:8765
# ws://0.0.0.0:8765/ws
```

### Frontend

Make sure your app scaffolding (Vite) is set. Install deps:

```bash
cd frontend
npm i
npm run dev
# → http://localhost:5173
```

---

## 🧪 RAG (PDF) Notes

- The backend expects a PDF named **`lecs105.pdf`** beside the server file.
  Change the filename/path inside `initialize_pdf_rag()` if needed.
- Embeddings: `models/text-embedding-004` (Google).
- Vector store priority:

  1. Milvus/Zilliz (if `ZILLIZ_CLOUD_URI` + `ZILLIZ_CLOUD_API_KEY` are set)
  2. FAISS (in-memory fallback)

---

## 🖥️ Learn Mode (Screen Share)

- Uses `navigator.mediaDevices.getDisplayMedia`.
- If `MediaStreamTrackProcessor` is available, frames are sampled \~5 FPS and JPEG-encoded to Base64.
- Backend converts Base64 → OpenCV → Pillow → Gemini Vision request.

---

## 🔊 Voice Mode Behaviors

- **ASR**: Web Speech API (Chrome/Edge). We restart gracefully after TTS playback to avoid echo.
- **TTS**: `edge-tts` with defaults:

  - Voice: `en-US-GuyNeural`
  - Rate: `-8%`
  - Pitch: `-6Hz`

You can tune these via the `text_to_speech` call (and `.env` `TTS_PROVIDER=edge`).

---

## 🔐 Security & Safety

- JWT on all protected routes, validated by middleware.
- WebSocket rejects missing/invalid tokens.
- Mermaid diagrams sanitized; invalid diagrams render as plain code.
- The AI can be wrong—UI reminds users to verify critical info.

---

---

## 🧯 Troubleshooting

- **401 Invalid token**: Re-login; frontend will redirect to /login.
- **RAG not initialized**: Ensure `lecs105.pdf` exists and your Zilliz credentials are valid; otherwise FAISS should activate.
- **No TTS playback**: Browser autoplay may block; user interaction typically fixes it.
- **ASR not working**: Ensure microphone permission and a browser that supports the Web Speech API.

---
