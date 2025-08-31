import { useState, useRef, useEffect, useCallback } from "react";
import {
  Send,
  Mic,
  ScreenShare,
  ChevronDown,
  Check,
  Copy,
  BookmarkPlus,
  Sparkles,
  Bot,
  User,
} from "lucide-react";

import ReactMarkdown from "react-markdown";
import mermaid from "mermaid";
import Prism from "prismjs";
import "prismjs/themes/prism-tomorrow.css";
import "prismjs/components/prism-javascript";
import "prismjs/components/prism-jsx";
import "prismjs/components/prism-typescript";
import "prismjs/components/prism-tsx";
import "prismjs/components/prism-python";
import "prismjs/components/prism-java";
import "prismjs/components/prism-csharp";
import "prismjs/components/prism-css";
import "prismjs/components/prism-json";

const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY = 3000;
const RESTART_GRACE_MS = 1000; // wait this long after audio ends
// true while TTS is speaking (and a bit after)

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8765";
const WS_BASE = import.meta.env.VITE_WS_BASE || API_BASE.replace(/^http/, "ws");

const MODE_LABELS = {
  teach: "Teach",
  learn: "Learn",
  quiz: "Quiz",
  rag: "RAG",
};

/* ---------- helpers ---------- */

function base64ToBlob(b64, type) {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Blob([buf], { type });
}

function highlight(code, lang) {
  const grammar = Prism.languages[lang] || Prism.languages.markup;
  return Prism.highlight(code, grammar, lang);
}

/* Mermaid renderer for ```mermaid blocks inside details */
function Mermaid({ code }) {
  const containerRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose",
      theme: "dark",
    });

    (async () => {
      try {
        const id = "mmd-" + Math.random().toString(36).slice(2);
        const { svg } = await mermaid.render(id, code);
        if (!cancelled && containerRef.current) {
          containerRef.current.innerHTML = svg;
        }
      } catch (e) {
        if (containerRef.current)
          containerRef.current.innerHTML = `<pre>Mermaid error: ${e.message}</pre>`;
      }
    })();

    return () => {
      cancelled = true;
      if (containerRef.current) containerRef.current.innerHTML = "";
    };
  }, [code]);

  return <div ref={containerRef} className="max-w-full overflow-x-auto" />;
}

/* Markdown renderer used in Details */
function DetailsMarkdown({ content }) {
  return (
    <div className="prose prose-invert max-w-none prose-pre:my-0">
      <ReactMarkdown
        components={{
          code({ inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            const raw = String(children).replace(/\n$/, "");
            if (!inline && match && match[1] === "mermaid") {
              return <Mermaid code={raw} />;
            }
            if (!inline) {
              const lang = match?.[1] || "plaintext";
              return (
                <pre className="rounded-lg overflow-x-auto bg-[#0c0f17] border border-white/10 p-3">
                  <code
                    dangerouslySetInnerHTML={{ __html: highlight(raw, lang) }}
                  />
                </pre>
              );
            }
            return (
              <code className="bg-white/10 px-1.5 py-0.5 rounded" {...props}>
                {children}
              </code>
            );
          },
        }}
      >
        {content || ""}
      </ReactMarkdown>
    </div>
  );
}

/* ---------- main component ---------- */

export default function ChatInterface({
  category,
  categoryIcon,
  categoryColor = "indigo",
  onBackClick,
  initialQuestion = "",
  addNote,
  onNavigate,
}) {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState(initialQuestion);
  const [selectedMode, setSelectedMode] = useState("teach");
  const [showModeMenu, setShowModeMenu] = useState(false);

  const [connectionError, setConnectionError] = useState(null);

  const [isVoiceModeActive, setIsVoiceModeActive] = useState(false);
  const [isRecognitionActive, setIsRecognitionActive] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState("");

  const [currentFrame, setCurrentFrame] = useState(null);

  const ws = useRef(null);
  const messagesEndRef = useRef(null);
  const recognitionRef = useRef(null);
  const audioElRef = useRef(null); // DOM <audio>, improves autoplay reliability
  const screenShareTrackRef = useRef(null);
  const asrLockedRef = useRef(false);
  /* scroll */
  const scrollToBottom = () =>
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });

  /* WS connect */
  const connectWebSocket = useCallback(
    (attempt = 0) => {
      if (
        ws.current &&
        [WebSocket.OPEN, WebSocket.CONNECTING].includes(ws.current.readyState)
      )
        return;

      const token = localStorage.getItem("authToken");
      if (!token) {
        setConnectionError("Please log in to continue.");
        setTimeout(() => onNavigate?.("login"), 800);
        return;
      }

      ws.current = new WebSocket(
        `${WS_BASE}/ws?token=${encodeURIComponent(token)}`
      );
      let reconnectTimer = null;

      ws.current.onopen = () => {
        if (reconnectTimer) clearTimeout(reconnectTimer);
        setConnectionError(null);
      };

      ws.current.onmessage = async (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data?.error && String(data.error).includes("Invalid token")) {
            setConnectionError("Session expired. Please log in again.");
            localStorage.removeItem("authToken");
            setTimeout(() => onNavigate?.("login"), 800);
            return;
          }

          // push assistant message
          setMessages((prev) => [
            ...prev,
            {
              type: "assistant",
              text: data.text ?? "",
              detailed: data.detailed ?? "",
              responseType: data.responseType ?? "answer",
              mode: data.mode ?? selectedMode,
            },
          ]);

          // play audio if provided (for voice mode)
          if (data.audio) {
            // fully gate ASR during playback
            asrLockedRef.current = true;
            stopRecognition(true); // abort immediately

            if (audioElRef.current) {
              audioElRef.current.pause();
              audioElRef.current.src = "";
            }

            const a = audioElRef.current;
            a.src = URL.createObjectURL(base64ToBlob(data.audio, "audio/mp3"));

            const unlock = () => {
              setIsPlayingAudio(false);
              // give the browser a moment to stop routing output audio to the mic
              setTimeout(() => {
                asrLockedRef.current = false;
                if (isVoiceModeActive) startRecognition();
              }, RESTART_GRACE_MS);
              a.removeEventListener("ended", unlock);
              a.removeEventListener("error", unlock);
            };

            a.addEventListener("ended", unlock);
            a.addEventListener("error", unlock);

            try {
              setIsPlayingAudio(true);
              await a.play();
              setVoiceStatus("Playing response…");
            } catch {
              setIsPlayingAudio(false);
              // could not autoplay – still keep ASR locked for a beat, then resume
              unlock();
            }
          }
        } catch {
          setConnectionError("Error processing server response.");
        } finally {
          scrollToBottom();
        }
      };

      ws.current.onerror = () => {
        setConnectionError("Connection problem. Reconnecting…");
        if (
          ![WebSocket.CLOSED, WebSocket.CLOSING].includes(
            ws.current?.readyState
          )
        ) {
          ws.current?.close();
        }
      };

      ws.current.onclose = (e) => {
        if (e.code === 1008 && String(e.reason).includes("Invalid token")) {
          setConnectionError("Session expired. Please log in again.");
          localStorage.removeItem("authToken");
          setTimeout(() => onNavigate?.("login"), 800);
          return;
        }
        if (attempt < MAX_RECONNECT_ATTEMPTS) {
          reconnectTimer = setTimeout(
            () => connectWebSocket(attempt + 1),
            RECONNECT_DELAY
          );
        } else {
          setConnectionError("Unable to connect to server.");
        }
      };
    },
    [onNavigate, isVoiceModeActive, selectedMode]
  );

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (ws.current?.readyState === WebSocket.OPEN)
        ws.current.close(1000, "unmount");
      stopRecognition();
      if (audioElRef.current) {
        audioElRef.current.pause();
        audioElRef.current.src = "";
      }
      if (screenShareTrackRef.current) screenShareTrackRef.current.stop();
    };
  }, [connectWebSocket]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  /* ASR */
  const startRecognition = () => {
    if (asrLockedRef.current || isPlayingAudio) return;
    if (
      !("SpeechRecognition" in window || "webkitSpeechRecognition" in window)
    ) {
      setVoiceStatus("Speech recognition not supported");
      return;
    }
    if (!recognitionRef.current) {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      const rec = new SR();
      recognitionRef.current = rec;
      rec.continuous = true;
      rec.interimResults = false;
      rec.lang = "en-US";

      rec.onstart = () => {
        setIsRecognitionActive(true);
        setVoiceStatus("Listening…");
      };
      rec.onresult = (e) => {
        if (asrLockedRef.current || isPlayingAudio) return; // <-- drop late/echoed results
        const transcript = e.results[e.results.length - 1][0].transcript;
        setInputText("");
        handleSendMessage(transcript, true);
      };
      rec.onend = () => {
        setIsRecognitionActive(false);
        if (isVoiceModeActive && !isPlayingAudio) startRecognition();
        else setVoiceStatus("");
      };
      rec.onerror = (e) => {
        setVoiceStatus(`ASR error: ${e.error}`);
        if (e.error === "no-speech" && isVoiceModeActive)
          setTimeout(startRecognition, 350);
        if (["not-allowed", "service-not-allowed"].includes(e.error)) {
          setIsVoiceModeActive(false);
        }
      };
    }

    if (!isRecognitionActive) {
      navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then(() => {
          try {
            recognitionRef.current.start();
          } catch {}
        })
        .catch(() => {
          setVoiceStatus("Microphone access denied");
          setIsVoiceModeActive(false);
        });
    }
  };

  const stopRecognition = (forceAbort = false) => {
    try {
      const rec = recognitionRef.current;
      if (!rec) return;
      if (forceAbort && typeof rec.abort === "function") rec.abort();
      else if (isRecognitionActive) rec.stop();
    } catch {}
    setIsRecognitionActive(false);
  };

  const handleMicToggle = () => {
    if (isVoiceModeActive) {
      setIsVoiceModeActive(false);
      stopRecognition();
      setVoiceStatus("");
    } else {
      setIsVoiceModeActive(true);
      startRecognition();
    }
  };

  /* Screen share only for Learn mode */
  async function startScreenShare() {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 5 },
      });
      const track = stream.getVideoTracks()[0];
      screenShareTrackRef.current = track;

      track.addEventListener("ended", () => {
        setCurrentFrame(null);
      });

      if ("MediaStreamTrackProcessor" in window) {
        const { MediaStreamTrackProcessor } = window;
        const processor = new MediaStreamTrackProcessor({ track });
        const reader = processor.readable.getReader();
        const loop = async () => {
          const { done, value } = await reader.read();
          if (done) return;
          const bitmap = await createImageBitmap(value);
          const canvas = document.createElement("canvas");
          canvas.width = bitmap.width;
          canvas.height = bitmap.height;
          canvas.getContext("2d").drawImage(bitmap, 0, 0);
          setCurrentFrame(canvas.toDataURL("image/jpeg").split(",")[1]);
          setTimeout(loop, 200);
        };
        loop();
      } else {
        const video = document.createElement("video");
        video.srcObject = stream;
        await video.play();
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        const interval = setInterval(() => {
          if (video.videoWidth && video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            setCurrentFrame(canvas.toDataURL("image/jpeg").split(",")[1]);
          }
          if (track.readyState === "ended") clearInterval(interval);
        }, 200);
      }
    } catch (e) {
      console.error("Screen share error:", e);
    }
  }

  /* send */
  const handleSendMessage = (text = inputText, isVoice = false) => {
    const question = (text || "").trim();
    if (!question) return;

    // show user bubble
    setMessages((prev) => [
      ...prev,
      { type: "user", text: question, mode: isVoice ? "voice" : "text" },
    ]);

    // build compact history
    const history = [...messages, { type: "user", text: question }].map(
      (m) => ({
        role: m.type === "assistant" ? "assistant" : "user",
        content: m.text || m.detailed || "",
      })
    );

    // only attach frame when in learn mode
    const framePayload = selectedMode === "learn" ? currentFrame : null;

    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(
        JSON.stringify({
          type: isVoice ? "voice_query" : "text_query",
          question, // ✅ FIX: send the actual text, not inputText
          frame: framePayload, // ✅ only for learn
          mode: selectedMode,
          category,
          history,
        })
      );
    } else {
      setConnectionError("Cannot send message. Server unavailable.");
      setTimeout(() => setConnectionError(null), 2000);
    }

    setInputText("");
  };

  const copyToClipboard = async (str) => {
    try {
      await navigator.clipboard.writeText(str);
      setVoiceStatus("Copied!");
      setTimeout(() => setVoiceStatus(""), 1000);
    } catch {
      setVoiceStatus("Copy failed");
      setTimeout(() => setVoiceStatus(""), 1000);
    }
  };

  const addMsgToNotes = (m) => {
    const content = (m.detailed || m.text || "").trim();
    if (!content) return;
    addNote?.({
      id: Date.now(),
      subject: category,
      content,
      timestamp: new Date().toISOString(),
      color: "bg-indigo-50 text-indigo-500",
    });
    onNavigate?.("notes");
  };

  return (
    <div className="flex h-[calc(100vh-140px)] w-full flex-col rounded-2xl bg-[#0b0f19] text-slate-100 border border-white/5 shadow-[0_10px_30px_-12px_rgba(0,0,0,0.35)] overflow-hidden">
      {/* hidden audio element for reliable playback */}
      <audio ref={audioElRef} hidden />

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/10 bg-[#0b0f19]/80 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="text-sm sm:text-base font-semibold tracking-tight flex items-center gap-2">
            <span className="opacity-80">{categoryIcon}</span>
            <span className="opacity-90">{category}</span>
          </div>

          {/* mode selector */}
          <div className="relative ml-2">
            <button
              className="inline-flex items-center gap-1 rounded-md border border-white/10 bg-white/5 px-2 py-1 text-xs hover:bg-white/10"
              onClick={() => setShowModeMenu((s) => !s)}
            >
              {MODE_LABELS[selectedMode]} <ChevronDown size={12} />
            </button>
            {showModeMenu && (
              <div className="absolute z-20 mt-1 w-36 rounded-md border border-white/10 bg-[#0f1528] p-1 shadow-lg">
                {Object.entries(MODE_LABELS).map(([k, v]) => (
                  <button
                    key={k}
                    className="flex w-full items-center justify-between rounded px-2 py-1 text-left text-xs hover:bg-white/10"
                    onClick={() => {
                      setSelectedMode(k);
                      setShowModeMenu(false);
                      if (k !== "learn") setCurrentFrame(null); // drop frame when leaving learn
                    }}
                  >
                    <span>{v}</span>
                    {selectedMode === k && (
                      <Check size={12} className="text-indigo-400" />
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isVoiceModeActive && (
            <span className="hidden sm:inline-flex text-xs px-2 py-0.5 rounded bg-red-500/15 text-red-300 border border-red-500/20">
              🎤 {voiceStatus || "Voice Mode"}
            </span>
          )}
          {selectedMode === "learn" && currentFrame && (
            <span className="hidden sm:inline-flex text-xs px-2 py-0.5 rounded bg-indigo-500/15 text-indigo-300 border border-indigo-500/20">
              🖥️ Sharing
            </span>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-3 sm:px-4 py-3 space-y-3">
        {!messages.length && (
          <div className="mx-auto max-w-2xl rounded-xl border border-white/10 bg-white/5 p-4 text-center text-sm text-slate-300">
            Ask about {category}. Pick a mode, type your question, or use the
            mic.
          </div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={`w-full flex ${
              m.type === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[82%] sm:max-w-[68%] rounded-2xl border ${
                m.type === "user"
                  ? "bg-indigo-600/90 border-indigo-500/60 text-white"
                  : "bg-white/[0.04] border-white/10 text-slate-100"
              } p-3 shadow`}
            >
              <div className="flex items-center gap-2 mb-2 opacity-70 text-xs">
                {m.type === "assistant" ? (
                  <Bot size={14} />
                ) : (
                  <User size={14} />
                )}
                <span className="capitalize">{m.type}</span>
                {m.mode && (
                  <span className="px-1.5 py-0.5 rounded bg-white/5 border border-white/10">
                    {m.mode}
                  </span>
                )}
              </div>

              {m.text && (
                <p className="text-sm leading-relaxed whitespace-pre-wrap">
                  {m.text}
                </p>
              )}

              {m.detailed && (
                <details className="mt-2 group">
                  <summary className="text-xs cursor-pointer select-none opacity-80 hover:opacity-100">
                    Details (Markdown / diagrams / code)
                  </summary>
                  <div className="mt-2 rounded-lg border border-white/10 p-3 bg-black/20">
                    <DetailsMarkdown content={m.detailed} />
                  </div>
                </details>
              )}

              <div className="mt-2 flex items-center gap-2">
                <button
                  className="inline-flex items-center gap-1 rounded-md border border-white/10 bg-white/5 px-2 py-1 text-xs hover:bg-white/10"
                  onClick={() =>
                    copyToClipboard((m.detailed || m.text || "").trim())
                  }
                  title="Copy"
                >
                  <Copy size={12} />
                  Copy
                </button>
                {m.type === "assistant" && (
                  <button
                    className="inline-flex items-center gap-1 rounded-md border border-white/10 bg-white/5 px-2 py-1 text-xs hover:bg-white/10"
                    onClick={() => addMsgToNotes(m)}
                    title="Add to Notes"
                  >
                    <BookmarkPlus size={12} />
                    Notes
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}

        <div ref={messagesEndRef} />
      </div>

      {/* Composer */}
      <div className="border-t border-white/10 bg-[#0b0f19]/80 backdrop-blur px-3 sm:px-4 py-3">
        <div className="mx-auto w-full sm:max-w-3xl">
          <div className="flex items-end gap-2">
            <div className="flex-1">
              <div className="flex items-center rounded-2xl border border-white/10 bg-white/5 px-2 py-1.5 shadow-inner">
                <textarea
                  placeholder={`Ask about ${category}…`}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  rows={1}
                  style={{ minHeight: 42, maxHeight: 120 }}
                  className="flex-1 resize-none bg-transparent text-sm text-slate-100 placeholder:text-slate-400 outline-none px-2 py-1.5"
                />
                <div className="flex items-center gap-1">
                  {/* mic toggle always available */}
                  <button
                    className={`inline-flex items-center justify-center rounded-xl border border-white/10 px-2.5 py-2 ${
                      isVoiceModeActive
                        ? "bg-red-500/20 border-red-500/30 text-red-200"
                        : "bg-white/5 text-slate-200 hover:bg-white/10"
                    }`}
                    title={isVoiceModeActive ? "Stop voice" : "Start voice"}
                    onClick={handleMicToggle}
                  >
                    <Mic size={16} />
                  </button>

                  {/* screen share ONLY in Learn mode */}
                  {selectedMode === "learn" && (
                    <button
                      className={`inline-flex items-center justify-center rounded-xl border border-white/10 bg-white/5 px-2.5 py-2 text-slate-200 hover:bg-white/10 ${
                        currentFrame ? "ring-1 ring-indigo-500/40" : ""
                      }`}
                      title="Share screen (Learn mode)"
                      onClick={startScreenShare}
                    >
                      <ScreenShare size={16} />
                    </button>
                  )}

                  <button
                    onClick={() => handleSendMessage()}
                    disabled={!inputText.trim()}
                    className={`ml-1 inline-flex items-center justify-center rounded-xl px-3 py-2 text-sm font-medium ${
                      inputText.trim()
                        ? "bg-indigo-600 hover:bg-indigo-700 text-white"
                        : "bg-white/5 text-slate-400 border border-white/10 cursor-not-allowed"
                    }`}
                    title="Send"
                  >
                    <Send size={16} />
                  </button>
                </div>
              </div>
              <div className="mt-2 text-[11px] text-slate-400/90 text-center">
                AI may be inaccurate. Verify important info.
              </div>
            </div>
          </div>

          {connectionError && (
            <div className="mt-2 rounded-lg border border-red-500/30 bg-red-500/10 p-2 text-xs text-red-300">
              {connectionError}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
