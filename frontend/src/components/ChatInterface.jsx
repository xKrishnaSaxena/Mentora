import { useState, useRef, useEffect } from "react";
import React, { Children, isValidElement } from "react";
import {
  Send,
  Mic,
  ScreenShare,
  ChevronDown,
  Check,
  Copy,
  BookmarkPlus,
  Bot,
  User,
} from "lucide-react";
import QuizCard from "./QuizCard";
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

const RESTART_GRACE_MS = 500; // Reduced slightly for snappier response

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8765";

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

/* Mermaid renderer helpers */
function escapeHtml(s) {
  return s.replace(
    /[&<>"']/g,
    (m) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[
        m
      ])
  );
}

function sanitizeMermaid(code) {
  let c = code || "";
  c = c.replace(/^\s*```+mermaid\s*/i, "").replace(/```+\s*$/i, "");
  c = c
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/[–—]/g, "-")
    .replace(/\u2192/g, "-->")
    .replace(/\u2190/g, "<--")
    .replace(/-{1,2}>/g, "-->")
    .replace(/<-{1,2}/g, "<--")
    .replace(/\t/g, "  ");

  const lines = c.split("\n");
  const firstNonEmpty = (lines.find((l) => l.trim().length) || "").trim();
  const hasHeader =
    /^(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|journey)\b/.test(
      firstNonEmpty
    );
  if (!hasHeader) c = "flowchart TD\n" + c;
  return c.trim();
}

function looksLikeMermaid(code) {
  const first = (code.split("\n").find((l) => l.trim()) || "").trim();
  return /^(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|journey)\b/.test(
    first
  );
}

function Mermaid({ code }) {
  const containerRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose",
      theme: "dark",
      deterministicIds: true,
    });

    (async () => {
      try {
        const cleaned = sanitizeMermaid(code);
        if (!looksLikeMermaid(cleaned)) throw new Error("Not a Mermaid diagram");
        try {
          await mermaid.parse(cleaned);
        } catch (e) {
          const patched = /^flowchart\b/.test(cleaned)
            ? cleaned
            : "flowchart TD\n" + cleaned;
          await mermaid.parse(patched);
        }
        const id = "mmd-" + Math.random().toString(36).slice(2);
        const { svg } = await mermaid.render(id, cleaned);
        if (!cancelled && containerRef.current) {
          containerRef.current.innerHTML = svg;
        }
      } catch (e) {
        if (!cancelled && containerRef.current) {
          containerRef.current.innerHTML = `<pre class="rounded-lg overflow-x-auto bg-[#0c0f17] border border-white/10 p-3"><code>${escapeHtml(
            code || ""
          )}</code></pre>`;
        }
      }
    })();
    return () => {
      cancelled = true;
      if (containerRef.current) containerRef.current.innerHTML = "";
    };
  }, [code]);

  return <div ref={containerRef} className="max-w-full overflow-x-auto" />;
}

function CodeBlock({ raw, lang }) {
  const html = React.useMemo(() => highlight(raw, lang), [raw, lang]);
  return (
    <pre className="rounded-lg overflow-x-auto bg-[#0c0f17] border border-white/10 p-3">
      <code dangerouslySetInnerHTML={{ __html: html }} />
    </pre>
  );
}

function DetailsMarkdown({ content }) {
  return (
    <div className="prose prose-invert max-w-none prose-pre:my-0">
      <ReactMarkdown
        components={{
          pre({ children }) {
            const arr = Children.toArray(children);
            const only = arr.length === 1 ? arr[0] : null;
            const isEl = isValidElement(only);
            const className = isEl ? only.props?.className || "" : "";
            const raw = String(
              isEl ? only.props?.children || "" : children || ""
            );
            const match = /language-(\w+)/.exec(className);
            const lang = match?.[1] || "plaintext";

            if (lang === "mermaid") {
              const cleaned = sanitizeMermaid(raw);
              if (!looksLikeMermaid(cleaned))
                return <CodeBlock raw={raw} lang="plaintext" />;
              return <Mermaid code={cleaned} />;
            }
            return <CodeBlock raw={raw} lang={lang} />;
          },
          code({ inline, className, children, ...props }) {
            return (
              <code className="bg-white/10 px-1.5 py-0.5 rounded" {...props}>
                {children}
              </code>
            );
          },
          p({ children, ...props }) {
            const arr = Children.toArray(children);
            if (arr.length === 1) {
              const child = arr[0];
              if (isValidElement(child) && child.type === "pre") {
                return child;
              }
            }
            return <p {...props}>{children}</p>;
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
  chatId,
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
  const [isGenerating, setIsGenerating] = useState(false);

  const [connectionError, setConnectionError] = useState(null);

  const [isVoiceModeActive, setIsVoiceModeActive] = useState(false);
  const [isRecognitionActive, setIsRecognitionActive] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState("");

  const [currentFrame, setCurrentFrame] = useState(null);

  // Refs
  const messagesEndRef = useRef(null);
  const recognitionRef = useRef(null);
  const audioElRef = useRef(null);
  const screenShareTrackRef = useRef(null);
  const asrLockedRef = useRef(false);
  
  // NEW: Track voice mode in a ref to avoid stale closures during async audio playback
  const voiceModeRef = useRef(isVoiceModeActive);

  /* scroll */
  const scrollToBottom = () =>
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });

  /* Sync Ref with State */
  useEffect(() => {
    voiceModeRef.current = isVoiceModeActive;
  }, [isVoiceModeActive]);

  /* Load history */
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (!chatId) return;
    const cached = localStorage.getItem(`chat:${chatId}`);
    if (cached) {
      try {
        setMessages(JSON.parse(cached));
      } catch {}
    } else {
      setMessages([]);
    }

    const token = localStorage.getItem("authToken");
    fetch(`${API_BASE}/chats/${chatId}/messages`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((r) => {
        if (r.status === 401) {
          onNavigate?.("login");
          return [];
        }
        return r.ok ? r.json() : [];
      })
      .then((data) =>
        setMessages(
          data.map((m) => ({
            type: m.role === "assistant" ? "assistant" : "user",
            text: m.text || "",
            detailed: m.detailed || "",
            mode: m.mode || "teach",
            quiz: m.quiz || null,
          }))
        )
      )
      .catch(() => {});
  }, [chatId, onNavigate]);

  useEffect(() => {
    if (!chatId) return;
    localStorage.setItem(`chat:${chatId}`, JSON.stringify(messages));
  }, [messages, chatId]);

  /* ASR Logic */
  const startRecognition = (force = false) => {
    // If not forced, check locks. If forced (audio just ended), ignore locks.
    if (!force && (asrLockedRef.current || isPlayingAudio)) return;

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
        // Double check lock in case audio started simultaneously
        if (asrLockedRef.current) return; 
        const transcript = e.results[e.results.length - 1][0].transcript;
        setInputText("");
        handleSendMessage(transcript, true);
      };
      rec.onend = () => {
        setIsRecognitionActive(false);
        // Use the ref here to ensure we restart based on current intent
        if (voiceModeRef.current && !asrLockedRef.current) {
            startRecognition();
        } else {
            setVoiceStatus("");
        }
      };
      rec.onerror = (e) => {
        setVoiceStatus(`ASR error: ${e.error}`);
        if (e.error === "no-speech" && voiceModeRef.current)
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

  /* Screen Share */
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

  /* Handle Send */
  const handleSendMessage = async (text = inputText, isVoice = false) => {
    const question = (text || "").trim();
    if (!question) return;

    const token = localStorage.getItem("authToken");
    if (!token) {
      setConnectionError("Please log in.");
      onNavigate?.("login");
      return;
    }

    setMessages((prev) => [
      ...prev,
      { type: "user", text: question, mode: isVoice ? "voice" : "text" },
    ]);
    setInputText("");
    setIsGenerating(true);
    setConnectionError(null);

    const framePayload = selectedMode === "learn" ? currentFrame : null;
    const payload = {
      type: isVoice ? "voice_query" : "text_query",
      chatId,
      question,
      frame: framePayload,
      mode: selectedMode,
      category,
    };

    try {
      const res = await fetch(`${API_BASE}/chat/interaction`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });

      if (res.status === 401) {
        setConnectionError("Session expired. Please log in.");
        localStorage.removeItem("authToken");
        onNavigate?.("login");
        return;
      }

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Server error");
      }

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          text: data.text ?? "",
          detailed: data.detailed ?? "",
          responseType: data.responseType ?? "answer",
          mode: data.mode ?? selectedMode,
          quiz: data.quiz ?? null,
        },
      ]);

      if (data.audio) {
        // Lock mic
        asrLockedRef.current = true;
        stopRecognition(true);

        if (audioElRef.current) {
          audioElRef.current.pause();
          audioElRef.current.src = "";
        }

        const a = audioElRef.current;
        a.src = URL.createObjectURL(base64ToBlob(data.audio, "audio/mp3"));

        const unlock = () => {
          setIsPlayingAudio(false);
          // Unlock after grace period
          setTimeout(() => {
            asrLockedRef.current = false;
            // CHECK THE REF (live value) NOT THE VARIABLE (captured value)
            if (voiceModeRef.current) {
                startRecognition(true); // Force start
            } else {
                setVoiceStatus(""); // Clear "Playing response..." if not listening
            }
          }, RESTART_GRACE_MS);
          
          a.removeEventListener("ended", unlock);
          a.removeEventListener("error", unlock);
        };

        a.addEventListener("ended", unlock);
        a.addEventListener("error", unlock);

        try {
          setIsPlayingAudio(true);
          setVoiceStatus("Playing response…"); // Set status immediately
          await a.play();
        } catch {
          setIsPlayingAudio(false);
          unlock();
        }
      }
    } catch (err) {
      setConnectionError(err.message || "Failed to send message");
    } finally {
      setIsGenerating(false);
      scrollToBottom();
    }
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

  function SpinnerDot() {
    return (
      <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-white/25 border-t-white/70" />
    );
  }

  return (
    <div className="flex h-full w-full min-h-0 flex-col rounded-2xl bg-[#0b0f19] text-slate-100 border border-white/5 shadow-[0_10px_30px_-12px_rgba(0,0,0,0.35)] overflow-hidden">
      <audio ref={audioElRef} hidden />

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/10 bg-[#0b0f19]/80 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="text-sm sm:text-base font-semibold tracking-tight flex items-center gap-2">
            <span className="opacity-80">{categoryIcon}</span>
            <span className="opacity-90">{category}</span>
          </div>

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
                      if (k !== "learn") setCurrentFrame(null);
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
      <div className="flex-1 min-h-0 overflow-y-auto px-3 sm:px-4 py-3 space-y-3">
        {!messages.length && (
          <div className="rounded-xl border border-white/10 bg-white/5 p-4 text-center text-sm text-slate-300">
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

              {m.responseType === "quiz" && m.quiz && Array.isArray(m.quiz) && (
                <div className="mt-2 rounded-lg border border-white/10 p-3 bg-black/20">
                  <QuizCard quiz={m.quiz} />
                </div>
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
        {isGenerating && (
          <div className="w-full flex justify-start">
            <div className="max-w-[82%] sm:max-w-[68%] rounded-2xl border bg-white/[0.04] border-white/10 text-slate-100 p-3 shadow">
              <div className="flex items-center gap-2 mb-1 opacity-70 text-xs">
                <Bot size={14} />
                <span>Assistant</span>
              </div>
              <div className="text-sm flex items-center gap-2 opacity-80">
                <SpinnerDot /> <span>Thinking…</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Composer */}
      <div className="border-t border-white/10 bg-[#0b0f19]/80 backdrop-blur px-3 sm:px-4 py-3">
        <div className="w-full">
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
                  className="flex-1 h-11 overflow-y-auto resize-none bg-transparent text-sm text-slate-100 placeholder:text-slate-400 outline-none px-2 py-1.5"
                />
                <div className="flex items-center gap-1">
                  {/* mic toggle */}
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

                  {/* screen share */}
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