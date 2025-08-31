import React, { useState } from "react";
import { NotebookPen, ArrowLeft, Copy, Download } from "lucide-react";

function accentClasses(hint = "") {
  const key =
    ["indigo", "violet", "blue", "sky", "emerald", "rose", "amber"].find((k) =>
      hint.toLowerCase().includes(k)
    ) || "indigo";

  const map = {
    indigo: "bg-indigo-500/15 text-indigo-300 border-indigo-500/30",
    violet: "bg-violet-500/15 text-violet-300 border-violet-500/30",
    blue: "bg-blue-500/15 text-blue-300 border-blue-500/30",
    sky: "bg-sky-500/15 text-sky-300 border-sky-500/30",
    emerald: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
    rose: "bg-rose-500/15 text-rose-300 border-rose-500/30",
    amber: "bg-amber-500/20 text-amber-300 border-amber-500/30",
  };
  return map[key];
}

function formatNoteFile(n, ext = "md") {
  const ts = new Date(n.timestamp || Date.now()).toISOString();
  if (ext === "md") {
    return `---
title: ${n.subject || "Note"}
date: ${ts}
---

${n.content || ""}\n`;
  }
  // txt
  return `${n.subject || "Note"}\n${ts}\n\n${n.content || ""}\n`;
}

export default function Notes({ notes = [], onBack }) {
  const [toast, setToast] = useState("");

  const copyNote = async (n) => {
    try {
      await navigator.clipboard.writeText(n.content || "");
      setToast("Copied note!");
    } catch {
      setToast("Copy failed");
    } finally {
      setTimeout(() => setToast(""), 1000);
    }
  };

  const downloadNote = (n, ext = "md") => {
    const blob = new Blob([formatNoteFile(n, ext)], {
      type: ext === "md" ? "text/markdown" : "text/plain",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const safeTitle = (n.subject || "note").replace(/[^\w\-]+/g, "_");
    a.href = url;
    a.download = `${safeTitle}_${Date.now()}.${ext}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="mx-auto max-w-6xl p-4 sm:p-6">
      <div className="mb-4 sm:mb-6 flex items-center justify-between">
        <h2 className="text-lg sm:text-xl font-semibold tracking-tight flex items-center gap-2">
          <span className="rounded-lg p-2 bg-indigo-500/15 border border-indigo-500/30 text-indigo-300">
            <NotebookPen size={18} />
          </span>
          <span className="bg-gradient-to-r from-indigo-200 to-sky-300 bg-clip-text text-transparent">
            My Notes
          </span>
        </h2>
        <button
          onClick={onBack}
          className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-sm text-slate-200 hover:bg-white/10"
        >
          <ArrowLeft size={16} />
          Back
        </button>
      </div>

      {notes.length === 0 ? (
        <div className="glass rounded-2xl p-6 text-slate-300 border border-white/10">
          No notes yet. Select text in the canvas and click “Add note”.
        </div>
      ) : (
        <div className="grid gap-4 sm:gap-5 md:grid-cols-2">
          {notes.map((n) => (
            <div
              key={n.id}
              className="glass rounded-2xl p-4 border border-white/10 hover:shadow-xl transition-shadow"
            >
              <div className="flex items-center justify-between gap-3">
                <span
                  className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs ${accentClasses(
                    n.color || ""
                  )}`}
                >
                  {n.subject || "Note"}
                </span>
                <span className="text-[11px] text-slate-400">
                  {new Date(n.timestamp).toLocaleString()}
                </span>
              </div>

              <div className="mt-3 text-sm leading-relaxed text-slate-200 whitespace-pre-wrap">
                {n.content}
              </div>

              <div className="mt-3 flex items-center gap-2">
                <button
                  className="inline-flex items-center gap-1 rounded-md border border-white/10 bg-white/5 px-2 py-1 text-xs hover:bg-white/10"
                  onClick={() => copyNote(n)}
                  title="Copy"
                >
                  <Copy size={12} />
                  Copy
                </button>
                <button
                  className="inline-flex items-center gap-1 rounded-md border border-white/10 bg-white/5 px-2 py-1 text-xs hover:bg-white/10"
                  onClick={() => downloadNote(n, "md")}
                  title="Download (.md)"
                >
                  <Download size={12} />
                  Download
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {toast && (
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 rounded-full border border-white/10 bg-white/10 px-3 py-1 text-xs text-slate-100">
          {toast}
        </div>
      )}
    </div>
  );
}
