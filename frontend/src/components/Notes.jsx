import React from "react";
import { NotebookPen } from "lucide-react";

export default function Notes({ notes = [], onBack }) {
  return (
    <div className="max-w-5xl mx-auto mt-8">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          <NotebookPen className="text-indigo-600" /> My Notes
        </h2>
        <button
          onClick={onBack}
          className="px-3 py-1.5 rounded-lg border bg-white hover:bg-slate-50 text-sm"
        >
          Back
        </button>
      </div>
      {notes.length === 0 ? (
        <div className="glass rounded-xl p-6 text-slate-600">
          No notes yet. Select text in the canvas and click “Add note”.
        </div>
      ) : (
        <div className="grid md:grid-cols-2 gap-4">
          {notes.map((n) => (
            <div
              key={n.id}
              className={`rounded-xl p-4 border ${
                n.color || "bg-blue-50 text-blue-600"
              } bg-white`}
            >
              <div className="text-xs text-slate-500">
                {new Date(n.timestamp).toLocaleString()}
              </div>
              <div className="text-sm mt-1 font-medium">{n.subject}</div>
              <p className="mt-2 text-slate-700">{n.content}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
