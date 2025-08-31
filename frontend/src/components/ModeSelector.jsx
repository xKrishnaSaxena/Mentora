import React from "react";
import classNames from "classnames";

const modes = [
  { key: "rag", label: "Ask (RAG)", desc: "Answer from textbook context" },
  {
    key: "learn",
    label: "Learn from Image",
    desc: "Analyze an image + question",
  },
  { key: "quiz", label: "Quiz", desc: "Generate a 5-question MCQ" },
];

export default function ModeSelector({ mode, setMode }) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
      {modes.map((m) => (
        <button
          key={m.key}
          onClick={() => setMode(m.key)}
          className={classNames(
            "border rounded-lg p-3 text-left hover:shadow transition",
            mode === m.key
              ? "border-indigo-500 ring-2 ring-indigo-200"
              : "border-gray-300"
          )}
        >
          <div className="font-semibold">{m.label}</div>
          <div className="text-sm text-gray-500">{m.desc}</div>
        </button>
      ))}
    </div>
  );
}
