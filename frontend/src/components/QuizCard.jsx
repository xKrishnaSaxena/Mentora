import React, { useMemo, useState } from "react";

/*
  Props:
  - quiz: [{ question, options: [..4], answer_index, explanation }, ...]
*/
export default function QuizCard({ quiz = [] }) {
  const [current, setCurrent] = useState(0);
  const [selections, setSelections] = useState(Array(quiz.length).fill(null));
  const [revealed, setRevealed] = useState(Array(quiz.length).fill(false));
  const [done, setDone] = useState(false);

  const score = useMemo(
    () =>
      selections.reduce(
        (s, sel, i) => s + (sel === quiz[i]?.answer_index ? 1 : 0),
        0
      ),
    [selections, quiz]
  );

  const q = quiz[current];

  const pick = (idx) => {
    if (revealed[current]) return; // lock after reveal
    setSelections((prev) => {
      const next = [...prev];
      next[current] = idx;
      return next;
    });
  };

  const reveal = () => {
    if (revealed[current]) return;
    setRevealed((prev) => {
      const next = [...prev];
      next[current] = true;
      return next;
    });
  };

  const next = () => {
    if (current < quiz.length - 1) setCurrent((c) => c + 1);
    else setDone(true);
  };

  if (!quiz.length) {
    return <div className="text-sm text-slate-300">No quiz data found.</div>;
  }

  return (
    <div className="space-y-3">
      {!done ? (
        <>
          <div className="text-xs opacity-70">
            Question {current + 1} of {quiz.length}
          </div>

          <div className="text-sm font-medium">{q.question}</div>

          <div className="mt-2 space-y-2">
            {q.options.map((opt, idx) => {
              const selected = selections[current] === idx;
              const isCorrect = q.answer_index === idx;
              const show = revealed[current];

              const ring =
                show && isCorrect
                  ? "ring-2 ring-green-500/40 border-green-500/30"
                  : show && selected && !isCorrect
                  ? "ring-2 ring-red-500/40 border-red-500/30"
                  : "border-white/10";

              return (
                <label
                  key={idx}
                  className={`flex items-center gap-2 rounded-lg border ${ring} bg-white/5 px-3 py-2 cursor-pointer`}
                >
                  <input
                    type="radio"
                    name={`q-${current}`}
                    disabled={show}
                    checked={selected}
                    onChange={() => pick(idx)}
                    className="accent-indigo-500"
                  />
                  <span className="text-sm">{opt}</span>
                </label>
              );
            })}
          </div>

          <div className="flex items-center gap-2 pt-1">
            {!revealed[current] ? (
              <button
                onClick={reveal}
                disabled={selections[current] === null}
                className={`rounded-md px-3 py-1.5 text-sm font-medium ${
                  selections[current] === null
                    ? "bg-white/5 text-slate-400 border border-white/10 cursor-not-allowed"
                    : "bg-indigo-600 hover:bg-indigo-700 text-white"
                }`}
              >
                Check answer
              </button>
            ) : (
              <button
                onClick={next}
                className="rounded-md px-3 py-1.5 text-sm font-medium bg-white/5 hover:bg-white/10 border border-white/10"
              >
                {current < quiz.length - 1 ? "Next" : "Finish"}
              </button>
            )}

            {revealed[current] && (
              <span className="text-xs px-2 py-1 rounded bg-white/5 border border-white/10">
                Correct answer: {String.fromCharCode(65 + q.answer_index)}
              </span>
            )}
          </div>

          {revealed[current] && (
            <div className="text-xs text-slate-300/90 bg-black/20 border border-white/10 rounded-lg p-2">
              {q.explanation}
            </div>
          )}
        </>
      ) : (
        <>
          <div className="text-sm font-semibold">
            Score: {score} / {quiz.length}
          </div>
          <div className="text-xs text-slate-400">Review:</div>
          <div className="space-y-2">
            {quiz.map((qq, i) => {
              const selected = selections[i];
              const good = selected === qq.answer_index;
              return (
                <div
                  key={i}
                  className="rounded-lg border border-white/10 bg-white/5 p-2"
                >
                  <div className="text-xs opacity-80">
                    Q{i + 1}. {qq.question}
                  </div>
                  <div className="mt-1 text-xs">
                    Your answer:{" "}
                    {selected !== null
                      ? String.fromCharCode(65 + selected)
                      : "—"}
                    {"  "}•{" "}
                    <span className={good ? "text-green-300" : "text-red-300"}>
                      {good ? "Correct" : "Incorrect"}
                    </span>
                  </div>
                  {!good && (
                    <div className="text-xs">
                      Correct: {String.fromCharCode(65 + qq.answer_index)}
                    </div>
                  )}
                  <div className="mt-1 text-xs text-slate-300/90">
                    {qq.explanation}
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
