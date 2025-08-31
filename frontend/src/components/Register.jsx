import React, { useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8765";

export default function Register({ onSwitch }) {
  const [form, setForm] = useState({
    name: "",
    email: "",
    studentRegNumber: "",
    dob: "",
    password: "",
  });
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");
  const [error, setError] = useState("");

  const onChange = (e) =>
    setForm((f) => ({ ...f, [e.target.name]: e.target.value }));

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMsg("");
    setError("");
    try {
      const res = await fetch(`${API_BASE}/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error(await res.text());
      setMsg("Registered! You can now log in.");
      setTimeout(() => onSwitch?.("login"), 600);
    } catch (err) {
      setError(err.message || "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  const inputCls =
    "mt-1 w-full rounded-xl border border-white/10 bg-white/5 text-slate-100 placeholder:text-slate-400 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500/40";

  return (
    <div className="max-w-xl w-full mx-auto glass rounded-2xl p-8 mt-12">
      <h1 className="text-xl font-semibold text-slate-100">
        Create your account
      </h1>
      <p className="text-sm text-slate-400 mt-1">It only takes a minute.</p>

      <form
        onSubmit={submit}
        className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-4"
      >
        <div>
          <label className="block text-xs font-medium text-slate-300">
            Name
          </label>
          <input
            name="name"
            value={form.name}
            onChange={onChange}
            required
            className={inputCls}
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-300">
            Email
          </label>
          <input
            type="email"
            name="email"
            value={form.email}
            onChange={onChange}
            required
            className={inputCls}
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-300">
            Student Reg. No.
          </label>
          <input
            name="studentRegNumber"
            value={form.studentRegNumber}
            onChange={onChange}
            required
            className={inputCls}
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-300">
            Date of Birth
          </label>
          <input
            type="date"
            name="dob"
            value={form.dob}
            onChange={onChange}
            required
            className={inputCls}
          />
        </div>
        <div className="sm:col-span-2">
          <label className="block text-xs font-medium text-slate-300">
            Password
          </label>
          <input
            type="password"
            name="password"
            value={form.password}
            onChange={onChange}
            required
            className={inputCls}
          />
        </div>

        {error && (
          <div className="sm:col-span-2 text-sm text-red-300 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
            {error}
          </div>
        )}
        {msg && (
          <div className="sm:col-span-2 text-sm text-green-300 bg-green-500/10 border border-green-500/20 rounded-lg px-3 py-2">
            {msg}
          </div>
        )}

        <div className="sm:col-span-2">
          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-xl bg-indigo-600 text-white py-2.5 hover:bg-indigo-700 shadow-[var(--shadow-soft)] transition"
          >
            {loading ? "Creating account…" : "Create account"}
          </button>
        </div>
      </form>

      <p className="text-xs text-slate-400 mt-4 text-center">
        Already have an account?{" "}
        <button
          className="text-indigo-300 hover:text-indigo-200 underline"
          onClick={() => onSwitch?.("login")}
        >
          Sign in
        </button>
      </p>
    </div>
  );
}
