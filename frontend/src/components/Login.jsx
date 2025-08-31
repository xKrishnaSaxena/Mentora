import React, { useState } from "react";
import { useAuth } from "../store/auth";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8765";

export default function Login({ onSwitch }) {
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      login(data.token, { email });
      onSwitch?.("home");
    } catch (err) {
      setError(err.message || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md w-full mx-auto glass rounded-2xl p-8 mt-12">
      <h1 className="text-xl font-semibold text-slate-100">
        Sign in to Mentora
      </h1>
      <p className="text-sm text-slate-400 mt-1">
        Welcome back. Enter your credentials.
      </p>

      <form onSubmit={submit} className="mt-6 space-y-4">
        <div>
          <label className="block text-xs font-medium text-slate-300">
            Email
          </label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="mt-1 w-full rounded-xl border border-white/10 bg-white/5 text-slate-100 placeholder:text-slate-400 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500/40"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-300">
            Password
          </label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            className="mt-1 w-full rounded-xl border border-white/10 bg-white/5 text-slate-100 placeholder:text-slate-400 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500/40"
          />
        </div>

        {error && (
          <div className="text-sm text-red-300 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-xl bg-indigo-600 text-white py-2.5 hover:bg-indigo-700 shadow-[var(--shadow-soft)] transition"
        >
          {loading ? "Signing in…" : "Sign in"}
        </button>
      </form>

      <p className="text-xs text-slate-400 mt-4 text-center">
        Don’t have an account?{" "}
        <button
          className="text-indigo-300 hover:text-indigo-200 underline"
          onClick={() => onSwitch?.("register")}
        >
          Create one
        </button>
      </p>
    </div>
  );
}
