import React, { useEffect, useMemo, useState } from "react";
import { AuthProvider, useAuth } from "./store/auth";
import Login from "./components/Login";
import Register from "./components/Register";
import ChatInterface from "./components/ChatInterface";
import Notes from "./components/Notes";
import { LogOut, Sparkles } from "lucide-react";
import CategorySelect from "./components/CategorySelect";

function Shell() {
  const { token, user, logout } = useAuth();
  const [screen, setScreen] = useState(token ? "home" : "login");
  const [notes, setNotes] = useState([]);
  const addNote = (n) => setNotes((p) => [n, ...p]);
  const [category, setCategory] = useState("Algorithms");

  const categoryIcon = useMemo(() => {
    const map = {
      Algorithms: "🧮",
      "Data Structures": "🧱",
      "Operating Systems": "🖥️",
      Physics: "🧪",
      Mathematics: "➗",
    };
    return map[category] ?? "🧠";
  }, [category]);

  const go = (to) => setScreen(to);
  useEffect(() => {
    // when token appears, default to home; when it disappears, go to login
    setScreen((prev) =>
      token ? (prev === "notes" ? "notes" : "home") : "login"
    );
  }, [token]);
  if (!token) {
    return (
      <div className="min-h-screen flex flex-col bg-[#090e1a] text-slate-100">
        <header className="px-6 py-4 flex items-center justify-between border-b border-white/10 bg-[#0b0f19]">
          <div className="flex items-center gap-2">
            <div className="font-semibold tracking-tight">Mentora</div>
          </div>
        </header>

        <main className="flex-1 flex items-start justify-center px-4 py-8">
          <div className="w-full max-w-lg">
            {screen === "login" ? (
              <Login onSwitch={go} />
            ) : (
              <Register onSwitch={go} />
            )}
          </div>
        </main>

        <footer className="py-6 text-center text-xs text-slate-400 border-t border-white/10">
          © {new Date().getFullYear()} Mentora
        </footer>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-[#090e1a] text-slate-100">
      <header className="px-4 md:px-6 py-3 flex items-center justify-between border-b border-white/10 bg-[#0b0f19]">
        <div className="flex items-center gap-3">
          <div className="font-semibold">Mentora</div>
          <div className="ml-3 text-sm text-slate-300">
            Hello, {user?.email}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <CategorySelect
            value={category}
            onChange={setCategory}
            options={[
              "Algorithms",
              "Data Structures",
              "Operating Systems",
              "Physics",
              "Mathematics",
            ]}
          />

          <button
            onClick={logout}
            className="text-sm rounded-lg border border-white/10 px-3 py-1.5 bg-white/5 hover:bg-white/10 flex items-center gap-1"
          >
            <LogOut size={16} /> Logout
          </button>
        </div>
      </header>

      <main className="flex-1 px-3 md:px-6 py-4">
        {screen === "home" && (
          <div className="rounded-2xl border border-white/10 bg-[#0b0f19] shadow-[0_10px_30px_-12px_rgba(0,0,0,0.35)] p-2 sm:p-3">
            <ChatInterface
              key={token || "guest"} // <— force remount on new token
              category={category}
              categoryIcon={categoryIcon}
              onBackClick={() => {}}
              addNote={addNote}
              onNavigate={setScreen}
              initialQuestion=""
            />
          </div>
        )}

        {screen === "notes" && (
          <div className="rounded-2xl border border-white/10 bg-[#0b0f19] shadow-[0_10px_30px_-12px_rgba(0,0,0,0.35)] p-2 sm:p-3">
            <Notes notes={notes} onBack={() => setScreen("home")} />
          </div>
        )}
      </main>

      <footer className="py-6 text-center text-xs text-slate-400 border-t border-white/10">
        © {new Date().getFullYear()} Mentora
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Shell />
    </AuthProvider>
  );
}
