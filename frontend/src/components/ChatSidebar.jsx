import { Plus, Trash2, Edit3 } from "lucide-react";
import { useState, useEffect } from "react";

export default function ChatSidebar({ token, activeChatId, onSelectChat }) {
  const [chats, setChats] = useState([]);
  const headers = { Authorization: `Bearer ${token}` };

  const load = async () => {
    const r = await fetch("http://localhost:8765/chats", { headers });
    if (!r.ok) return;
    const data = await r.json();
    setChats(data);

    const saved = localStorage.getItem("activeChatId");
    if (saved && data.some((c) => c.id === saved)) {
      if (activeChatId !== saved) onSelectChat(saved);
      return;
    }

    if (!activeChatId && data[0]) onSelectChat(data[0].id);
  };

  useEffect(() => {
    load();
  }, []);

  const newChat = async () => {
    const r = await fetch("http://localhost:8765/chats", {
      method: "POST",
      headers: { ...headers, "Content-Type": "application/json" },
      body: JSON.stringify({ title: "New chat", mode: "teach" }),
    });
    if (!r.ok) return;
    const data = await r.json();
    await load();
    onSelectChat(data.id);
  };

  const renameChat = async (id) => {
    const title = prompt("Rename chat:");
    if (!title) return;
    await fetch(`http://localhost:8765/chats/${id}`, {
      method: "PATCH",
      headers: { ...headers, "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    load();
  };

  const deleteChat = async (id) => {
    if (!confirm("Delete chat?")) return;
    await fetch(`http://localhost:8765/chats/${id}`, {
      method: "DELETE",
      headers,
    });
    load();
    if (id === activeChatId) onSelectChat(null);
  };
  return (
    <div className="flex h-full w-72 min-w-60 shrink-0 flex-col border-r border-white/10 bg-[#0b0f19]">
      <div className="p-3">
        <button
          onClick={newChat}
          className="w-full inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-200 hover:bg-white/10"
        >
          <Plus size={16} /> New chat
        </button>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto px-2 pb-3 space-y-1">
        {chats.map((c) => (
          <div
            key={c.id}
            className={`group flex items-center justify-between rounded-lg px-2 py-2 text-sm cursor-pointer ${
              activeChatId === c.id ? "bg-white/10" : "hover:bg-white/5"
            }`}
            onClick={() => onSelectChat(c.id)}
          >
            <div className="truncate">{c.title}</div>
            <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1">
              <button
                className="p-1 rounded hover:bg-white/10"
                onClick={(e) => {
                  e.stopPropagation();
                  renameChat(c.id);
                }}
                title="Rename"
              >
                <Edit3 size={14} />
              </button>
              <button
                className="p-1 rounded hover:bg-white/10 text-rose-300"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteChat(c.id);
                }}
                title="Delete"
              >
                <Trash2 size={14} />
              </button>
            </div>
          </div>
        ))}
        {!chats.length && (
          <div className="text-xs text-slate-400 px-2 py-1">No chats yet.</div>
        )}
      </div>
    </div>
  );
}
