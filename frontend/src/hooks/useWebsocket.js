import { useEffect, useRef, useState, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8765";
const WS_BASE = import.meta.env.VITE_WS_BASE || API_BASE.replace(/^http/, "ws");

export function useMentoraSocket(token) {
  const wsRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState(null);

  const connect = useCallback(() => {
    if (!token || wsRef.current) return;
    const url = `${WS_BASE}/ws?token=${encodeURIComponent(token)}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        setMessages((prev) => [
          ...prev,
          { role: "assistant", data, ts: Date.now() },
        ]);
      } catch (e) {
        setError("Invalid message from server");
      }
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
    };

    ws.onerror = () => {
      setError("WebSocket error");
    };
  }, [token]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (token) connect();
    return () => disconnect();
  }, [token, connect, disconnect]);

  const send = useCallback((payload) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      throw new Error("Socket not connected");
    }
    wsRef.current.send(JSON.stringify(payload));
    // also push the user message into the timeline
    setMessages((prev) => [
      ...prev,
      { role: "user", data: payload, ts: Date.now() },
    ]);
  }, []);

  return { connected, messages, send, error };
}
