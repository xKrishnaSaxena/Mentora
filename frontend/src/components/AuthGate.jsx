import React, { useState } from "react";
import Login from "./Login";
import Register from "./Register";

export default function AuthGate() {
  const [tab, setTab] = useState("login");

  return (
    <div className="max-w-md mx-auto mt-16 bg-white border rounded-xl shadow p-6">
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setTab("login")}
          className={`flex-1 py-2 rounded ${
            tab === "login" ? "bg-indigo-600 text-white" : "bg-gray-100"
          }`}
        >
          Login
        </button>
        <button
          onClick={() => setTab("register")}
          className={`flex-1 py-2 rounded ${
            tab === "register" ? "bg-indigo-600 text-white" : "bg-gray-100"
          }`}
        >
          Register
        </button>
      </div>
      {tab === "login" ? (
        <Login />
      ) : (
        <Register onRegistered={() => setTab("login")} />
      )}
    </div>
  );
}
