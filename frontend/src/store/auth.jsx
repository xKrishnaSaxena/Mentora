import React, { createContext, useContext, useEffect, useState } from "react";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(
    () => localStorage.getItem("authToken") || null
  );
  const [user, setUser] = useState(() => {
    const raw = localStorage.getItem("authUser");
    return raw ? JSON.parse(raw) : null;
  });

  const login = (tok, usr) => {
    setToken(tok);
    setUser(usr || null);
    localStorage.setItem("authToken", tok);
    if (usr) localStorage.setItem("authUser", JSON.stringify(usr));
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem("authToken");
    localStorage.removeItem("authUser");
  };

  useEffect(() => {
    const onStorage = () => {
      setToken(localStorage.getItem("authToken") || null);
      const raw = localStorage.getItem("authUser");
      setUser(raw ? JSON.parse(raw) : null);
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  return (
    <AuthContext.Provider value={{ token, user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
