import React, { useState } from "react";
import "../style/Login.css";
import { ShieldCheck, User, Lock, ChevronRight } from "lucide-react";

const LoginPage = ({ setLoginStatus, notify }) => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  // Di dalam Login.js
  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch("http://localhost:5001/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      const data = await response.json(); // Data berisi { success, userData }

      if (data.success) {
        // PANGGIL PROPS LOGIN DENGAN MELEMPAR 'data'
        setLoginStatus(data);
      } else {
        notify("error", "Username atau Password salah!");
      }
    } catch (err) {
      notify("error", "Koneksi Server Gagal");
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        {/* Shield Icon SVG */}
        <div
          className="bg-primary rounded-4 d-inline-flex align-items-center justify-content-center mb-3"
          style={{
            width: "64px",
            height: "64px",
            boxShadow: "0 8px 16px rgba(13, 110, 253, 0.25)",
          }}
        >
          <ShieldCheck size={35} color="white" strokeWidth={2} />
        </div>

        <h1 className="brand-name">SOPGuard AI</h1>
        <p className="subtitle">Smart Manufacturing Vision System</p>

        <form onSubmit={handleLogin}>
          <div className="input-group">
            <label>Username</label>
            <div className="input-wrapper">
              <span
                style={{ position: "absolute", left: "15px", color: "#cbd5e1" }}
              >
                <User size={18} className="text-secondary" />
              </span>
              <input
                type="text"
                placeholder="Enter Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
          </div>

          <div className="input-group">
            <label>Password</label>
            <div className="input-wrapper">
              <span
                style={{ position: "absolute", left: "15px", color: "#cbd5e1" }}
              >
                <Lock size={18} className="text-secondary" />
              </span>
              <input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>

          <button type="submit" className="login-button">
            LOGIN <span>→</span>
          </button>
        </form>
      </div>
      <div className="footer-text">
        TMIND - Team ASTRAtech ScanInnovate • v1.0.0
      </div>
    </div>
  );
};

export default LoginPage;
