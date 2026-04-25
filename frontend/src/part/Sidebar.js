import React, { useState } from "react";
import {
  Shield,
  LayoutGrid,
  FileText,
  ShieldCheck,
  Settings,
  LogOut,
  ChevronRight,
  Menu,
  User,
} from "lucide-react";
import "../style/Sidebar.css";

// Pastikan props didefinisikan semua: onLogout, onNavigateDashboard, onNavigateProfile, activePage, userData
const Sidebar = ({ 
  onLogout, 
  onNavigateDashboard, 
  onNavigateSop,
  onNavigateProfile, 
  onNavigateConfig,
  onNavigateApd,
  activePage, 
  userData 
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Mengubah ID agar sesuai dengan state activePage di App.js
  const menuAccess = [
    { name: "OVERVIEW UTAMA", icon: <LayoutGrid size={22} />, id: "DASHBOARD" },
    { name: "PROSEDUR (SOP)", icon: <FileText size={22} />, id: "SOP" },
    { name: "STATISTIK APD", icon: <ShieldCheck size={22} />, id: "APD" },
    { name: "PUSAT KONFIGURASI", icon: <Settings size={22} />, id: "CONFIG" },
  ];

  return (
    <div className={`sidebar-container ${isCollapsed ? "collapsed" : ""}`}>
      {/* 1. BRANDING / LOGO */}
      <div className="sidebar-brand">
        <div className="brand-logo-wrapper" onClick={onNavigateDashboard} style={{cursor: 'pointer'}}>
          {!isCollapsed && (
            <div className="brand-icon-box">
              <ShieldCheck
                fill="#3b82f6"
                color="#3b82f6"
                size={32}
                strokeWidth={1}
              />
            </div>
          )}

          {!isCollapsed && (
            <div className="brand-text">
              <h2 className="brand-main">SOPGUARD</h2>
              <p className="brand-sub">AI ANALYTICS</p>
            </div>
          )}
        </div>

        {/* Tombol toggle collapse */}
        <button
          className="collapse-btn-toggle"
          onClick={() => setIsCollapsed(!isCollapsed)}
        >
          <Menu size={20} color="white" />
        </button>
      </div>

      {/* 2. DYNAMIC MENU */}
      <nav className="sidebar-menu">
        {menuAccess.map((menu) => (
          <div
            key={menu.id}
            // Class active ditentukan dari activePage milik App.js
            className={`menu-item ${activePage === menu.id ? "active" : ""}`}
            onClick={() => {
              if (menu.id === "DASHBOARD") onNavigateDashboard();
              if (menu.id === "SOP") onNavigateSop();
              if (menu.id === "APD") onNavigateApd();
              if (menu.id === "CONFIG") onNavigateConfig();
              // Tambahkan kondisi untuk SOP/Statistik jika sudah ada halamannya
            }}
          >
            <div className="menu-icon">{menu.icon}</div>
            {!isCollapsed && (
              <>
                <span className="menu-name">{menu.name}</span>
                {activePage === menu.id && (
                  <ChevronRight size={16} className="active-arrow" />
                )}
              </>
            )}
          </div>
        ))}
      </nav>

      {/* 3. PROFILE & LOGOUT SECTION */}
      <div className="sidebar-footer">
        {/* User Card - Jika diklik pindah ke halaman profile */}
        <div 
          className={`user-profile-card ${activePage === "PROFILE" ? "active-profile" : ""}`} 
          onClick={onNavigateProfile}
          style={{ cursor: "pointer" }}
        >
          <div className="avatar-box">
            {/* Pakai avatar bot kalau belum ada foto asli */}
            <img 
               src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${userData?.username || 'user'}`} 
               alt="Avatar" 
               style={{width: '100%', borderRadius: '12px'}}
            />
          </div>
          {!isCollapsed && (
            <div className="user-info">
              {/* NAMA DI AMBIL DARI DATABASE (userData) */}
              <h4>{userData?.fullName?.toUpperCase() || "BACHTIAR.W"}</h4>
              <p>
                PROFIL AKUN <span>›</span>
              </p>
            </div>
          )}
        </div>

        {/* Tombol Logout (Membuka Modal Konfirmasi) */}
        <button className="signout-btn" onClick={onLogout}>
          <LogOut size={18} />
          {!isCollapsed && <span>SIGN OUT</span>}
        </button>
      </div>
    </div>
  );
};

export default Sidebar;