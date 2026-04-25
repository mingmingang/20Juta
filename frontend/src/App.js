import React, { useState, useEffect } from "react";
import "bootstrap/dist/css/bootstrap.min.css";

// IMPORT PAGES
import LoginPage from "./pages/Login";
import DashboardView from "./pages/Dashboard";
import ProfileView from "./pages/Profile";
import SopView from "./pages/SOP";
import APD from "./pages/APD";
import ConfigView from "./pages/Config";

// IMPORT PARTS (COMPONENTS)
import Sidebar from "./part/Sidebar";
import Header from "./part/Header";
import Toast from "./part/Toast";
import LogoutModal from "./part/Logout";
import UserManagement from "./pages/ManageUser/UserManagement";

function App() {
  // 1. STATE LOGIN (Anti-Reload: Mengambil status dari LocalStorage)
  const [isLoggedIn, setIsLoggedIn] = useState(() => {
    return localStorage.getItem("isLoggedIn") === "true";
  });

  // 2. STATE DATA USER (Menyimpan detail nama, divisi, id dari database)
  const [userData, setUserData] = useState(() => {
    const saved = localStorage.getItem("userData");
    return saved ? JSON.parse(saved) : null;
  });

  // 3. STATE NAVIGASI HALAMAN (Mengontrol menu mana yang aktif)
  const [activePage, setActivePage] = useState("DASHBOARD");

  // 4. STATE KOMPONEN PENDUKUNG (Toast & Modal)
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
  const [toast, setToast] = useState({ show: false, type: "", message: "" });

  // Fungsi untuk memunculkan notifikasi geser (Success/Error)
  const showNotification = (type, message) => {
    setToast({ show: true, type, message });
  };

  // --- LOGIKA SAAT LOGIN SUKSES ---
  // Di dalam App.js
  const handleLoginSuccess = (data) => {
    console.log("Data masuk dari Login:", data); // Lihat di F12 untuk debug

    // Pengecekan ekstra aman
    if (data && data.userData) {
      setIsLoggedIn(true);
      setUserData(data.userData); // Simpan ke State

      // Simpan ke Browser Memori
      localStorage.setItem("isLoggedIn", "true");
      localStorage.setItem("userData", JSON.stringify(data.userData));

      // Gunakan ?. agar jika salah satu kosong, web TIDAK crash
      const name = data.userData?.fullName || "User";
      showNotification("success", `Selamat Datang, ${name}`);
    } else {
      console.error("Format data dari server salah:", data);
      showNotification("error", "Data dari server tidak valid");
    }
  };

  // --- LOGIKA SAAT LOGOUT (Dijalankan dari Modal Konfirmasi) ---
  const handleFinalLogout = () => {
    setIsLoggedIn(false);
    setUserData(null);
    setActivePage("DASHBOARD");
    localStorage.removeItem("isLoggedIn");
    localStorage.removeItem("userData");
    setShowLogoutConfirm(false);
    showNotification("success", "Berhasil keluar dari sistem");
  };

  return (
    <div style={{ backgroundColor: "#f8fafc", minHeight: "100vh" }}>
      {/* 1. NOTIFIKASI TOAST (Hadir Global) */}
      <Toast
        show={toast.show}
        type={toast.type}
        message={toast.message}
        onClose={() => setToast({ ...toast, show: false })}
      />

      {/* 2. MODAL LOGOUT (Akan muncul melayang jika logout diklik) */}
      <LogoutModal
        show={showLogoutConfirm}
        onCancel={() => setShowLogoutConfirm(false)}
        onConfirm={handleFinalLogout}
      />

      {!isLoggedIn ? (
        // ==========================================
        // TAMPILAN JIKA BELUM LOGIN
        // ==========================================
        <LoginPage
          setLoginStatus={handleLoginSuccess}
          notify={showNotification}
        />
      ) : (
        // ==========================================
        // TAMPILAN DASHBOARD SETELAH LOGIN SUKSES
        // ==========================================
        <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
          {/* A. SIDEBAR (BAGIAN KIRI) */}
          <Sidebar
            activePage={activePage}
            userData={userData}
            onNavigateDashboard={() => setActivePage("DASHBOARD")}
            onNavigateProfile={() => setActivePage("PROFILE")}
            // Bisa tambah navigasi lain di sini (misal: SOP atau STATISTIK)
            onNavigateSop={() => setActivePage("SOP")}
            onNavigateApd={() => setActivePage("APD")}
            onNavigateConfig={() => setActivePage("CONFIG")}
            onNavigateStatistik={() => setActivePage("STAT")}
            onLogout={() => setShowLogoutConfirm(true)}
          />

          {/* B. BAGIAN KANAN (AREA UTAMA: HEADER + KONTEN) */}
          <div
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              backgroundColor: "#f8fafc",
            }}
          >
            {/* BAGIAN ATAS: HEADER DINAMIS */}

            <Header
              activePage={activePage}
              userData={userData} // Kirim data person dari DB
              onLogoutClick={() => setShowLogoutConfirm(true)} // Panggil modal logout
              onNavigateProfile={() => setActivePage("PROFILE")} // Navigasi ke page profil
            />

            {/* BAGIAN TENGAH: ISI HALAMAN BERUBAH SESUAI TAB */}
            <div
              style={{
                flex: 1,
                padding: "0 30px 30px 30px", // Memberi padding bawah dan samping
                overflowY: "auto", // Supaya isi dashboard bisa scroll tanpa Sidebar ikut geser
              }}
            >
              {/* Kontrol Konten Berdasarkan activePage */}
              {activePage === "PROFILE" ? (
                <ProfileView
                  userData={userData}
                  notify={showNotification}
                  onBack={() => setActivePage("DASHBOARD")}
                  onUpdateSuccess={(newData) => {
                    setUserData(newData); // Update state global
                    localStorage.setItem("userData", JSON.stringify(newData)); // Update LocalStorage
                  }}
                />
              ) : activePage === "SOP" ? (
                <SopView />
              ) : activePage === "APD" ? (
                <APD />
              ) : activePage === "CONFIG" ? (
                 <ConfigView onNavigateUserMgmt={() => setActivePage('USER_MGMT')} /> 
              ) : activePage === "USER_MGMT" ? (
                <UserManagement
                  onBack={() => setActivePage("CONFIG")}
                  notify={showNotification}
                />
              ) : activePage === "DASHBOARD" ? (
                <DashboardView />
              ) : (
                /* Halaman pengganti untuk tab yang belum dibuat (SOP/Statistik) */
                <div className="d-flex align-items-center justify-content-center h-100 bg-white rounded-5 shadow-sm mt-2">
                  <div className="text-center">
                    <p className="text-muted fw-bold">
                      Modul AI {activePage} Sedang Dalam Pengembangan
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
