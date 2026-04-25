import React, { useState, useEffect } from 'react';
import { Bell, User, LogOut, Settings, ChevronDown, Circle } from 'lucide-react';

const Header = ({ activePage, userData, onLogoutClick, onNavigateProfile }) => {
  const [time, setTime] = useState(new Date());
  const [showUserMenu, setShowUserMenu] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const pageTitles = {
    'DASHBOARD': 'DASHBOARD VIEW',
    'SOP': 'PROSEDUR (SOP)',
    'STAT': 'STATISTIK APD',
    'CONFIG': 'KONFIGURASI AI',
    'PROFILE': 'PROFIL AKUN',
    'USER_MGMT' : 'MANAGEMENT USER'
  };

  return (
    <div className="d-flex justify-content-between align-items-center mb-4 py-3 pe-4 bg-white position-relative" 
         style={{ borderBottom: '1px solid #f1f5f9', paddingLeft: '30px', zIndex: 100 }}>
      
      {/* BAGIAN KIRI: JUDUL */}
      <div>
        <p className="text-muted fw-bold m-0" style={{ fontSize: '10px', letterSpacing: '2px', opacity: 0.7 }}>
          SOP GUARD AI ANALYTICS
        </p>
        <h3 className="fw-black fst-italic text-dark m-0" style={{ fontWeight: 900 }}>
          {pageTitles[activePage] || 'SYSTEM VIEW'}
        </h3>
      </div>

      {/* BAGIAN KANAN: NOTIF, JAM, USER */}
      <div className="d-flex align-items-center">
        
        {/* ICON NOTIFIKASI */}
        <button className="btn border-0 position-relative p-2 me-3" style={{ background: '#f8fafc', borderRadius: '12px' }}>
          <Bell size={20} color="#64748b" />
          <span className="position-absolute top-0 start-100 translate-middle p-1 bg-danger border border-light rounded-circle"></span>
        </button>

        <div className="mx-2" style={{ borderLeft: '1px solid #e2e8f0', height: '30px' }}></div>

        {/* JAM WIB */}
        <div className="text-end px-3">
          <h4 className="m-0 fw-bold text-dark" style={{ letterSpacing: '1px' }}>
            {time.toLocaleTimeString('en-US', { hour12: true, hour: 'numeric', minute: '2-digit', second: '2-digit' })}
          </h4>
          <p className="m-0 text-primary fw-bold fst-italic text-end" style={{ fontSize: '10px' }}>REAL-TIME WIB</p>
        </div>

        <div className="mx-2" style={{ borderLeft: '1px solid #e2e8f0', height: '30px' }}></div>

        {/* USER PROFILE ACTION */}
        <div className="position-relative ms-2">
          <div 
            className="d-flex align-items-center gap-2 px-3 py-2 rounded-pill shadow-sm cursor-pointer" 
            style={{ backgroundColor: '#f0f9ff', border: '1px solid #e0f2fe', cursor: 'pointer' }}
            onClick={() => setShowUserMenu(!showUserMenu)}
          >
            <div className="avatar-box">
            {/* Pakai avatar bot kalau belum ada foto asli */}
            <img 
               src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${userData?.username || 'user'}`} 
               alt="Avatar" 
               style={{width: '70%', borderRadius: '12px'}}
            />
          </div>
            <span className="fw-bold text-dark d-none d-md-inline" style={{ fontSize: '12px' }}>
               {userData?.fullName?.split(' ')[0] || 'User'}
            </span>
            <ChevronDown size={14} color="#64748b" />
          </div>

          {/* POP-UP MENU DROPDOWN */}
          {showUserMenu && (
            <div className="position-absolute end-0 mt-2 shadow-lg border-0 card p-3 animate-slide-up" 
                 style={{ width: '250px', borderRadius: '20px', zIndex: 999 }}>
              
              <div className="d-flex align-items-center gap-3 mb-3 pb-3 border-bottom">
                 <div className="avatar-box">
            {/* Pakai avatar bot kalau belum ada foto asli */}
            <img 
               src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${userData?.username || 'user'}`} 
               alt="Avatar" 
               style={{width: '100%', borderRadius: '12px'}}
            />
          </div>
                 <div className="overflow-hidden">
                    <h6 className="m-0 fw-bold text-dark text-truncate">{userData?.fullName || 'Bachtiar. W'}</h6>
                    <p className="m-0 text-muted" style={{fontSize: '11px'}}>{userData?.employeeId || 'ID: TMMIN-001'}</p>
                 </div>
              </div>

              <div className="d-flex flex-column gap-1">
                <button className="btn btn-light border-0 text-start py-2 px-3 rounded-3 d-flex align-items-center gap-3 small fw-bold" 
                        onClick={() => {onNavigateProfile(); setShowUserMenu(false)}}>
                  <Settings size={16} className="text-secondary" /> Edit Profile
                </button>
                <button className="btn btn-light border-0 text-start py-2 px-3 rounded-3 d-flex align-items-center gap-3 small fw-bold text-danger" 
                        onClick={() => {onLogoutClick(); setShowUserMenu(false)}}>
                  <LogOut size={16} /> Logout
                </button>
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
};

export default Header;