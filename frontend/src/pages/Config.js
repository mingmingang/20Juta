import React from 'react';
import { 
  Settings, UserCog, ShieldCheck, Cpu, 
  Camera, Terminal, FileBarChart 
} from 'lucide-react';
import '../style/Config.css';

const ConfigView = ({ onNavigateUserMgmt }) => { // Tambahkan Props navigasi
  const configMenu = [
    { 
      id: "USER_MGMT", // Ubah ID agar mudah dideteksi
      title: "MANAJEMEN USER", 
      subtitle: "AUDIT HAK AKSES", 
      icon: <UserCog size={32} /> 
    },
    { 
      id: "THRESHOLD", 
      title: "THRESHOLD APD", 
      subtitle: "AKURASI DETEKSI", 
      icon: <ShieldCheck size={32} /> 
    },
    { 
      id: "WEIGHTS", 
      title: "MODEL WEIGHTS", 
      subtitle: "UPDATE YOLO V11", 
      icon: <Cpu size={32} /> 
    },
    { 
      id: "CAM_SET", 
      title: "PENGATURAN CAM", 
      subtitle: "STREAM & FOV", 
      icon: <Camera size={32} /> 
    },
    { 
      id: "LOGS", 
      title: "SYSTEM LOGS", 
      subtitle: "DEBUG & AUDIT", 
      icon: <Terminal size={32} /> 
    },
    { 
      id: "REPORTS", 
      title: "FORMAT LAPORAN", 
      subtitle: "PDF & EXCEL", 
      icon: <FileBarChart size={32} /> 
    },
  ];

  return (
    <div className="config-container animate-fade-in text-center px-4">
      
      {/* 1. HEADER PUSAT KONFIGURASI */}
      <div className="config-header">
        <div className="main-settings-icon shadow-sm mx-auto">
          <Settings size={45} color="#2563eb" strokeWidth={1.5} />
        </div>
        <h1 className="fw-black fst-italic text-dark tracking-tight mt-3">
            PUSAT KONFIGURASI AI
        </h1>
        <p className="text-muted mx-auto mb-5" style={{ maxWidth: '600px', fontSize: '15px', fontWeight: 500 }}>
            Manajemen parameter deteksi YOLO, otorisasi personil, dan kustomisasi notifikasi SOPGuard.
        </p>
      </div>

      {/* 2. GRID CARDS */}
      <div className="row g-4 justify-content-center">
        {configMenu.map((item) => (
          <div key={item.id} className="col-lg-4 col-md-6">
            <div 
              className="config-card p-4 h-100 shadow-hover clickable"
              onClick={() => {
                 // JIKA YANG DIKLIK ADALAH MANAJEMEN USER
                 if(item.id === "USER_MGMT") onNavigateUserMgmt();
              }}
            >
              <div className="config-icon-box mb-4">
                {item.icon}
              </div>
              <h5 className="fw-black m-0">{item.title}</h5>
              <p className="config-subtitle text-muted mt-1 small fw-bold tracking-widest uppercase">
                {item.subtitle}
              </p>
            </div>
          </div>
        ))}
      </div>

    </div>
  );
};

export default ConfigView;