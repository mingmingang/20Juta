import React from 'react';
import { Calendar, Download, CheckCircle2, Clock, CircleDot, HelpCircle } from 'lucide-react';
import '../style/SOP.css';

const SopView = () => {
  return (
    <div className="sop-container animate-fade-in">
      
      {/* 1. HEADER CARD (Judul Lini & Download) */}
      <div className="sop-header-card shadow-sm border-0 card p-4 mb-4 rounded-4">
        <div className="d-flex justify-content-between align-items-center">
          <div>
            <h2 className="sop-main-title fw-black fst-italic m-0">FIX TIME FIX POSITION STATUS</h2>
            <div className="d-flex gap-3 mt-2">
              <span className="info-item"><Calendar size={14} className="me-1" /> SHIFT 1 (SENIN - KAMIS)</span>
              <span className="info-item"><CircleDot size={14} className="me-1" /> LINI ASSY A</span>
            </div>
          </div>
          <button className="btn-download-report shadow-sm">
            <Download size={18} className="me-2" /> DOWNLOAD REPORT
          </button>
        </div>
      </div>

      {/* 2. GROUP PERSIAPAN PRODUKSI */}
      <div className="sop-group-section mb-5">
        <div className="group-title-box mb-4">
          <div className="number-badge">1</div>
          <h5 className="group-name">PERSIAPAN PRODUKSI</h5>
        </div>
        
        <div className="row g-4">
          <StatusCard title="CHECK KESEHATAN MEMBER" subtitle="CHECK SHEET" time="5m" status="VERIFIED" />
          <StatusCard title="CHECK ABSENSI" subtitle="CHECK SHEET" time="5m" status="VERIFIED" />
          <StatusCard title="CHECK EQUIPMENT (TPM)" subtitle="CHECK SHEET" time="5m" status="PROGRESS" />
          <StatusCard title="CHECK ACTUAL STOCK PART" subtitle="CHECK SHEET" time="5m" status="WAIT" />
          <StatusCard title="INFORMASI KOMUNIKASI" subtitle="BUKU KOMUNIKASI" time="5m" status="WAIT" />
        </div>
      </div>

      {/* 3. GROUP PROSES PRODUKSI */}
      <div className="sop-group-section mb-4">
        <div className="group-title-box mb-4">
          <div className="number-badge">2</div>
          <h5 className="group-name">PROSES PRODUKSI</h5>
        </div>
        
        <div className="row g-4">
          <StatusCard title="SKILL MP = TANOKO" subtitle="CHECK SHEET" time="30m" status="WAIT" />
          <StatusCard title="OUT PUT EQUIPMENT = STD INSPEKSI" subtitle="CHECK SHEET" time="20m" status="WAIT" />
          <StatusCard title="OUT PUT PROSES = STANDAR QUALITY" subtitle="QUALITY INSP & GATE" time="25m" status="WAIT" />
        </div>
      </div>

    </div>
  );
};

// --- KOMPONEN KARTU STATUS (REUSABLE) ---
const StatusCard = ({ title, subtitle, time, status }) => {
  const getStatusConfig = () => {
    switch (status) {
      case 'VERIFIED': return { class: 'verified', icon: <CheckCircle2 size={16} />, text: 'VERIFIED' };
      case 'PROGRESS': return { class: 'progressing', icon: <Clock size={16} />, text: 'IN PROGRESS' };
      default: return { class: 'waiting', icon: <HelpCircle size={16} />, text: 'WAIT' };
    }
  };

  const config = getStatusConfig();

  return (
    <div className="col-lg-4 col-md-6">
      <div className={`status-card p-4 rounded-4 border-0 card h-100 ${config.class}`}>
        <div className="d-flex justify-content-between mb-3">
          <h6 className="item-title fw-black">{title}</h6>
          <span className="badge-time">{time}</span>
        </div>
        <div className="d-flex justify-content-between align-items-end mt-auto">
          <span className="item-subtitle">{subtitle}</span>
          <div className="status-label d-flex align-items-center gap-2">
            {config.icon} <strong>{config.text}</strong>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SopView;