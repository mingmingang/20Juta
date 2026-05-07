import React, { useState } from 'react';
import { Circle, ShieldCheck, AlertTriangle, Activity, BarChart3, Wrench, CheckCircle } from 'lucide-react';
import '../style/APD.css';

const APD = () => {
  return (
    <div className="ppe-container animate-fade-in px-2 pb-4">
      
      {/* --- TOP: RINGKASAN KPIS (Point Penting) --- */}
      <div className="row g-4 mb-4">
        <KPICard title="Compliance Rate" value="98.2%" status="Normal" color="#10b981" />
        <KPICard title="5S Tool Return" value="4/5" status="1 Missing" color="#f59e0b" />
        <KPICard title="Danger Incident" value="0" status="Clean" color="#3b82f6" />
        <KPICard title="Overall Safety" value="A+" status="Excellent" color="#10b981" />
      </div>

      <div className="row g-4">
        {/* --- LEFT: SINGLE CAMERA MONITORING --- */}
        <div className="col-lg-8">
          <div className="card border-0 shadow-sm rounded-5 overflow-hidden bg-white h-100">
            <div className="bg-white p-3 border-bottom d-flex justify-content-between align-items-center px-4">
              <div className="d-flex align-items-center gap-2">
                <Circle size={10} fill="#f43f5e" color="#f43f5e" className="animate-pulse" />
                <span className="fw-black small uppercase letter-spacing-wide text-dark">Live Safety Stream - Station A-1</span>
              </div>
              <span className="badge bg-light text-primary border rounded-pill x-small fw-bold px-3 py-1">WEB-CAM AI</span>
            </div>
            
            <div className="camera-area bg-dark position-relative" style={{ height: '520px' }}>
              <img src="http://127.0.0.1:5001/video_feed" alt="Stream" className="w-100 h-100 object-fit-cover" />
              
              {/* Corner Info: Data yang didapat dari Kamera */}
              <div className="camera-overlay-info">
                 <div className="info-box bg-white p-2 px-3 rounded-4 shadow-sm mb-2">
                    <p className="m-0 x-small fw-bold text-muted uppercase">Detected PPE</p>
                    <h6 className="m-0 fw-black text-success">✓ HELMET ● ✓ VEST ● ✓ SHOES</h6>
                 </div>
              </div>
            </div>
          </div>
        </div>

        {/* --- RIGHT: DATA ANALITIK PENTING --- */}
        <div className="col-lg-4">
          <div className="card border-0 shadow-sm rounded-5 p-4 bg-white mb-4">
             <div className="d-flex align-items-center gap-2 mb-4">
                <BarChart3 size={20} className="text-primary" />
                <h6 className="fw-black fst-italic text-dark m-0 small uppercase">Compliance Analysis</h6>
             </div>
             
             {/* Progress Kepatuhan yang Paling Penting */}
             <ComplianceBar label="Helm Safety" value={100} color="#10b981" />
             <ComplianceBar label="Rompi Reflektor" value={98} color="#10b981" />
             <ComplianceBar label="Sarung Tangan" value={65} color="#ef4444" />
          </div>

          <div className="card border-0 shadow-sm rounded-5 p-4 bg-white">
             <div className="d-flex align-items-center gap-2 mb-4">
                <CheckCircle size={20} className="text-primary" />
                <h6 className="fw-black fst-italic text-dark m-0 small uppercase">5S Checklist (Tools)</h6>
             </div>

             <div className="tool-return-list">
                <ToolItem name="Impact Wrench #1" status="RETURNED" isDone />
                <ToolItem name="Digital Calibrator" status="IN USE" />
                <ToolItem name="Measurement Jig" status="RETURNED" isDone />
             </div>

             <div className="mt-4 pt-3 border-top text-center">
                <span className="small text-muted fw-bold">Daily Safety Suggestion:</span>
                <p className="x-small fw-medium text-dark mt-1">"Operator kelelahan terdeteksi pada Shift 2. Rekomendasi istirahat singkat."</p>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// --- SUB KOMPONEN (Penting untuk visual bersih) ---

const KPICard = ({ title, value, status, color }) => (
  <div className="col-md-3">
    <div className="card border-0 shadow-sm p-4 rounded-5 bg-white border border-light transition-hover">
       <p className="m-0 text-muted fw-bold x-small uppercase tracking-widest mb-1">{title}</p>
       <div className="d-flex align-items-baseline gap-2">
          <h2 className="fw-black fst-italic m-0" style={{fontSize: '32px'}}>{value}</h2>
          <span className="x-small fw-bold fst-italic" style={{color: color}}>{status}</span>
       </div>
    </div>
  </div>
);

const ComplianceBar = ({ label, value, color }) => (
  <div className="mb-4">
    <div className="d-flex justify-content-between mb-2">
       <span className="x-small fw-black text-dark uppercase">{label}</span>
       <span className="x-small fw-black">{value}%</span>
    </div>
    <div className="progress rounded-pill" style={{height:'10px', background:'#f1f5f9'}}>
       <div className="progress-bar rounded-pill shadow-sm" style={{width: `${value}%`, background: color}}></div>
    </div>
  </div>
);

const ToolItem = ({ name, status, isDone }) => (
  <div className={`d-flex justify-content-between align-items-center p-2 mb-2 rounded-4 ${isDone ? 'bg-light opacity-50' : 'bg-primary-subtle border-primary border'}`}>
    <span className="x-small fw-bold">{name}</span>
    <span className={`fw-black ${isDone ? 'text-success' : 'text-primary'}`} style={{fontSize:'9px'}}>{status}</span>
  </div>
);

export default APD;