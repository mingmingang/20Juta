import React, { useState, useEffect } from 'react';
import { Circle, Zap, CheckCircle2, AlertCircle, Info, Activity, History } from 'lucide-react';
import '../style/SOP.css';

const SopView = () => {
  const [steps, setSteps] = useState([]);
  const [mismatchLogs, setMismatchLogs] = useState([]);

  useEffect(() => {
    // Simulasi data urutan kerja
    fetch('http://localhost:5001/api/work-sequence').then(r => r.json()).then(setSteps);
    // Kita anggap kita ambil logs insiden untuk area mismatch
    fetch('http://localhost:5001/api/incidents').then(r => r.json()).then(setMismatchLogs);
  }, []);

  return (
    <div className="sop-wrapper animate-fade-in px-2">
      <div className="row g-4">
        
        {/* ================== KIRI (COL 8): BEARING MONITORING HUB ================== */}
        <div className="col-lg-8">
          {/* CAMERA SECTION */}
          <div className="card border-0 shadow-sm rounded-5 overflow-hidden bg-white mb-4">
            <div className="bg-white p-3 border-bottom d-flex justify-content-between align-items-center px-4">
              <div className="d-flex align-items-center gap-2">
                <Circle size={10} fill="#f43f5e" color="#f43f5e" className="animate-pulse" />
                <span className="fw-black small text-uppercase letter-spacing-wide">High Precision - Bearing Inspection</span>
              </div>
              <span className="badge bg-dark rounded-pill px-3 py-2 fw-bold" style={{fontSize: '9px'}}>CV ACTIVE</span>
            </div>

            <div className="sop-camera-area bg-dark position-relative" style={{ height: 'auto' }}>
              <img src="http://127.0.0.1:5001/checkbearing_feed" alt="Bearing Focus" className="w-100 h-100" />
              
              {/* Overlay Deteksi Spesifik */}
              <div className="vision-labels-overlay">
                 <div className="tracking-marker-blue shadow">BEARING_ID: A-420</div>
                 <div className="tracking-marker-green shadow">STATUS: ALIGNED</div>
              </div>

              {/* Alert Card jika terjadi Mismatch */}
              <div className="mismatch-alert-float card border-0 p-3 rounded-4 shadow-xl backdrop-blur animate-bounce-slow">
                 <div className="d-flex align-items-center gap-3">
                    <div className="bg-danger p-2 rounded-3 text-white"><AlertCircle size={20}/></div>
                    <div>
                       <h6 className="m-0 fw-black text-white italic">DETEKSI MISMATCH</h6>
                       <p className="m-0 x-small text-white opacity-75">Tipe bearing tidak sesuai Work Instruction!</p>
                    </div>
                 </div>
              </div>
            </div>
          </div>

          {/* MISMATCH ANALYTICS GRAPH (Area Chart Simpel) */}
          <div className="card border-0 shadow-sm rounded-5 p-4 bg-white">
             <div className="d-flex justify-content-between align-items-center mb-4">
                <h6 className="fw-black fst-italic m-0 small letter-spacing-wide uppercase">Analisa Error Pemasangan (Daily)</h6>
                <div className="small fw-bold text-muted">Akurasi Rata-rata: <span className="text-primary">99.2%</span></div>
             </div>
             <div className="chart-placeholder bg-light rounded-4 d-flex align-items-end justify-content-between p-3" style={{height:'120px'}}>
                {[40, 20, 60, 30, 80, 25, 45].map((h, i) => (
                  <div key={i} className="text-center w-100">
                    <div className="bg-primary rounded-pill mx-auto mb-2 opacity-75" style={{height: `${h}px`, width:'12px'}}></div>
                    <span className="x-small text-muted fw-bold">D{i+1}</span>
                  </div>
                ))}
             </div>
          </div>
        </div>

        {/* ================== KANAN (COL 4): SEQUENCE & REAL-TIME LOGS ================== */}
        <div className="col-lg-4">
          
          {/* WORK INSTRUCTION STEPS */}
          <div className="card border-0 shadow-sm rounded-5 p-4 bg-white mb-4">
             <div className="d-flex align-items-center gap-2 mb-4">
                <Zap size={18} className="text-primary" />
                <h6 className="fw-black fst-italic text-dark m-0 text-uppercase small">Work Instruction Hub</h6>
             </div>
             <div className="wi-sequence-list">
                {steps.map((step, i) => (
                  <div key={i} className={`wi-item mb-3 p-3 rounded-4 border ${step.status}`}>
                     <div className="d-flex justify-content-between">
                        <span className="x-small fw-black opacity-50 italic">SEQ_{step.id}</span>
                        {step.status === 'done' && <CheckCircle2 size={16} className="text-success" />}
                     </div>
                     <h6 className="fw-black mt-1 mb-1">{step.name}</h6>
                     {step.status === 'active' && <p className="m-0 x-small fw-bold opacity-75 text-primary">SCANNING LIVE...</p>}
                  </div>
                ))}
             </div>
             <button className="btn btn-primary w-100 rounded-4 py-3 fw-black shadow-blue mt-2">COMPLETE WORK ORDER</button>
          </div>

          {/* MISMATCH HISTORY LOGS */}
          <div className="card border-0 shadow-sm rounded-5 p-4 bg-white">
             <div className="d-flex align-items-center gap-2 mb-4">
                <History size={18} className="text-muted" />
                <h6 className="fw-black fst-italic text-dark m-0 text-uppercase small">Mismatch History</h6>
             </div>
             <div className="logs-scroll" style={{maxHeight:'230px', overflowY:'auto'}}>
                {mismatchLogs.map((log, i) => (
                   <div key={i} className="log-item d-flex gap-3 mb-3 pb-3 border-bottom border-light">
                      <div className="log-dot mt-1" style={{backgroundColor: log.color}}></div>
                      <div>
                         <div className="d-flex gap-2 align-items-center">
                            <span className="fw-black small">{log.title}</span>
                            <span className="x-small text-muted">{log.time}</span>
                         </div>
                         <p className="m-0 x-small text-muted line-height-sm">{log.msg}</p>
                      </div>
                   </div>
                ))}
             </div>
          </div>

        </div>

      </div>
    </div>
  );
};

export default SopView;