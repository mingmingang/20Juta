import React from 'react';
import { Info, Circle, Zap } from 'lucide-react';
import '../style/APD.css';

const APD = () => {
  return (
    <div className="ppe-container animate-fade-in">
      
      <div className="row g-4">
        {/* SISI KIRI: LIVE CAMERA DETECTION */}
        <div className="col-lg-8">
          <div className="card border-0 shadow-sm rounded-5 overflow-hidden bg-white h-100">
            <div className="bg-white p-3 border-bottom d-flex justify-content-between align-items-center px-4">
               <div className="d-flex align-items-center gap-2">
                  <Circle size={10} fill="#10b981" color="#10b981" />
                  <span className="fw-bold small">DETEKSI KEPATUHAN APD (REAL PERSON)</span>
               </div>
               <span className="badge bg-dark rounded-pill px-3" style={{fontSize:'10px'}}>LIVE STREAM</span>
            </div>
            <div className="camera-display bg-dark position-relative" style={{ minHeight: '500px' }}>
                {/* REPRESENTASI DIAGRAM DETEKSI SESUAI GAMBAR */}
                <div className="detection-overlay d-flex flex-column align-items-center justify-content-center h-100">
                    <img 
                      src="http://127.0.0.1:5001/video_feed" 
                      alt="Stream" 
                      className="w-100 h-100 opacity-50" 
                    />
                    
                    {/* Bounding Box Simulasi (Labeling Tengah) */}
                    <div className="simulated-labels">
                        <div className="box helm text-center">HELM: 99.8%</div>
                        <div className="box body text-center">ROMPI: OK</div>
                        <div className="box shoe text-center">SEPATU SAFETY: OK</div>
                    </div>
                </div>

                <div className="tech-badge-footer position-absolute bottom-0 start-0 p-3">
                   <div className="bg-dark bg-opacity-75 text-white px-3 py-1 rounded-pill small border border-secondary border-opacity-25">
                      <Zap size={12} className="me-2 text-primary" /> YOLOv11s • 12.4ms
                   </div>
                </div>
            </div>
          </div>
        </div>

        {/* SISI KANAN: ANALISA & TIPS */}
        <div className="col-lg-4 d-flex flex-column gap-4">
          
          {/* ANALISA PELANGGARAN CARD */}
          <div className="card border-0 shadow-sm rounded-5 p-4 bg-white">
            <h6 className="fw-black fst-italic text-muted mb-4 small letter-spacing-wide">ANALISA PELANGGARAN APD</h6>
            <div className="space-y-4">
                <ViolationBar label="HELM SAFETY" value={99.8} color="#ef4444" />
                <ViolationBar label="ROMPI REFLEKTOR" value={95.2} color="#f43f5e" />
                <ViolationBar label="SEPATU SAFETY" value={100} color="#10b981" />
                <ViolationBar label="SARUNG TANGAN" value={92.4} color="#f87171" />
            </div>
          </div>

          {/* TIPS KEPATUHAN CARD */}
          <div className="card border-0 shadow-sm rounded-5 p-4 text-white position-relative overflow-hidden h-100" style={{ backgroundColor: '#2563eb' }}>
             <div className="position-relative z-index-10">
                <div className="bg-white bg-opacity-25 p-2 d-inline-block rounded-circle mb-3">
                    <Info size={24} />
                </div>
                <h4 className="fw-black fst-italic mb-2">TIPS KEPATUHAN</h4>
                <p className="small opacity-75 lh-base">
                  Sistem mendeteksi penurunan penggunaan sarung tangan di Shift 2. 
                  Pastikan stok APD tersedia di workstation A-12.
                </p>
             </div>
             {/* Decorative Background Batik Sederhana */}
             <div className="batik-pattern opacity-10"></div>
          </div>

        </div>
      </div>

      {/* BOTTOM SECTION: GRAFIK MINGGUAN */}
      <div className="mt-4">
        <div className="card border-0 shadow-sm rounded-5 p-4 bg-white">
          <div className="d-flex justify-content-between align-items-center mb-5">
            <h6 className="fw-black fst-italic text-dark m-0 small letter-spacing-wide text-uppercase">Statistik Kepatuhan Mingguan</h6>
            <div className="d-flex gap-3 small fw-bold">
               <div className="d-flex align-items-center gap-2"><div className="rounded-circle bg-primary" style={{width:8, height:8}}></div> PATUH</div>
               <div className="d-flex align-items-center gap-2"><div className="rounded-circle bg-danger" style={{width:8, height:8}}></div> PELANGGARAN</div>
            </div>
          </div>
          
          <div className="d-flex justify-content-between align-items-end px-2" style={{ height: '120px' }}>
              {[1, 2, 3, 4, 5, 6, 7].map(day => (
                <div key={day} className="text-center w-100">
                    {/* Placeholder Batang Bar Sederhana */}
                    <div className="d-flex flex-column align-items-center gap-1 mb-2">
                       <div className="bg-primary rounded-top" style={{ width: '30%', height: `${Math.random() * 80 + 20}px`, opacity: 0.1 }}></div>
                    </div>
                    <span className="text-muted fw-bold small">Day {day}</span>
                </div>
              ))}
          </div>
        </div>
      </div>

    </div>
  );
};

// SUB KOMPONEN PROGRESS BAR
const ViolationBar = ({ label, value, color }) => (
  <div className="mb-4">
    <div className="d-flex justify-content-between align-items-center mb-1">
      <span className="small fw-black text-dark" style={{fontSize:'12px'}}>{label}</span>
      <span className="small fw-bold" style={{ color: color }}>{value}%</span>
    </div>
    <div className="progress" style={{ height: '6px', backgroundColor: '#f1f5f9' }}>
      <div 
        className="progress-bar rounded-pill" 
        style={{ width: `${value}%`, backgroundColor: color }}
      ></div>
    </div>
  </div>
);

export default APD;