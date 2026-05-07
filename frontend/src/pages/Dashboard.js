import React, { useState, useEffect } from 'react';
import { ShieldCheck, Zap, Clock, AlertTriangle, Circle, CheckCircle2 } from 'lucide-react';
import '../style/Dashboard.css';

const Dashboard = () => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [steps, setSteps] = useState([]);
  const [incidents, setIncidents] = useState([]);

  // Jam Real-Time
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Ambil Data WI & Insiden
  useEffect(() => {
    fetch('http://localhost:5001/api/work-sequence').then(r => r.json()).then(setSteps);
    fetch('http://localhost:5001/api/incidents').then(r => r.json()).then(setIncidents);
  }, []);

  return (
    <div className="animate-fade-in px-2 pb-5">

      {/* --- SECTION 1: STATISTIC CARDS (Tetap di Atas) --- */}
      {/* --- SECTION 1: 3 NEW STATISTIC CARDS --- */}
      <div className="row g-4 mb-4">
        {/* Card 1: Total Assembly */}
        <div className="col-md-4">
          <StatCard
            title="TOTAL ASSEMBLY"
            value="1,240"
            trend="+12"
            icon={<Zap color="#3b82f6" size={24} />}
          />
        </div>

        {/* Card 2: Akurasi Pemasangan */}
        <div className="col-md-4">
          <StatCard
            title="AKURASI PEMASANGAN"
            value="96.8%"
            trend="-0.2%"
            icon={<ShieldCheck color="#f43f5e" size={24} />}
            isRose={true}
          />
        </div>

        {/* Card 3: Waktu Rata-rata */}
        <div className="col-md-4">
          <StatCard
            title="WAKTU RATA-RATA"
            value="45s"
            trend="-4s"
            icon={<Clock color="#2563eb" size={24} />}
          />
        </div>
      </div>

      {/* --- SECTION 2: DUAL CAMERA (Tengah) --- */}
      <div className="row g-4 mb-4">
        <div className="col-lg-6">
          <CameraFeed
            title="CAMERA 01: BEARING DETECTION"
            url="http://127.0.0.1:5001/checkbearing_feed"
            tech="YOLOv11s • HP-IP Camera"
          />
        </div>
        <div className="col-lg-6">
          <CameraFeed
            title="CAMERA 02: OPERATOR MONITORING"
            url="http://127.0.0.1:5001/picking_feed"
            tech="YOLOv8n • MediaPipe Pose"
          />
        </div>
      </div>

      {/* --- SECTION 3: WORK INSTRUCTION & INCIDENTS (SEKARANG DI BAWAH) --- */}
      <div className="row g-4">

        {/* BOX WORK INSTRUCTION (BAWAH KIRI) */}
        <div className="col-lg-7">
          <div className="card border-0 shadow-sm rounded-5 p-4 bg-white border border-light h-100">
            <h6 className="fw-black fst-italic text-dark mb-4 text-uppercase small letter-spacing-wide">Work Instruction</h6>
            <div className="row">
              {steps.map((step, index) => (
                <div key={index} className="col-md-4">
                  <div className={`step-item mb-2 p-3 rounded-4 border-0 shadow-sm h-100 ${step.status}`}>
                    <div className="d-flex justify-content-between align-items-center mb-1">
                      <small className="fw-black opacity-50 italic">STEP {index + 1}</small>
                      {step.status === 'done' && <CheckCircle2 size={16} className="text-success" />}
                    </div>
                    <p className="m-0 fw-black small">{step.name}</p>
                    {step.status === 'active' && <p className="m-0 mt-1 opacity-75" style={{ fontSize: '10px' }}>{step.desc}</p>}
                  </div>
                </div>
              ))}
            </div>
            <button className="btn btn-primary w-100 rounded-4 p-3 mt-4 fw-black text-uppercase small">
              KONFIRMASI & LANJUT KE LANGKAH BERIKUTNYA
            </button>
          </div>
        </div>

        {/* BOX LAPORAN INSIDEN (BAWAH KANAN) */}
        <div className="col-lg-5">
          <div className="card border-0 shadow-sm rounded-5 p-4 bg-white border border-light h-100">
            <div className="d-flex justify-content-between align-items-center mb-4">
              <h6 className="fw-black fst-italic text-dark m-0 text-uppercase small">Laporan Insiden Real-time</h6>
              <span className="badge bg-danger rounded-pill px-3 py-1" style={{ fontSize: '9px' }}>{incidents.length} INSIDEN</span>
            </div>
            <div className="incident-scroll pe-2" style={{ maxHeight: '250px', overflowY: 'auto' }}>
              {incidents.map((item, i) => (
                <div key={i} className="incident-card p-3 border rounded-4 mb-3 bg-white shadow-sm border-light">
                  <div className="d-flex justify-content-between mb-2">
                    <span className="fw-black italic" style={{ color: item.color, fontSize: '11px' }}>{item.type}</span>
                    <span className="text-muted" style={{ fontSize: '10px' }}>{item.time}</span>
                  </div>
                  <h6 className="fw-black small mb-1">{item.title}</h6>
                  <p className="text-muted m-0 small" style={{ lineHeight: '1.4' }}>{item.msg}</p>
                </div>
              ))}
            </div>
            <button className="btn btn-outline-dark w-100 border-light-subtle rounded-4 p-2 fw-black text-uppercase small mt-2">
              BUAT LAPORAN MANUAL
            </button>
          </div>
        </div>

      </div>
    </div>
  );
};

// --- SUB-KOMPONEN TETAP SAMA ---
const StatCard = ({ title, value, trend, icon }) => (
  <div className="card border-0 shadow-sm p-4 rounded-5 bg-white h-100 transition-hover border border-light">
    <div className="d-flex justify-content-between mb-4">
      <p className="text-muted fw-bold small m-0 uppercase" style={{ fontSize: '10px', letterSpacing: '1px' }}>{title}</p>
      <div className="p-2 rounded-4 bg-light shadow-sm">{icon}</div>
    </div>
    <div className="d-flex align-items-end gap-2">
      <h2 className="fw-black m-0 p-0" style={{ fontSize: '28px' }}>{value}</h2>
      <span className={`small fw-bold pb-1 ${trend.includes('+') || trend === 'On Target' ? 'text-success' : 'text-danger'}`}>{trend}</span>
    </div>
  </div>
);

const CameraFeed = ({ title, url, tech }) => (
  <div className="card border-0 shadow-sm rounded-5 overflow-hidden bg-white border border-light">
    <div className="bg-white p-3 border-bottom d-flex align-items-center justify-content-between px-4">
      <div className="d-flex align-items-center gap-2">
        <Circle size={10} fill="#198754" color="#198754" />
        <span className="fw-bold small">{title}</span>
      </div>
      <span className="badge bg-danger rounded-pill px-2" style={{ fontSize: '9px', opacity: 0.8 }}>LIVE STREAM</span>
    </div>
    <div className="bg-dark" style={{ height: '420px', position: 'relative' }}>
      <img src={url} alt="AI Feed" className="w-100 h-100 opacity-75" />
      <div className="position-absolute bottom-0 start-0 p-3 w-100">
        <div className="bg-dark bg-opacity-75 backdrop-blur border border-secondary border-opacity-25 text-white px-3 py-2 rounded-4 small d-inline-block">
          <Zap size={12} className="me-2 text-primary" fill="currentColor" /> {tech}
        </div>
      </div>
    </div>
  </div>
);

export default Dashboard;