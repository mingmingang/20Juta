import React, { useState, useEffect } from 'react';
import { ShieldCheck, Zap, Clock, AlertTriangle, Circle } from 'lucide-react';
import '../style/Dashboard.css';

const Dashboard = () => {
  const [currentTime, setCurrentTime] = useState(new Date());

  // Update Jam Real-Time
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="animate-fade-in">
      <div className="row g-4 mb-5">
        <StatCard title="KEPATUHAN APD" value="98.5%" trend="+0.5%" icon={<ShieldCheck color="#0d6efd"/>} />
        <StatCard title="TARGET / SHIFT" value="215 / 215" trend="On Target" icon={<Zap color="#6610f2"/>} />
        <StatCard title="EFISIENSI WAKTU" value="92.1%" trend="+2.1%" icon={<Clock color="#0dcaf0"/>} />
        <StatCard title="ANOMALI POSE" value="2" trend="-14%" icon={<AlertTriangle color="#dc3545"/>} />
      </div>

      <div className="row g-4">
        <div className="col-lg-6">
          <CameraFeed 
            title="CAMERA 01: WORK SEQUENCE AI" 
            url="http://127.0.0.1:5001/video_feed" 
          />
        </div>
        <div className="col-lg-6">
          <CameraFeed 
            title="CAMERA 02: PPE & POSE AI (KARYAWAN BERDIRI)" 
            url="http://127.0.0.1:5001/video_feed" 
          />
        </div>
      </div>

      <div className="mt-5 d-flex justify-content-between">
        <h6 className="fw-black text-dark text-uppercase fst-italic opacity-75">Audit Keamanan & APD Terbaru</h6>
        <span className="small text-muted fw-bold">KLIK UNTUK DETAIL BUKTI</span>
      </div>
    </div>
  );
};

// --- KOMPONEN KECIL DALAM DASHBOARD ---
const StatCard = ({ title, value, trend, icon }) => (
  <div className="col-md-3">
    <div className="card border-0 shadow-sm p-4 rounded-4 bg-white h-100 transition-hover">
      <div className="d-flex justify-content-between mb-4">
        <p className="text-muted fw-bold small m-0 letter-spacing-wide">{title}</p>
        <div className="p-2 rounded-3 bg-light">{icon}</div>
      </div>
      <div className="d-flex align-items-end gap-2">
        <h2 className="fw-black m-0 p-0" style={{ fontSize: '32px' }}>{value}</h2>
        <span className={`small fw-bold ${trend.includes('+') || trend === 'On Target' ? 'text-success' : 'text-danger'}`}>{trend}</span>
      </div>
    </div>
  </div>
);

const CameraFeed = ({ title, url }) => (
  <div className="card border-0 shadow-sm rounded-4 overflow-hidden bg-white">
    <div className="bg-white p-3 border-bottom d-flex align-items-center gap-2">
      <Circle size={8} fill="#198754" color="#198754" />
      <span className="fw-bold small">{title}</span>
    </div>
    <div className="bg-dark" style={{ height: '350px', position: 'relative' }}>
      <img src={url} alt="Streaming Feed" className="w-159 h-100 object-fit-cover opacity-75" />
      <div className="position-absolute bottom-0 start-0 p-3 w-100 d-flex gap-2">
        <div className="bg-dark bg-opacity-75 backdrop-blur border border-secondary border-opacity-25 text-white px-2 py-1 rounded small">
           <Zap size={10} className="me-1 text-primary"/> YOLOv11s • MediaPipe Pose
        </div>
      </div>
    </div>
  </div>
);

export default Dashboard;