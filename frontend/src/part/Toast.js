import React, { useEffect } from 'react';
import { CheckCircle, AlertCircle, X } from 'lucide-react';

const Toast = ({ show, type, message, onClose }) => {
  // Notifikasi otomatis hilang setelah 3 detik
  useEffect(() => {
    if (show) {
      const timer = setTimeout(onClose, 3000);
      return () => clearTimeout(timer);
    }
  }, [show, onClose]);

  return (
    <div 
      className={`shadow-lg border-0 card p-3 rounded-4`}
      style={{
        position: 'fixed',
        top: '20px',
        right: show ? '20px' : '-400px', // Animasi Geser
        transition: 'all 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55)',
        zIndex: 9999,
        width: '320px',
        backgroundColor: type === 'success' ? '#dcfce7' : '#fee2e2',
        color: type === 'success' ? '#166534' : '#991b1b',
      }}
    >
      <div className="d-flex align-items-center justify-content-between">
        <div className="d-flex align-items-center gap-2">
          {type === 'success' ? <CheckCircle size={24} /> : <AlertCircle size={24} />}
          <strong className="small uppercase">{type === 'success' ? 'Berhasil' : 'Gagal'}</strong>
        </div>
        <X size={18} style={{ cursor: 'pointer', opacity: 0.5 }} onClick={onClose} />
      </div>
      <div className="mt-1 small fw-medium">{message}</div>
    </div>
  );
};

export default Toast;