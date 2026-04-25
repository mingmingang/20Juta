import React from 'react';
import { LogOut, X } from 'lucide-react';
import '../style/Logout.css';

const LogoutModal = ({ show, onCancel, onConfirm }) => {
  if (!show) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-box animate-pop">
        <button className="close-x" onClick={onCancel}><X size={20} /></button>
        
        <div className="modal-icon-container">
          <div className="modal-icon-bg">
            <LogOut size={32} color="#f43f5e" />
          </div>
        </div>

        <h3 className="modal-title">Konfirmasi Keluar</h3>
        <p className="modal-desc">Anda yakin ingin keluar dari sistem <strong>SOPGuard AI</strong>?</p>

        <div className="modal-actions">
          <button className="btn-cancel" onClick={onCancel}>Batal</button>
          <button className="btn-confirm-logout" onClick={onConfirm}>Ya, Keluar</button>
        </div>
      </div>
    </div>
  );
};

export default LogoutModal;