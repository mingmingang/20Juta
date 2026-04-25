import React, { useState } from 'react';
import { ArrowLeft, Save, X, User, Briefcase, Tag } from 'lucide-react';
import '../style/Profile.css';

const ProfileView = ({ userData, onBack, onUpdateSuccess, notify }) => {
  const [isEditing, setIsEditing] = useState(false);
  
  // State untuk form edit
  const [formData, setFormData] = useState({
    fullName: userData?.fullName || '',
    division: userData?.division || '',
    employeeId: userData?.employeeId || ''
  });

  const handleSave = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/update-profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: userData.username,
          ...formData
        }),
      });

      const resData = await response.json();
      if (resData.success) {
        notify('success', 'Profil Berhasil Diperbarui!');
        setIsEditing(false);
        // Beri tahu App.js untuk update data terbaru di Header & Sidebar
        onUpdateSuccess({ ...userData, ...formData }); 
      } else {
        notify('error', 'Gagal memperbarui database');
      }
    } catch (err) {
      notify('error', 'Gagal menghubungi server');
    }
  };

  return (
    <div className="profile-container animate-fade-in">
      <div className="profile-card shadow-lg">
        
        {/* Banner Biru Atas */}
        <div className="profile-header-banner">
          <button className="back-btn" onClick={onBack}>
            <ArrowLeft size={24} color="white" />
          </button>
        </div>

        <div className="profile-body p-5">
          <div className="row align-items-end mb-5">
            <div className="col-auto">
              <div className="avatar-wrapper shadow">
                <img 
                  src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${userData?.username || 'user'}`}
                  alt="Avatar" 
                />
              </div>
            </div>

            <div className="col ms-4 pb-2">
              {!isEditing ? (
                /* MODE VIEWING */
                <>
                  <h1 className="user-fullname fw-black fst-italic">{formData.fullName}</h1>
                  <div className="row mt-3">
                    <div className="col-auto">
                      <p className="info-label">DIVISION</p>
                      <p className="info-value">{formData.division}</p>
                    </div>
                    <div className="col-auto ms-5">
                      <p className="info-label">ID KARYAWAN</p>
                      <p className="info-value">{formData.employeeId}</p>
                    </div>
                  </div>
                </>
              ) : (
                /* MODE EDITING (FORM) */
                <div className="row g-3">
                   <div className="col-12 mb-2">
                      <label className="small fw-bold text-primary mb-1">NAMA LENGKAP</label>
                      <input 
                        className="form-control rounded-3 border-primary-subtle"
                        value={formData.fullName}
                        onChange={(e) => setFormData({...formData, fullName: e.target.value})}
                      />
                   </div>
                   <div className="col-md-6">
                      <label className="small fw-bold text-primary mb-1">DIVISION</label>
                      <input 
                        className="form-control rounded-3 border-primary-subtle"
                        value={formData.division}
                        onChange={(e) => setFormData({...formData, division: e.target.value})}
                      />
                   </div>
                   <div className="col-md-6">
                      <label className="small fw-bold text-primary mb-1">ID KARYAWAN</label>
                      <input 
                        className="form-control rounded-3 border-primary-subtle"
                        value={formData.employeeId}
                        onChange={(e) => setFormData({...formData, employeeId: e.target.value})}
                      />
                   </div>
                </div>
              )}
            </div>

            <div className="col-auto pb-3">
              {!isEditing ? (
                <button className="btn-edit-profil shadow" onClick={() => setIsEditing(true)}>
                  EDIT PROFIL
                </button>
              ) : (
                <div className="d-flex gap-2">
                   <button className="btn btn-danger rounded-4 px-4 py-2 fw-bold" onClick={() => setIsEditing(false)}>
                      <X size={18} />
                   </button>
                   <button className="btn btn-success rounded-4 px-4 py-2 fw-bold shadow-sm" onClick={handleSave}>
                      <Save size={18} className="me-2" /> SIMPAN
                   </button>
                </div>
              )}
            </div>
          </div>

          <div className="row g-4 mt-2">
             <StatMiniCard label="TOTAL MONITORING" value={userData?.monitoringHours} />
             <StatMiniCard label="STATUS KEAMANAN" value={userData?.securityStatus} isStatus />
             <StatMiniCard label="AKSES SISTEM" value={userData?.accessLevel} />
          </div>
        </div>
      </div>
    </div>
  );
};

const StatMiniCard = ({ label, value, isStatus }) => (
  <div className="col-md-4">
    <div className="stat-mini-card p-4 rounded-4 border-0">
      <p className="mini-label">{label}</p>
      <h3 className={`mini-value ${isStatus ? 'text-success fst-italic' : ''}`}>
        {value}
      </h3>
    </div>
  </div>
);

export default ProfileView;