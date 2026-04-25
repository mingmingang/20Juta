import React, { useState, useEffect } from "react";
// Tambahkan Eye, Edit3, Trash2 untuk icon aksi
import {
  Search,
  Filter,
  Plus,
  MoreVertical,
  ArrowLeft,
  X,
  Save,
  Eye,
  Edit3,
  Trash2,
} from "lucide-react";
import "../../style/UserManagement.css";

const UserManagement = ({ onBack, notify }) => {
  const [users, setUsers] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [showAddModal, setShowAddModal] = useState(false);

  // --- STATE TAMBAHAN UNTUK AKSI ---
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [selectedUser, setSelectedUser] = useState(null); // User yang sedang di-klik
  const [menuOpenId, setMenuOpenId] = useState(null); // Untuk handle dropdown mana yang buka

  const [newUser, setNewUser] = useState({
    fullName: "",
    username: "",
    password: "",
    employeeId: "",
    division: "",
    accessLevel: "STAFF",
  });

  const fetchUsers = async () => {
    const res = await fetch("http://localhost:5001/api/users");
    const data = await res.json();
    setUsers(data);
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  useEffect(() => {
    const handleOutsideClick = (event) => {
      // Jika yang diklik bukan elemen dengan class "btn-action-wrapper" atau isinya
      if (!event.target.closest(".btn-action-wrapper")) {
        setMenuOpenId(null);
      }
    };

    if (menuOpenId !== null) {
      // Daftarkan event listener hanya ketika menu terbuka
      document.addEventListener("mousedown", handleOutsideClick);
    }

    return () => {
      // Bersihkan event listener saat komponen ditutup atau menu hilang
      document.removeEventListener("mousedown", handleOutsideClick);
    };
  }, [menuOpenId]); // Akan aktif setiap kali menuOpenId berubah

  const filteredUsers = users.filter(
    (user) =>
      user.fullName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.employeeId.toLowerCase().includes(searchTerm.toLowerCase()),
  );

  const handleSave = async (e) => {
    e.preventDefault();
    const res = await fetch("http://localhost:5001/api/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(newUser),
    });
    if (res.ok) {
      notify("success", "User berhasil ditambahkan");
      setShowAddModal(false);
      fetchUsers();
    }
  };

  // --- FUNGSI UPDATE DATA (EDIT) ---
  const handleUpdate = async (e) => {
    e.preventDefault();
    // Simulasikan panggil API update
    const res = await fetch("http://localhost:5001/api/users/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(selectedUser),
    });
    if (res.ok) {
      notify("success", "Perubahan data personil disimpan");
      setShowEditModal(false);
      fetchUsers();
    }
  };

  return (
    <div className="user-mgmt-container animate-fade-in">
      <div className="d-flex align-items-center gap-3 mb-4">
        <button className="back-circle-btn" onClick={onBack}>
          <ArrowLeft size={20} />
        </button>
        <div>
          <h2 className="fw-black fst-italic m-0">MANAJEMEN PERSONIL</h2>
          <p className="text-muted small fw-bold">
            Audit hak akses dan status operator TMMIN
          </p>
        </div>
      </div>

      <div className="card border-0 shadow-sm rounded-5 overflow-hidden p-4 bg-white">
        <div className="d-flex justify-content-between mb-4 gap-3">
          <div className="search-box-wrapper flex-grow-1">
            <Search className="search-icon" size={18} />
            <input
              className="search-input"
              placeholder="Cari Nama atau ID Operator..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <button className="btn btn-light rounded-4 px-4 fw-bold text-muted border d-flex align-items-center gap-2">
            <Filter size={18} /> FILTER
          </button>
          <button
            className="btn btn-primary rounded-4 px-4 fw-bold d-flex align-items-center gap-2"
            onClick={() => setShowAddModal(true)}
          >
            <Plus size={20} /> TAMBAH USER
          </button>
        </div>

        <div className="table-responsive">
          <table className="table table-hover align-middle custom-table">
            <thead>
              <tr>
                <th>NAMA LENGKAP</th>
                <th>ID TMMIN</th>
                <th>JABATAN</th>
                <th>LEVEL AKSES</th>
                <th>STATUS</th>
                <th className="text-center">AKSI</th>
              </tr>
            </thead>
            <tbody>
              {filteredUsers.map((user) => (
                <tr key={user.id}>
                  <td>
                    <div className="d-flex align-items-center gap-3">
                      <img
                        src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${user.username}`}
                        alt="avatar"
                        className="avatar-sm"
                      />
                      <span className="fw-black text-dark">
                        {user.fullName.toUpperCase()}
                      </span>
                    </div>
                  </td>
                  <td className="text-muted fw-bold">{user.employeeId}</td>
                  <td className="text-muted fw-bold">
                    {user.division.toUpperCase()}
                  </td>
                  <td>
                    <span
                      className={`badge-access ${user.accessLevel?.replace(" ", "-").toLowerCase()}`}
                    >
                      {user.accessLevel?.toUpperCase()}
                    </span>
                  </td>
                  <td>
                    <div className="d-flex align-items-center gap-2">
                      <div
                        className={`status-dot ${user.status?.toLowerCase() === "excellent" || user.status?.toLowerCase() === "online" ? "bg-success" : "bg-secondary"}`}
                      ></div>
                      <span
                        className={`fw-black fst-italic ${user.status?.toLowerCase() === "excellent" || user.status?.toLowerCase() === "online" ? "text-success" : "text-muted"}`}
                      >
                        {user.status || "OFFLINE"}
                      </span>
                    </div>
                  </td>

                  {/* UPDATE BAGIAN AKSI MENJADI DINAMIS */}
                  <td className="text-center position-relative">
                    {/* BUNGKUS DENGAN DIV CLASS "btn-action-wrapper" */}
                    <div className="btn-action-wrapper d-inline-block">
                      <button
                        className="btn-action-trigger"
                        onClick={() =>
                          setMenuOpenId(menuOpenId === user.id ? null : user.id)
                        }
                      >
                        <MoreVertical size={20} />
                      </button>

                      {menuOpenId === user.id && (
                        <div className="action-dropdown shadow-lg">
                          <button
                            onClick={() => {
                              setSelectedUser(user);
                              setShowDetailModal(true);
                              setMenuOpenId(null);
                            }}
                          >
                            <Eye size={16} /> <span>Detail</span>
                          </button>
                          <button
                            onClick={() => {
                              setSelectedUser(user);
                              setShowEditModal(true);
                              setMenuOpenId(null);
                            }}
                          >
                            <Edit3 size={16} /> <span>Edit</span>
                          </button>
                          <button
                            className="text-danger"
                            style={{ borderTop: "1px solid #f1f5f9" }}
                          >
                            <Trash2 size={16} /> <span>Delete</span>
                          </button>
                        </div>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ----------------- MODAL TAMBAH USER (ASLI KAMU) ----------------- */}
      {showAddModal && (
        <div className="modal-overlay">
          <form
            className="modal-card p-4 animate-slide-up"
            onSubmit={handleSave}
          >
            <div className="d-flex justify-content-between mb-4">
              <h4 className="fw-black fst-italic">TAMBAH PERSONIL</h4>
              <X
                onClick={() => setShowAddModal(false)}
                className="cursor-pointer"
              />
            </div>
            <div className="row g-3">
              <div className="col-12">
                <input
                  required
                  placeholder="Nama Lengkap"
                  className="form-control"
                  onChange={(e) =>
                    setNewUser({ ...newUser, fullName: e.target.value })
                  }
                />
              </div>
              <div className="col-md-6">
                <input
                  required
                  placeholder="Username"
                  className="form-control"
                  onChange={(e) =>
                    setNewUser({ ...newUser, username: e.target.value })
                  }
                />
              </div>
              <div className="col-md-6">
                <input
                  required
                  type="password"
                  placeholder="Password"
                  className="form-control"
                  onChange={(e) =>
                    setNewUser({ ...newUser, password: e.target.value })
                  }
                />
              </div>
              <div className="col-md-6">
                <input
                  required
                  placeholder="ID Karyawan"
                  className="form-control"
                  onChange={(e) =>
                    setNewUser({ ...newUser, employeeId: e.target.value })
                  }
                />
              </div>
              <div className="col-md-6">
                <input
                  required
                  placeholder="Jabatan"
                  className="form-control"
                  onChange={(e) =>
                    setNewUser({ ...newUser, division: e.target.value })
                  }
                />
              </div>
            </div>
            <button
              type="submit"
              className="btn btn-primary w-100 mt-4 rounded-4 fw-bold p-3"
            >
              SIMPAN USER
            </button>
          </form>
        </div>
      )}

      {/* ----------------- MODAL DETAIL (PERUBAHAN BARU) ----------------- */}
      {showDetailModal && selectedUser && (
        <div className="modal-overlay">
          <div className="modal-card detail p-5 shadow-2xl">
            <div className="d-flex justify-content-end">
              <X
                onClick={() => setShowDetailModal(false)}
                className="cursor-pointer text-muted"
              />
            </div>
            <div className="text-center">
              <img
                src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${selectedUser.username}`}
                className="avatar-lg mb-4"
                alt="avatar"
              />
              <h2 className="fw-black fst-italic mb-0">
                {selectedUser.fullName.toUpperCase()}
              </h2>
              <p className="text-primary fw-bold tracking-widest">
                {selectedUser.employeeId}
              </p>

              <div className="row mt-5 text-start">
                <div className="col-6 mb-3">
                  <label className="mini-label">JABATAN</label>
                  <p className="fw-bold">{selectedUser.division}</p>
                </div>
                <div className="col-6 mb-3">
                  <label className="mini-label">AKSES</label>
                  <p className="fw-bold">{selectedUser.accessLevel}</p>
                </div>
                <div className="col-12">
                  <label className="mini-label">STATUS</label>
                  <p className="text-success fw-black fst-italic">
                    ● {selectedUser.status}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ----------------- MODAL EDIT (PERUBAHAN BARU) ----------------- */}
      {showEditModal && selectedUser && (
        <div className="modal-overlay">
          <form
            className="modal-card p-4 animate-slide-up"
            onSubmit={handleUpdate}
          >
            <div className="d-flex justify-content-between mb-4 border-bottom pb-3">
              <h4 className="fw-black fst-italic m-0">EDIT DATA PERSONIL</h4>
              <X
                onClick={() => setShowEditModal(false)}
                className="cursor-pointer"
              />
            </div>
            <div className="row g-3">
              <div className="col-12">
                <label className="label-edit">NAMA LENGKAP</label>
                <input
                  className="form-control"
                  value={selectedUser.fullName}
                  onChange={(e) =>
                    setSelectedUser({
                      ...selectedUser,
                      fullName: e.target.value,
                    })
                  }
                />
              </div>
              <div className="col-md-6">
                <label className="label-edit">ID TMMIN</label>
                <input
                  className="form-control"
                  value={selectedUser.employeeId}
                  onChange={(e) =>
                    setSelectedUser({
                      ...selectedUser,
                      employeeId: e.target.value,
                    })
                  }
                />
              </div>
              <div className="col-md-6">
                <label className="label-edit">JABATAN</label>
                <input
                  className="form-control"
                  value={selectedUser.division}
                  onChange={(e) =>
                    setSelectedUser({
                      ...selectedUser,
                      division: e.target.value,
                    })
                  }
                />
              </div>
              <div className="col-md-6">
                <label className="label-edit">LEVEL AKSES</label>
                <select
                  className="form-control"
                  value={selectedUser.accessLevel}
                  onChange={(e) =>
                    setSelectedUser({
                      ...selectedUser,
                      accessLevel: e.target.value,
                    })
                  }
                >
                  <option value="SUPER ADMIN">SUPER ADMIN</option>
                  <option value="ADMIN">ADMIN</option>
                  <option value="STAFF">STAFF</option>
                  <option value="OPERATOR">OPERATOR</option>
                </select>
              </div>
              <div className="col-md-6">
                <label className="label-edit">STATUS</label>
                <input
                  className="form-control"
                  value={selectedUser.status}
                  onChange={(e) =>
                    setSelectedUser({ ...selectedUser, status: e.target.value })
                  }
                />
              </div>
            </div>
            <button
              type="submit"
              className="btn btn-primary w-100 mt-4 rounded-4 fw-bold p-3"
            >
              SIMPAN PERUBAHAN
            </button>
          </form>
        </div>
      )}
    </div>
  );
};

export default UserManagement;
