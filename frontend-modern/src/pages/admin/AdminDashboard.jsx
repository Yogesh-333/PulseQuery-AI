import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { Users, FileText, Upload, Plus, Trash2, Shield, X, AlertTriangle, Search } from 'lucide-react';

const AdminDashboard = () => {
    const { user: currentUser } = useAuth();
    const [activeTab, setActiveTab] = useState('users');
    const [users, setUsers] = useState([]);
    const [patients, setPatients] = useState([]);
    const [selectedPatient, setSelectedPatient] = useState('');
    const [file, setFile] = useState(null);
    const [uploadStatus, setUploadStatus] = useState('');

    const [isAddUserModalOpen, setIsAddUserModalOpen] = useState(false);
    const [newUser, setNewUser] = useState({
        username: '',
        password: '',
        name: '',
        role: 'doctor',
        department: '',
        email: '',
        phone: '',
        address: ''
    });

    const [isEditUserModalOpen, setIsEditUserModalOpen] = useState(false);
    const [editingUser, setEditingUser] = useState({
        username: '',
        name: '',
        role: '',
        department: '',
        password: '',
        email: '',
        phone: '',
        address: ''
    });

    const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
    const [userToDelete, setUserToDelete] = useState(null);

    const [searchTerm, setSearchTerm] = useState('');
    const [roleFilter, setRoleFilter] = useState('all');

    const filteredUsers = users.filter(user => {
        const matchesSearch = (
            user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
            user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            (user.email && user.email.toLowerCase().includes(searchTerm.toLowerCase()))
        );
        const matchesRole = roleFilter === 'all' || user.role === roleFilter;
        return matchesSearch && matchesRole;
    });

    useEffect(() => {
        fetchUsers();
        fetchPatients();
    }, []);

    const fetchUsers = async () => {
        try {
            const response = await api.get('/api/admin/users');
            setUsers(response.data);
        } catch (error) {
            console.error("Failed to fetch users", error);
        }
    };

    const handleCreateUser = async (e) => {
        e.preventDefault();
        try {
            await api.post('/api/admin/users', newUser);
            setIsAddUserModalOpen(false);
            setNewUser({
                username: '',
                password: '',
                name: '',
                role: 'doctor',
                department: '',
                email: '',
                phone: '',
                address: ''
            });
            fetchUsers();
        } catch (error) {
            console.error("Failed to create user", error);
            alert("Failed to create user: " + (error.response?.data?.error || error.message));
        }
    };

    const confirmDeleteUser = (user) => {
        setUserToDelete(user);
        setIsDeleteModalOpen(true);
    };

    const handleDeleteUser = async () => {
        if (!userToDelete) return;
        try {
            await api.delete(`/api/admin/users/${userToDelete.username}`);
            setIsDeleteModalOpen(false);
            setUserToDelete(null);
            fetchUsers();
        } catch (error) {
            console.error("Failed to delete user", error);
            alert("Failed to delete user: " + (error.response?.data?.error || error.message));
        }
    };

    const openEditModal = (user) => {
        setEditingUser({ ...user, password: '' }); // Clear password field
        setIsEditUserModalOpen(true);
    };

    const handleUpdateUser = async (e) => {
        e.preventDefault();
        try {
            await api.put(`/api/admin/users/${editingUser.username}`, editingUser);
            setIsEditUserModalOpen(false);
            fetchUsers();
        } catch (error) {
            console.error("Failed to update user", error);
            alert("Failed to update user: " + (error.response?.data?.error || error.message));
        }
    };

    const fetchPatients = async () => {
        try {
            const response = await api.get('/api/patients');
            setPatients(response.data);
        } catch (error) {
            console.error("Failed to fetch patients", error);
        }
    };

    const handleUpload = async (e) => {
        e.preventDefault();
        if (!file || !selectedPatient) return;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('patient_id', selectedPatient);

        setUploadStatus('uploading');
        try {
            await api.post('/api/rag/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            setUploadStatus('success');
            setTimeout(() => setUploadStatus(''), 3000);
        } catch (error) {
            setUploadStatus('error');
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-white">Admin Console</h1>
                    <p className="text-slate-400">System management and configuration</p>
                </div>
            </div>

            <div className="flex gap-4 border-b border-white/10">
                <button
                    onClick={() => setActiveTab('users')}
                    className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 ${activeTab === 'users' ? 'border-primary-500 text-white' : 'border-transparent text-slate-400 hover:text-white'
                        }`}
                >
                    User Management
                </button>
                <button
                    onClick={() => setActiveTab('documents')}
                    className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 ${activeTab === 'documents' ? 'border-primary-500 text-white' : 'border-transparent text-slate-400 hover:text-white'
                        }`}
                >
                    Patient Documents
                </button>
            </div>

            {activeTab === 'users' && (
                <div className="glass-panel rounded-2xl overflow-hidden">
                    <div className="p-4 border-b border-white/5 bg-surfaceHighlight/30 flex flex-col gap-4">
                        <div className="flex items-center justify-between">
                            <h2 className="font-bold text-white flex items-center gap-2">
                                <Users className="w-5 h-5 text-primary-400" />
                                System Users
                            </h2>
                            <button
                                onClick={() => setIsAddUserModalOpen(true)}
                                className="px-3 py-1.5 bg-primary-600 hover:bg-primary-500 text-white text-xs font-medium rounded-lg flex items-center gap-1"
                            >
                                <Plus className="w-3 h-3" /> Add User
                            </button>
                        </div>
                        <div className="flex items-center gap-3">
                            <div className="relative flex-1 max-w-md">
                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                                <input
                                    type="text"
                                    placeholder="Search by name, username, or email..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                    className="w-full bg-surface border border-white/10 rounded-lg pl-9 pr-4 py-2 text-sm text-white focus:outline-none focus:border-primary-500"
                                />
                            </div>
                            <select
                                value={roleFilter}
                                onChange={(e) => setRoleFilter(e.target.value)}
                                className="bg-surface border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500"
                            >
                                <option value="all">All Roles</option>
                                <option value="admin">Admin</option>
                                <option value="doctor">Doctor</option>
                                <option value="lab">Lab Tech</option>
                                <option value="patient">Patient</option>
                            </select>
                        </div>
                    </div>
                    <table className="w-full text-left text-sm text-slate-400">
                        <thead className="bg-surfaceHighlight/50 text-xs uppercase font-medium text-slate-300">
                            <tr>
                                <th className="px-6 py-4">Username</th>
                                <th className="px-6 py-4">Name</th>
                                <th className="px-6 py-4">Role</th>
                                <th className="px-6 py-4">Status</th>
                                <th className="px-6 py-4">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                            {filteredUsers.length === 0 ? (
                                <tr>
                                    <td colSpan="5" className="px-6 py-8 text-center text-slate-500">
                                        No users found matching your filters.
                                    </td>
                                </tr>
                            ) : (
                                filteredUsers.map((user) => (
                                    <tr key={user.id} className="hover:bg-white/5 transition-colors">
                                        <td className="px-6 py-4 font-medium text-white">{user.username}</td>
                                        <td className="px-6 py-4">{user.name}</td>
                                        <td className="px-6 py-4">
                                            <span className={`px-2 py-1 rounded-md text-xs font-medium ${user.role === 'admin' ? 'bg-purple-500/20 text-purple-400' :
                                                user.role === 'doctor' ? 'bg-blue-500/20 text-blue-400' :
                                                    user.role === 'lab' ? 'bg-amber-500/20 text-amber-400' :
                                                        'bg-emerald-500/20 text-emerald-400'
                                                }`}>
                                                {user.role.toUpperCase()}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-emerald-400">Active</td>
                                        <td className="px-6 py-4 flex gap-2">
                                            <button
                                                onClick={() => openEditModal(user)}
                                                className="text-slate-500 hover:text-primary-400 transition-colors"
                                                title="Edit User"
                                            >
                                                <FileText className="w-4 h-4" />
                                            </button>
                                            {currentUser?.user_id !== user.username && (
                                                <button
                                                    onClick={() => confirmDeleteUser(user)}
                                                    className="text-slate-500 hover:text-red-400 transition-colors"
                                                    title="Delete User"
                                                >
                                                    <Trash2 className="w-4 h-4" />
                                                </button>
                                            )}
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            )}

            {activeTab === 'documents' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="glass-panel rounded-2xl p-6">
                        <h2 className="font-bold text-white mb-4 flex items-center gap-2">
                            <Upload className="w-5 h-5 text-primary-400" />
                            Upload Patient Records
                        </h2>
                        <p className="text-sm text-slate-400 mb-6">
                            Upload medical documents (PDF, DOCX, TXT) to the RAG system. These will be indexed specifically for the selected patient.
                        </p>

                        <form onSubmit={handleUpload} className="space-y-4">
                            <div>
                                <label className="block text-xs font-medium text-slate-400 mb-1.5">Select Patient</label>
                                <select
                                    value={selectedPatient}
                                    onChange={(e) => setSelectedPatient(e.target.value)}
                                    className="w-full bg-surface border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-primary-500/50"
                                >
                                    <option value="">-- Select Patient --</option>
                                    {patients.map(p => (
                                        <option key={p.id} value={p.id}>{p.name} (ID: {p.id})</option>
                                    ))}
                                </select>
                            </div>

                            <div className="border-2 border-dashed border-white/10 rounded-xl p-8 text-center hover:border-primary-500/50 transition-colors cursor-pointer relative">
                                <input
                                    type="file"
                                    onChange={(e) => setFile(e.target.files[0])}
                                    className="absolute inset-0 opacity-0 cursor-pointer"
                                />
                                <FileText className="w-10 h-10 text-slate-500 mx-auto mb-2" />
                                <p className="text-sm text-slate-300 font-medium">
                                    {file ? file.name : 'Click to upload or drag and drop'}
                                </p>
                                <p className="text-xs text-slate-500 mt-1">PDF, DOCX, TXT up to 10MB</p>
                            </div>

                            <button
                                type="submit"
                                disabled={!file || !selectedPatient || uploadStatus === 'uploading'}
                                className="w-full py-2.5 bg-primary-600 hover:bg-primary-500 text-white font-medium rounded-xl transition-colors disabled:opacity-50"
                            >
                                {uploadStatus === 'uploading' ? 'Indexing...' : 'Upload & Index'}
                            </button>

                            {uploadStatus === 'success' && (
                                <p className="text-sm text-emerald-400 text-center">Document successfully indexed!</p>
                            )}
                            {uploadStatus === 'error' && (
                                <p className="text-sm text-red-400 text-center">Upload failed. Please try again.</p>
                            )}
                        </form>
                    </div>

                    <div className="glass-panel rounded-2xl p-6 flex flex-col items-center justify-center text-center">
                        <Shield className="w-16 h-16 text-slate-600 mb-4" />
                        <h3 className="text-lg font-bold text-white mb-2">Secure Indexing</h3>
                        <p className="text-sm text-slate-400 max-w-xs">
                            Documents are encrypted and indexed with strict patient isolation. Only authorized doctors can access this data during consultation.
                        </p>
                    </div>
                </div>
            )}

            {/* Add User Modal */}
            {isAddUserModalOpen && (
                <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <div className="bg-surface border border-white/10 rounded-2xl p-6 w-full max-w-md shadow-2xl">
                        <div className="flex justify-between items-center mb-6">
                            <h3 className="text-lg font-bold text-white">Add New User</h3>
                            <button
                                onClick={() => setIsAddUserModalOpen(false)}
                                className="text-slate-400 hover:text-white transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>
                        <form onSubmit={handleCreateUser} className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-xs font-medium text-slate-400 mb-1">Username</label>
                                    <input
                                        type="text"
                                        value={newUser.username}
                                        onChange={(e) => setNewUser({ ...newUser, username: e.target.value })}
                                        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500"
                                        placeholder="jdoe"
                                    />
                                </div>
                                <div>
                                    <label className="block text-xs font-medium text-slate-400 mb-1">Full Name</label>
                                    <input
                                        type="text"
                                        value={newUser.name}
                                        onChange={(e) => setNewUser({ ...newUser, name: e.target.value })}
                                        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500"
                                        placeholder="John Doe"
                                    />
                                </div>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-xs font-medium text-slate-400 mb-1">Role</label>
                                    <select
                                        value={newUser.role}
                                        onChange={(e) => setNewUser({ ...newUser, role: e.target.value })}
                                        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500"
                                    >
                                        <option value="doctor">Doctor</option>
                                        <option value="lab">Lab Technician</option>
                                        <option value="patient">Patient</option>
                                        <option value="admin">Administrator</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-xs font-medium text-slate-400 mb-1">Department</label>
                                    <input
                                        type="text"
                                        value={newUser.department}
                                        onChange={(e) => setNewUser({ ...newUser, department: e.target.value })}
                                        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500"
                                        placeholder="Cardiology"
                                    />
                                </div>
                            </div>

                            {/* Optional Fields */}
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-xs font-medium text-slate-400 mb-1">Email (Optional)</label>
                                    <input
                                        type="email"
                                        value={newUser.email || ''}
                                        onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
                                        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500"
                                        placeholder="john@example.com"
                                    />
                                </div>
                                <div>
                                    <label className="block text-xs font-medium text-slate-400 mb-1">Phone (Optional)</label>
                                    <input
                                        type="tel"
                                        value={newUser.phone || ''}
                                        onChange={(e) => setNewUser({ ...newUser, phone: e.target.value })}
                                        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500"
                                        placeholder="+1 555-0123"
                                    />
                                </div>
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-slate-400 mb-1">Address (Optional)</label>
                                <input
                                    type="text"
                                    value={newUser.address || ''}
                                    onChange={(e) => setNewUser({ ...newUser, address: e.target.value })}
                                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500"
                                    placeholder="123 Medical Center Dr"
                                />
                            </div>

                            <div>
                                <label className="block text-xs font-medium text-slate-400 mb-1">Password</label>
                                <input
                                    type="password"
                                    value={newUser.password}
                                    onChange={(e) => setNewUser({ ...newUser, password: e.target.value })}
                                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500"
                                    placeholder="••••••••"
                                />
                            </div>
                            <button
                                type="submit"
                                className="w-full py-2.5 bg-primary-600 hover:bg-primary-500 text-white font-medium rounded-xl transition-colors mt-2"
                            >
                                Create User
                            </button>
                        </form>
                    </div>
                </div>
            )}

            {/* Edit User Modal */}
            {isEditUserModalOpen && (
                <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <div className="bg-surface border border-white/10 rounded-2xl p-6 w-full max-w-md shadow-2xl">
                        <div className="flex justify-between items-center mb-6">
                            <h3 className="text-lg font-bold text-white">Edit User</h3>
                            <button
                                onClick={() => setIsEditUserModalOpen(false)}
                                className="text-slate-400 hover:text-white transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>
                        <form onSubmit={handleUpdateUser} className="space-y-4">
                            <div>
                                <label className="block text-xs font-medium text-slate-400 mb-1.5">Name</label>
                                <input
                                    type="text"
                                    required
                                    value={editingUser.name}
                                    onChange={(e) => setEditingUser({ ...editingUser, name: e.target.value })}
                                    className="w-full bg-surfaceHighlight border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-primary-500/50"
                                    placeholder="Full Name"
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-slate-400 mb-1.5">Username</label>
                                <input
                                    type="text"
                                    disabled
                                    value={editingUser.username}
                                    className="w-full bg-surfaceHighlight/50 border border-white/5 rounded-xl px-4 py-2.5 text-sm text-slate-400 cursor-not-allowed"
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-slate-400 mb-1.5">New Password (leave blank to keep current)</label>
                                <input
                                    type="password"
                                    value={editingUser.password || ''}
                                    onChange={(e) => setEditingUser({ ...editingUser, password: e.target.value })}
                                    className="w-full bg-surfaceHighlight border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-primary-500/50"
                                    placeholder="••••••••"
                                />
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-xs font-medium text-slate-400 mb-1.5">Role</label>
                                    <select
                                        value={editingUser.role}
                                        onChange={(e) => setEditingUser({ ...editingUser, role: e.target.value })}
                                        className="w-full bg-surfaceHighlight border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-primary-500/50"
                                    >
                                        <option value="doctor">Doctor</option>
                                        <option value="lab">Lab Technician</option>
                                        <option value="admin">Administrator</option>
                                        <option value="patient">Patient</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-xs font-medium text-slate-400 mb-1.5">Department</label>
                                    <input
                                        type="text"
                                        value={editingUser.department || ''}
                                        onChange={(e) => setEditingUser({ ...editingUser, department: e.target.value })}
                                        className="w-full bg-surfaceHighlight border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-primary-500/50"
                                        placeholder="Optional"
                                    />
                                </div>
                            </div>

                            {/* Optional Fields for Edit */}
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-xs font-medium text-slate-400 mb-1">Email (Optional)</label>
                                    <input
                                        type="email"
                                        value={editingUser.email || ''}
                                        onChange={(e) => setEditingUser({ ...editingUser, email: e.target.value })}
                                        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500"
                                        placeholder="john@example.com"
                                    />
                                </div>
                                <div>
                                    <label className="block text-xs font-medium text-slate-400 mb-1">Phone (Optional)</label>
                                    <input
                                        type="tel"
                                        value={editingUser.phone || ''}
                                        onChange={(e) => setEditingUser({ ...editingUser, phone: e.target.value })}
                                        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500"
                                        placeholder="+1 555-0123"
                                    />
                                </div>
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-slate-400 mb-1">Address (Optional)</label>
                                <input
                                    type="text"
                                    value={editingUser.address || ''}
                                    onChange={(e) => setEditingUser({ ...editingUser, address: e.target.value })}
                                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary-500"
                                    placeholder="123 Medical Center Dr"
                                />
                            </div>

                            <button
                                type="submit"
                                className="w-full py-2.5 bg-primary-600 hover:bg-primary-500 text-white font-medium rounded-xl transition-colors mt-2"
                            >
                                Update User
                            </button>
                        </form>
                    </div>
                </div>
            )}

            {/* Delete Confirmation Modal */}
            {isDeleteModalOpen && (
                <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <div className="bg-surface border border-white/10 rounded-2xl p-6 w-full max-w-sm shadow-2xl">
                        <div className="flex flex-col items-center text-center mb-6">
                            <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center mb-4">
                                <AlertTriangle className="w-6 h-6 text-red-500" />
                            </div>
                            <h3 className="text-lg font-bold text-white mb-2">Delete User?</h3>
                            <p className="text-sm text-slate-400">
                                Are you sure you want to delete <strong>{userToDelete?.name}</strong>? This action cannot be undone.
                            </p>
                        </div>
                        <div className="flex gap-3">
                            <button
                                onClick={() => setIsDeleteModalOpen(false)}
                                className="flex-1 py-2.5 bg-surfaceHighlight hover:bg-white/10 text-slate-300 font-medium rounded-xl transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleDeleteUser}
                                className="flex-1 py-2.5 bg-red-500 hover:bg-red-600 text-white font-medium rounded-xl transition-colors"
                            >
                                Delete
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AdminDashboard;
