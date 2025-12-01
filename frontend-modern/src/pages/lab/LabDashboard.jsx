import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { FlaskConical, CheckCircle, Clock, AlertCircle } from 'lucide-react';

const LabDashboard = () => {
    const [requests, setRequests] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchRequests();
    }, []);

    const fetchRequests = async () => {
        try {
            const response = await api.get('/api/labs/requests');
            setRequests(response.data);
        } catch (error) {
            console.error("Failed to fetch lab requests", error);
        } finally {
            setLoading(false);
        }
    };

    const handleUpdateResult = async (id) => {
        const result = prompt("Enter test result:");
        if (!result) return;

        try {
            await api.post(`/api/labs/update/${id}`, { result });
            fetchRequests(); // Refresh
        } catch (error) {
            alert("Failed to update result");
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-white">Laboratory Console</h1>
                    <p className="text-slate-400">Manage test requests and results</p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="glass-card rounded-2xl p-6 border-l-4 border-amber-500">
                    <div className="flex items-center gap-3 mb-2">
                        <Clock className="w-5 h-5 text-amber-500" />
                        <h3 className="font-bold text-white">Pending</h3>
                    </div>
                    <p className="text-2xl font-bold text-white">
                        {requests.filter(r => r.status === 'pending').length}
                    </p>
                </div>
                <div className="glass-card rounded-2xl p-6 border-l-4 border-emerald-500">
                    <div className="flex items-center gap-3 mb-2">
                        <CheckCircle className="w-5 h-5 text-emerald-500" />
                        <h3 className="font-bold text-white">Completed</h3>
                    </div>
                    <p className="text-2xl font-bold text-white">
                        {requests.filter(r => r.status === 'completed').length}
                    </p>
                </div>
            </div>

            <div className="glass-panel rounded-2xl overflow-hidden">
                <div className="p-4 border-b border-white/5 bg-surfaceHighlight/30">
                    <h2 className="font-bold text-white">Test Requests Queue</h2>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm text-slate-400">
                        <thead className="bg-surfaceHighlight/50 text-xs uppercase font-medium text-slate-300">
                            <tr>
                                <th className="px-6 py-4">Date</th>
                                <th className="px-6 py-4">Patient</th>
                                <th className="px-6 py-4">Test Type</th>
                                <th className="px-6 py-4">Doctor</th>
                                <th className="px-6 py-4">Status</th>
                                <th className="px-6 py-4">Action</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                            {requests.map((req) => (
                                <tr key={req.id} className="hover:bg-white/5 transition-colors">
                                    <td className="px-6 py-4">{new Date(req.requested_at).toLocaleDateString()}</td>
                                    <td className="px-6 py-4 font-medium text-white">{req.patient_name}</td>
                                    <td className="px-6 py-4">{req.test_type}</td>
                                    <td className="px-6 py-4">{req.doctor_name}</td>
                                    <td className="px-6 py-4">
                                        <span className={`px-2 py-1 rounded-md text-xs font-medium ${req.status === 'completed' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'
                                            }`}>
                                            {req.status.toUpperCase()}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4">
                                        {req.status === 'pending' ? (
                                            <button
                                                onClick={() => handleUpdateResult(req.id)}
                                                className="text-primary-400 hover:text-primary-300 font-medium"
                                            >
                                                Enter Result
                                            </button>
                                        ) : (
                                            <span className="text-slate-500">View Result</span>
                                        )}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default LabDashboard;
