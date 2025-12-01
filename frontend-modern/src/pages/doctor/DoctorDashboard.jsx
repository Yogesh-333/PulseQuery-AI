import React, { useEffect, useState } from 'react';
import api from '../../services/api';
import { Users, Activity, Calendar, Search, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';

const DoctorDashboard = () => {
    const [patients, setPatients] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchPatients = async () => {
            try {
                const response = await api.get('/api/patients');
                setPatients(response.data);
            } catch (error) {
                console.error("Failed to fetch patients", error);
            } finally {
                setLoading(false);
            }
        };
        fetchPatients();
    }, []);

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-white">Doctor Dashboard</h1>
                    <p className="text-slate-400">Manage patients and consultations</p>
                </div>
                <div className="flex gap-3">
                    <div className="bg-surfaceHighlight rounded-xl p-3 flex items-center gap-3 border border-white/5">
                        <Users className="w-5 h-5 text-primary-400" />
                        <div>
                            <p className="text-xs text-slate-400">Total Patients</p>
                            <p className="text-lg font-bold text-white leading-none">{patients.length}</p>
                        </div>
                    </div>
                </div>
            </div>

            <div className="glass-panel rounded-2xl overflow-hidden">
                <div className="p-4 border-b border-white/5 bg-surfaceHighlight/30 flex items-center justify-between">
                    <h2 className="font-bold text-white">Patient List</h2>
                    <div className="relative">
                        <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                        <input
                            type="text"
                            placeholder="Search patients..."
                            className="bg-surface border border-white/10 rounded-lg pl-9 pr-4 py-2 text-sm text-white focus:outline-none focus:border-primary-500/50"
                        />
                    </div>
                </div>

                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm text-slate-400">
                        <thead className="bg-surfaceHighlight/50 text-xs uppercase font-medium text-slate-300">
                            <tr>
                                <th className="px-6 py-4">Name</th>
                                <th className="px-6 py-4">Age/Gender</th>
                                <th className="px-6 py-4">Contact</th>
                                <th className="px-6 py-4">History</th>
                                <th className="px-6 py-4">Action</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                            {patients.map((patient) => (
                                <tr key={patient.id} className="hover:bg-white/5 transition-colors">
                                    <td className="px-6 py-4 font-medium text-white">{patient.name}</td>
                                    <td className="px-6 py-4">{patient.age} / {patient.gender}</td>
                                    <td className="px-6 py-4">{patient.contact}</td>
                                    <td className="px-6 py-4 truncate max-w-xs">{patient.medical_history}</td>
                                    <td className="px-6 py-4">
                                        <Link
                                            to={`/doctor/patient/${patient.id}`}
                                            className="inline-flex items-center gap-1 text-primary-400 hover:text-primary-300 font-medium"
                                        >
                                            Consult <ArrowRight className="w-4 h-4" />
                                        </Link>
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

export default DoctorDashboard;
