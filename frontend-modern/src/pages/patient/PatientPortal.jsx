import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { Calendar, FileText, Activity, User } from 'lucide-react';

const PatientPortal = () => {
    const [patientData, setPatientData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // For demo, we'll fetch patient ID 1 (John Doe)
        // In real app, we'd get the patient ID associated with the logged-in user
        const fetchMyData = async () => {
            try {
                const response = await api.get('/api/patients/1');
                setPatientData(response.data);
            } catch (error) {
                console.error("Failed to fetch patient data", error);
            } finally {
                setLoading(false);
            }
        };
        fetchMyData();
    }, []);

    if (loading) return <div className="p-8 text-white">Loading your health records...</div>;
    if (!patientData) return <div className="p-8 text-white">No records found.</div>;

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-white">My Health Portal</h1>
                    <p className="text-slate-400">Welcome back, {patientData.name}</p>
                </div>
                <div className="w-10 h-10 rounded-full bg-primary-500/20 flex items-center justify-center text-primary-400 border border-primary-500/30">
                    <User className="w-5 h-5" />
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="glass-card rounded-2xl p-6 border-t-4 border-primary-500">
                    <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                        <Calendar className="w-5 h-5 text-primary-400" />
                        Appointments
                    </h3>
                    <div className="space-y-3">
                        {patientData.appointments?.map((appt) => (
                            <div key={appt.id} className="p-3 bg-surfaceHighlight/30 rounded-xl border border-white/5">
                                <p className="text-sm font-medium text-white">{new Date(appt.date).toLocaleDateString()}</p>
                                <p className="text-xs text-slate-400">{appt.reason}</p>
                                <span className="text-xs text-primary-400 mt-1 block">{appt.status}</span>
                            </div>
                        ))}
                        {(!patientData.appointments || patientData.appointments.length === 0) && (
                            <p className="text-sm text-slate-500">No upcoming appointments</p>
                        )}
                    </div>
                </div>

                <div className="glass-card rounded-2xl p-6 border-t-4 border-emerald-500">
                    <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                        <Activity className="w-5 h-5 text-emerald-400" />
                        Recent Vitals
                    </h3>
                    {patientData.latest_vitals ? (
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Blood Pressure</span>
                                <span className="text-white font-medium">{patientData.latest_vitals.bp}</span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Heart Rate</span>
                                <span className="text-white font-medium">{patientData.latest_vitals.heart_rate} bpm</span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Weight</span>
                                <span className="text-white font-medium">{patientData.latest_vitals.weight} kg</span>
                            </div>
                        </div>
                    ) : (
                        <p className="text-sm text-slate-500">No vitals recorded</p>
                    )}
                </div>

                <div className="glass-card rounded-2xl p-6 border-t-4 border-amber-500">
                    <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                        <FileText className="w-5 h-5 text-amber-400" />
                        Lab Reports
                    </h3>
                    <div className="space-y-3">
                        {patientData.lab_requests?.map((req) => (
                            <div key={req.id} className="p-3 bg-surfaceHighlight/30 rounded-xl border border-white/5">
                                <div className="flex justify-between mb-1">
                                    <span className="text-white text-sm font-medium">{req.test_type}</span>
                                    <span className={`text-xs px-1.5 py-0.5 rounded ${req.status === 'completed' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'
                                        }`}>{req.status}</span>
                                </div>
                                {req.result && (
                                    <p className="text-xs text-slate-300 mt-1 bg-black/20 p-2 rounded">
                                        Result: {req.result}
                                    </p>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PatientPortal;
