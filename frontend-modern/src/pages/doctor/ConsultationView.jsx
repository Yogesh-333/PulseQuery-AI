import React, { useEffect, useState, useRef } from 'react';
import { useParams } from 'react-router-dom';
import api from '../../services/api';
import { Activity, FileText, MessageSquare, Send, Plus, FlaskConical, History, User } from 'lucide-react';
import { motion } from 'framer-motion';

const ConsultationView = () => {
    const { id } = useParams();
    const [patient, setPatient] = useState(null);
    const [chatHistory, setChatHistory] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const chatEndRef = useRef(null);

    useEffect(() => {
        const fetchPatient = async () => {
            try {
                const response = await api.get(`/api/patients/${id}`);
                setPatient(response.data);
            } catch (error) {
                console.error("Error fetching patient", error);
            }
        };
        fetchPatient();
    }, [id]);

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = { role: 'user', content: input };
        setChatHistory(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const response = await api.post('/api/consultation/chat', {
                query: userMsg.content,
                patient_id: id
            });

            const aiMsg = { role: 'assistant', content: response.data.response };
            setChatHistory(prev => [...prev, aiMsg]);
        } catch (error) {
            console.error("Chat error", error);
        } finally {
            setLoading(false);
        }
    };

    if (!patient) return <div className="p-8 text-white">Loading patient data...</div>;

    return (
        <div className="h-[calc(100vh-6rem)] grid grid-cols-12 gap-6">
            {/* Left Column: Patient Info & Vitals */}
            <div className="col-span-3 space-y-6 overflow-y-auto pr-2 custom-scrollbar">
                <div className="glass-panel rounded-2xl p-6">
                    <div className="flex items-center gap-4 mb-6">
                        <div className="w-16 h-16 rounded-full bg-primary-500/20 flex items-center justify-center text-primary-400 border border-primary-500/30">
                            <User className="w-8 h-8" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-white">{patient.name}</h2>
                            <p className="text-sm text-slate-400">{patient.age} yrs • {patient.gender}</p>
                        </div>
                    </div>

                    <div className="space-y-4">
                        <div className="p-3 bg-surfaceHighlight/50 rounded-xl border border-white/5">
                            <p className="text-xs text-slate-500 mb-1">Medical History</p>
                            <p className="text-sm text-slate-200">{patient.medical_history}</p>
                        </div>
                        <div className="p-3 bg-surfaceHighlight/50 rounded-xl border border-white/5">
                            <p className="text-xs text-slate-500 mb-1">Contact</p>
                            <p className="text-sm text-slate-200">{patient.contact}</p>
                        </div>
                    </div>
                </div>

                <div className="glass-card rounded-2xl p-6 border-t-4 border-emerald-500">
                    <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                        <Activity className="w-5 h-5 text-emerald-400" />
                        Latest Vitals
                    </h3>
                    {patient.latest_vitals ? (
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <p className="text-xs text-slate-500">Blood Pressure</p>
                                <p className="text-lg font-bold text-white">{patient.latest_vitals.bp}</p>
                            </div>
                            <div>
                                <p className="text-xs text-slate-500">Heart Rate</p>
                                <p className="text-lg font-bold text-white">{patient.latest_vitals.heart_rate} <span className="text-xs font-normal">bpm</span></p>
                            </div>
                            <div>
                                <p className="text-xs text-slate-500">Temperature</p>
                                <p className="text-lg font-bold text-white">{patient.latest_vitals.temperature}°C</p>
                            </div>
                            <div>
                                <p className="text-xs text-slate-500">Weight</p>
                                <p className="text-lg font-bold text-white">{patient.latest_vitals.weight} <span className="text-xs font-normal">kg</span></p>
                            </div>
                        </div>
                    ) : (
                        <p className="text-sm text-slate-500">No vitals recorded</p>
                    )}
                </div>
            </div>

            {/* Center Column: AI Consultation Chat */}
            <div className="col-span-6 flex flex-col glass-panel rounded-2xl overflow-hidden">
                <div className="p-4 border-b border-white/10 bg-surfaceHighlight/30 flex items-center justify-between">
                    <h3 className="font-bold text-white flex items-center gap-2">
                        <MessageSquare className="w-5 h-5 text-primary-400" />
                        AI Consultation Assistant
                    </h3>
                    <span className="text-xs px-2 py-1 bg-primary-500/10 text-primary-400 rounded-full border border-primary-500/20">
                        Context Aware
                    </span>
                </div>

                <div className="flex-1 p-4 overflow-y-auto space-y-4 custom-scrollbar bg-surface/50">
                    {chatHistory.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-full text-slate-500 opacity-50">
                            <Activity className="w-12 h-12 mb-2" />
                            <p>Ask about patient history, diagnosis, or treatment options.</p>
                        </div>
                    )}

                    {chatHistory.map((msg, idx) => (
                        <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-[80%] p-4 rounded-2xl ${msg.role === 'user'
                                    ? 'bg-primary-600 text-white rounded-tr-none'
                                    : 'bg-surfaceHighlight border border-white/10 text-slate-200 rounded-tl-none'
                                }`}>
                                <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                            </div>
                        </div>
                    ))}
                    {loading && (
                        <div className="flex justify-start">
                            <div className="bg-surfaceHighlight border border-white/10 p-4 rounded-2xl rounded-tl-none">
                                <div className="flex gap-1">
                                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" />
                                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-100" />
                                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-200" />
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={chatEndRef} />
                </div>

                <div className="p-4 bg-surfaceHighlight/30 border-t border-white/10">
                    <form onSubmit={handleSendMessage} className="relative">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask AI about this patient..."
                            className="w-full bg-surface border border-white/10 rounded-xl pl-4 pr-12 py-3 text-sm text-white focus:outline-none focus:border-primary-500/50"
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || loading}
                            className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-primary-500 text-white rounded-lg hover:bg-primary-400 transition-colors disabled:opacity-50"
                        >
                            <Send className="w-4 h-4" />
                        </button>
                    </form>
                </div>
            </div>

            {/* Right Column: Actions & History */}
            <div className="col-span-3 space-y-6">
                <div className="glass-panel rounded-2xl p-6">
                    <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                        <FlaskConical className="w-5 h-5 text-accent" />
                        Lab Requests
                    </h3>
                    <div className="space-y-3 mb-4">
                        {patient.lab_requests?.slice(0, 3).map((req) => (
                            <div key={req.id} className="p-3 bg-surfaceHighlight/30 rounded-xl border border-white/5 text-sm">
                                <div className="flex justify-between mb-1">
                                    <span className="text-white font-medium">{req.test_type}</span>
                                    <span className={`text-xs px-1.5 py-0.5 rounded ${req.status === 'completed' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'
                                        }`}>{req.status}</span>
                                </div>
                                <p className="text-xs text-slate-500">{new Date(req.requested_at).toLocaleDateString()}</p>
                            </div>
                        ))}
                    </div>
                    <button className="w-full py-2 bg-surfaceHighlight hover:bg-white/10 border border-white/10 rounded-xl text-sm text-white font-medium transition-colors flex items-center justify-center gap-2">
                        <Plus className="w-4 h-4" /> New Request
                    </button>
                </div>

                <div className="glass-panel rounded-2xl p-6">
                    <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                        <History className="w-5 h-5 text-slate-400" />
                        Past Visits
                    </h3>
                    <div className="space-y-3">
                        {patient.appointments?.map((appt) => (
                            <div key={appt.id} className="flex items-center gap-3 p-2 hover:bg-white/5 rounded-lg transition-colors cursor-pointer">
                                <div className="w-10 h-10 rounded-lg bg-surfaceHighlight flex flex-col items-center justify-center text-xs border border-white/5">
                                    <span className="font-bold text-white">{new Date(appt.date).getDate()}</span>
                                    <span className="text-slate-500">{new Date(appt.date).toLocaleString('default', { month: 'short' })}</span>
                                </div>
                                <div>
                                    <p className="text-sm text-white font-medium">{appt.reason}</p>
                                    <p className="text-xs text-slate-500">{appt.status}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ConsultationView;
