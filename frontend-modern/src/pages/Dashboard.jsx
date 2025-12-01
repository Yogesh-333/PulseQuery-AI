import React, { useEffect, useState } from 'react';
import api from '../services/api';
import { Activity, Database, FileText, Cpu, CheckCircle, XCircle, Clock, Zap, TrendingUp, Shield } from 'lucide-react';
import { motion } from 'framer-motion';

const StatusCard = ({ title, value, status, icon: Icon, subtext, delay, color = "primary" }) => {
    const colorClasses = {
        primary: "from-primary-500/20 to-primary-600/5 border-primary-500/20 text-primary-400",
        accent: "from-accent/20 to-accent/5 border-accent/20 text-accent",
        emerald: "from-emerald-500/20 to-emerald-600/5 border-emerald-500/20 text-emerald-400",
        amber: "from-amber-500/20 to-amber-600/5 border-amber-500/20 text-amber-400",
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay }}
            className={`relative overflow-hidden rounded-3xl border backdrop-blur-md bg-gradient-to-br ${colorClasses[color]} p-6 group hover:scale-[1.02] transition-transform duration-300`}
        >
            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity transform group-hover:scale-110 duration-500">
                <Icon className="w-32 h-32" />
            </div>

            <div className="relative z-10">
                <div className="flex items-center justify-between mb-6">
                    <div className={`p-3 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-sm`}>
                        <Icon className="w-6 h-6" />
                    </div>
                    {status && (
                        <div className={`flex items-center gap-1.5 text-xs font-bold px-3 py-1.5 rounded-full border ${status === 'active' ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' :
                                status === 'loading' ? 'bg-amber-500/10 border-amber-500/20 text-amber-400' :
                                    'bg-red-500/10 border-red-500/20 text-red-400'
                            }`}>
                            <div className={`w-1.5 h-1.5 rounded-full ${status === 'active' ? 'bg-emerald-400 animate-pulse' : 'bg-current'}`} />
                            {status === 'active' ? 'ONLINE' : status === 'loading' ? 'LOADING' : 'OFFLINE'}
                        </div>
                    )}
                </div>

                <h3 className="text-slate-400 text-sm font-medium mb-1 uppercase tracking-wider">{title}</h3>
                <p className="text-3xl font-bold text-white mb-2 tracking-tight">{value}</p>
                <p className="text-xs text-white/50 font-medium flex items-center gap-1">
                    {subtext}
                </p>
            </div>
        </motion.div>
    );
};

const Dashboard = () => {
    const [health, setHealth] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchHealth = async () => {
            try {
                const response = await api.get('/health');
                setHealth(response.data);
            } catch (error) {
                console.error('Failed to fetch health status', error);
            } finally {
                setLoading(false);
            }
        };

        fetchHealth();
        const interval = setInterval(fetchHealth, 10000);
        return () => clearInterval(interval);
    }, []);

    if (loading) return (
        <div className="flex items-center justify-center h-[50vh]">
            <div className="relative w-20 h-20">
                <div className="absolute inset-0 border-4 border-primary-500/20 rounded-full" />
                <div className="absolute inset-0 border-4 border-primary-500 border-t-transparent rounded-full animate-spin" />
            </div>
        </div>
    );

    const components = health?.components || {};
    const medgemmaReady = components.medgemma?.includes('Ready');
    const ragReady = components.rag_system?.includes('Ready');

    return (
        <div className="space-y-8">
            <div className="flex items-end justify-between">
                <div>
                    <h2 className="text-3xl font-bold text-white mb-2 tracking-tight">System Overview</h2>
                    <p className="text-slate-400">Real-time telemetry and performance metrics</p>
                </div>
                <div className="flex items-center gap-2 text-xs font-mono text-slate-500 bg-surfaceHighlight/50 px-3 py-1 rounded-lg border border-white/5">
                    <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                    SYSTEM OPERATIONAL
                </div>
            </div>

            {/* Bento Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {/* Large Card - MedGemma */}
                <div className="md:col-span-2">
                    <StatusCard
                        title="MedGemma Core"
                        value={medgemmaReady ? "Active & Ready" : "Initializing..."}
                        status={medgemmaReady ? "active" : "loading"}
                        icon={Cpu}
                        subtext="4B Parameter Medical LLM • GPU Acceleration Enabled"
                        delay={0.1}
                        color="accent"
                    />
                </div>

                <StatusCard
                    title="RAG Vector DB"
                    value={ragReady ? "Connected" : "Disconnected"}
                    status={ragReady ? "active" : "error"}
                    icon={Database}
                    subtext="ChromaDB Persistent Storage"
                    delay={0.2}
                    color="primary"
                />

                <StatusCard
                    title="Prompt Engine"
                    value={components.prompt_optimizer?.includes('Ready') ? "Optimized" : "Disabled"}
                    status={components.prompt_optimizer?.includes('Ready') ? "active" : "error"}
                    icon={Zap}
                    subtext="English Medical Templates V2"
                    delay={0.3}
                    color="amber"
                />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Activity Feed */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="lg:col-span-2 glass-panel rounded-3xl p-8"
                >
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-lg font-bold text-white flex items-center gap-2">
                            <Activity className="w-5 h-5 text-primary-400" />
                            Live Activity Feed
                        </h3>
                        <button className="text-xs text-primary-400 hover:text-primary-300 font-medium">View All</button>
                    </div>

                    <div className="space-y-4">
                        <div className="p-4 bg-surface/50 rounded-2xl border border-white/5 flex items-center gap-4 group hover:border-primary-500/20 transition-colors">
                            <div className="w-10 h-10 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-400">
                                <CheckCircle className="w-5 h-5" />
                            </div>
                            <div>
                                <p className="text-white font-medium">System Startup Complete</p>
                                <p className="text-xs text-slate-500">All services initialized successfully</p>
                            </div>
                            <span className="ml-auto text-xs font-mono text-slate-500">Just now</span>
                        </div>

                        <div className="p-4 bg-surface/50 rounded-2xl border border-white/5 flex items-center gap-4 group hover:border-primary-500/20 transition-colors">
                            <div className="w-10 h-10 rounded-full bg-primary-500/10 flex items-center justify-center text-primary-400">
                                <Shield className="w-5 h-5" />
                            </div>
                            <div>
                                <p className="text-white font-medium">Security Check Passed</p>
                                <p className="text-xs text-slate-500">Authentication service verified</p>
                            </div>
                            <span className="ml-auto text-xs font-mono text-slate-500">2m ago</span>
                        </div>
                    </div>
                </motion.div>

                {/* Stats Column */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="glass-panel rounded-3xl p-8 flex flex-col justify-between"
                >
                    <div>
                        <h3 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
                            <Database className="w-5 h-5 text-accent" />
                            Knowledge Base
                        </h3>

                        <div className="space-y-6">
                            <div>
                                <p className="text-slate-400 text-sm mb-1">Total Documents</p>
                                <p className="text-4xl font-bold text-white">{health?.debug_info?.rag_stats?.total_documents || 0}</p>
                            </div>

                            <div>
                                <p className="text-slate-400 text-sm mb-1">Embedding Model</p>
                                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-lg bg-surface border border-white/10">
                                    <div className="w-2 h-2 bg-accent rounded-full" />
                                    <span className="text-sm text-white font-mono">{health?.debug_info?.rag_stats?.embedding_model || 'Unknown'}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="mt-8 pt-6 border-t border-white/5">
                        <div className="flex items-center justify-between text-sm">
                            <span className="text-slate-400">Storage Usage</span>
                            <span className="text-white font-medium">12%</span>
                        </div>
                        <div className="w-full h-2 bg-surface rounded-full mt-2 overflow-hidden">
                            <div className="h-full w-[12%] bg-gradient-to-r from-primary-500 to-accent rounded-full" />
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
};

export default Dashboard;
