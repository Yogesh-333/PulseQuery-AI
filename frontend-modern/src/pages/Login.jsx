import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Activity, Lock, User, ArrowRight, ShieldCheck } from 'lucide-react';
import { motion } from 'framer-motion';
import logo from '../assets/logo.png';

const Login = () => {
    const [userId, setUserId] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        const result = await login(userId, password);
        if (result.success) {
            // Redirect based on role
            switch (result.user.role) {
                case 'doctor':
                    navigate('/doctor');
                    break;
                case 'admin':
                    navigate('/admin');
                    break;
                case 'lab':
                    navigate('/lab');
                    break;
                case 'patient':
                    navigate('/portal');
                    break;
                default:
                    navigate('/');
            }
        } else {
            setError(result.error);
        }
        setIsLoading(false);
    };

    return (
        <div className="min-h-screen bg-[#030712] flex items-center justify-center p-4 relative overflow-hidden">
            {/* Immersive Background */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] bg-primary-600/20 rounded-full blur-[150px] animate-blob" />
                <div className="absolute bottom-[-20%] right-[-10%] w-[60%] h-[60%] bg-accent/20 rounded-full blur-[150px] animate-blob animation-delay-2000" />
                <div className="absolute top-[40%] left-[40%] w-[30%] h-[30%] bg-indigo-500/10 rounded-full blur-[100px] animate-pulse-slow" />

                {/* Grid Pattern Overlay */}
                <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20" />
            </div>

            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, ease: "easeOut" }}
                className="relative w-full max-w-md"
            >
                <div className="glass-panel rounded-3xl p-1 shadow-2xl border border-white/10">
                    <div className="bg-surface/50 rounded-[22px] p-8 backdrop-blur-sm">
                        <div className="flex flex-col items-center mb-10">
                            <div className="relative mb-6">
                                <div className="absolute inset-0 bg-primary-500 blur-xl opacity-50 animate-pulse" />
                                <img src={logo} alt="OmniSynapse AI" className="relative w-24 h-24 object-contain drop-shadow-2xl" />
                            </div>
                            <h1 className="text-4xl font-bold text-white mb-2 tracking-tight">OmniSynapse AI</h1>
                            <p className="text-slate-400 font-medium tracking-wide">AI Medical Intelligence</p>
                        </div>

                        <form onSubmit={handleSubmit} className="space-y-6">
                            {error && (
                                <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-red-200 text-sm flex items-center gap-3"
                                >
                                    <ShieldCheck className="w-5 h-5 shrink-0" />
                                    {error}
                                </motion.div>
                            )}

                            <div className="space-y-4">
                                <div className="relative group">
                                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                        <User className="h-5 w-5 text-slate-500 group-focus-within:text-primary-400 transition-colors" />
                                    </div>
                                    <input
                                        type="text"
                                        value={userId}
                                        onChange={(e) => setUserId(e.target.value)}
                                        className="block w-full pl-12 pr-4 py-4 bg-surfaceHighlight/50 border border-white/5 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500/50 transition-all"
                                        placeholder="User ID"
                                        required
                                    />
                                </div>

                                <div className="relative group">
                                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                        <Lock className="h-5 w-5 text-slate-500 group-focus-within:text-primary-400 transition-colors" />
                                    </div>
                                    <input
                                        type="password"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        className="block w-full pl-12 pr-4 py-4 bg-surfaceHighlight/50 border border-white/5 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500/50 transition-all"
                                        placeholder="Password"
                                        required
                                    />
                                </div>
                            </div>

                            <button
                                type="submit"
                                disabled={isLoading}
                                className="w-full flex items-center justify-center py-4 px-4 bg-gradient-to-r from-primary-600 to-indigo-600 hover:from-primary-500 hover:to-indigo-500 text-white font-bold rounded-xl shadow-lg shadow-primary-500/25 transition-all transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed group"
                            >
                                {isLoading ? (
                                    <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                ) : (
                                    <>
                                        Access System
                                        <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
                                    </>
                                )}
                            </button>
                        </form>
                    </div>

                    <div className="px-8 py-5 bg-surface/30 border-t border-white/5 text-center rounded-b-[22px]">
                        <p className="text-xs text-slate-500 font-medium flex items-center justify-center gap-2">
                            <ShieldCheck className="w-3 h-3" />
                            Secure Medical Environment • 256-bit Encryption
                        </p>
                    </div>
                </div>
            </motion.div>
        </div>
    );
};

export default Login;
