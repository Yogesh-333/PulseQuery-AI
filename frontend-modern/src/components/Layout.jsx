import React from 'react';
import { Outlet, NavLink, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
    LayoutDashboard,
    MessageSquare,
    Upload,
    LogOut,
    Activity,
    Menu,
    X,
    ChevronRight,
    Shield
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import logo from '../assets/logo.png';

const SidebarItem = ({ to, icon: Icon, label }) => {
    const location = useLocation();
    const isActive = location.pathname === to;

    return (
        <NavLink
            to={to}
            className="relative group block"
        >
            {isActive && (
                <motion.div
                    layoutId="activeNav"
                    className="absolute inset-0 bg-gradient-to-r from-primary-500/20 to-indigo-500/20 rounded-xl border border-primary-500/30"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                />
            )}
            <div className={`relative flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${isActive ? 'text-white' : 'text-slate-400 group-hover:text-slate-200'
                }`}>
                <Icon className={`w-5 h-5 transition-colors ${isActive ? 'text-primary-400' : 'group-hover:text-white'}`} />
                <span className="font-medium">{label}</span>
                {isActive && (
                    <motion.div
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="ml-auto"
                    >
                        <ChevronRight className="w-4 h-4 text-primary-400" />
                    </motion.div>
                )}
            </div>
        </NavLink>
    );
};

const Layout = () => {
    const { logout, user } = useAuth();
    const navigate = useNavigate();
    const [isMobileMenuOpen, setIsMobileMenuOpen] = React.useState(false);

    const handleLogout = async () => {
        await logout();
        navigate('/login');
    };

    return (
        <div className="min-h-screen bg-mesh text-slate-200 flex overflow-hidden">
            {/* Background Orbs */}
            <div className="fixed top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
                <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary-500/10 rounded-full blur-[120px] animate-blob" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-accent/10 rounded-full blur-[120px] animate-blob animation-delay-2000" />
            </div>

            {/* Sidebar - Desktop */}
            <aside className="hidden md:flex flex-col w-72 h-[calc(100vh-2rem)] m-4 glass-panel rounded-3xl z-20 relative">
                <div className="p-8 flex items-center gap-4">
                    <div className="relative">
                        <div className="absolute inset-0 bg-primary-500 blur-lg opacity-50 animate-pulse-slow" />
                        <img src={logo} alt="OmniSynapse AI" className="relative w-12 h-12 object-contain drop-shadow-lg" />
                    </div>
                    <div>
                        <h1 className="font-bold text-xl text-white tracking-tight">OmniSynapse AI</h1>
                        <p className="text-xs text-primary-400 font-medium tracking-wider uppercase">AI Medical Core</p>
                    </div>
                </div>

                <nav className="flex-1 px-4 space-y-2 overflow-y-auto py-4">
                    {/* Admin Menu - Sees Everything */}
                    {user?.role === 'admin' && (
                        <>
                            <div className="px-4 pb-2">
                                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Main Menu</p>
                            </div>
                            <SidebarItem to="/" icon={LayoutDashboard} label="Dashboard" />
                            <SidebarItem to="/optimize" icon={MessageSquare} label="Prompt Optimizer" />
                            <SidebarItem to="/upload" icon={Upload} label="Knowledge Base" />
                            <SidebarItem to="/admin" icon={Shield} label="Admin Console" />
                        </>
                    )}

                    {/* Doctor Menu */}
                    {user?.role === 'doctor' && (
                        <>
                            <SidebarItem to="/doctor" icon={Activity} label="Doctor Dashboard" />
                            <SidebarItem to="/doctor/ai-ask" icon={MessageSquare} label="New AI Ask" />
                        </>
                    )}

                    {/* Lab Menu */}
                    {user?.role === 'lab' && (
                        <SidebarItem to="/lab" icon={Activity} label="Lab Dashboard" />
                    )}

                    {/* Patient Menu */}
                    {user?.role === 'patient' && (
                        <SidebarItem to="/patient" icon={Activity} label="Patient Dashboard" />
                    )}
                </nav>

                <div className="p-4 m-4 mt-auto glass-card rounded-2xl">
                    <div className="flex items-center gap-3 mb-3">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-slate-700 to-slate-900 border border-white/10 flex items-center justify-center text-sm font-bold text-white shadow-inner">
                            {user?.user_name?.charAt(0) || 'U'}
                        </div>
                        <div className="flex-1 min-w-0">
                            <p className="text-sm font-bold text-white truncate">{user?.user_name}</p>
                            <p className="text-xs text-slate-400 truncate">{user?.role || 'Medical Staff'}</p>
                        </div>
                    </div>
                    <button
                        onClick={handleLogout}
                        className="w-full flex items-center justify-center gap-2 px-4 py-2 text-xs font-medium text-red-300 hover:text-red-200 hover:bg-red-500/10 rounded-lg transition-colors border border-transparent hover:border-red-500/20"
                    >
                        <LogOut className="w-3 h-3" />
                        <span>Sign Out</span>
                    </button>
                </div>
            </aside>

            {/* Mobile Header */}
            <div className="md:hidden fixed top-0 left-0 right-0 h-16 glass-panel border-b border-white/5 flex items-center justify-between px-4 z-30">
                <div className="flex items-center gap-2">
                    <img src={logo} alt="OmniSynapse AI" className="w-8 h-8 object-contain" />
                    <span className="font-bold text-white text-lg">OmniSynapse AI</span>
                </div>
                <button
                    onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                    className="p-2 text-slate-400 hover:text-white"
                >
                    {isMobileMenuOpen ? <X /> : <Menu />}
                </button>
            </div>

            {/* Mobile Menu */}
            <AnimatePresence>
                {isMobileMenuOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="md:hidden fixed inset-0 top-16 bg-background/95 backdrop-blur-xl z-20 p-4"
                    >
                        <nav className="space-y-2">
                            {user?.role === 'admin' && (
                                <>
                                    <SidebarItem to="/" icon={LayoutDashboard} label="Dashboard" />
                                    <SidebarItem to="/optimize" icon={MessageSquare} label="Prompt Optimizer" />
                                    <SidebarItem to="/upload" icon={Upload} label="Knowledge Base" />
                                    <SidebarItem to="/admin" icon={Shield} label="Admin Console" />
                                </>
                            )}
                            {user?.role === 'doctor' && (
                                <>
                                    <SidebarItem to="/doctor" icon={Activity} label="Doctor Dashboard" />
                                    <SidebarItem to="/doctor/ai-ask" icon={MessageSquare} label="New AI Ask" />
                                </>
                            )}
                            {user?.role === 'lab' && <SidebarItem to="/lab" icon={Activity} label="Lab Dashboard" />}
                            {user?.role === 'patient' && <SidebarItem to="/patient" icon={Activity} label="Patient Dashboard" />}

                            <button
                                onClick={handleLogout}
                                className="w-full flex items-center gap-3 px-4 py-3 text-red-400 hover:bg-red-500/10 rounded-xl transition-colors mt-4 border border-red-500/10"
                            >
                                <LogOut className="w-5 h-5" />
                                <span>Sign Out</span>
                            </button>
                        </nav>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Main Content */}
            <main className="flex-1 p-4 md:p-8 pt-20 md:pt-8 overflow-y-auto min-h-screen relative z-10 scroll-smooth">
                <div className="max-w-7xl mx-auto pb-10">
                    <Outlet />
                </div>
            </main>
        </div>
    );
};

export default Layout;
