import React, { useState } from 'react';
import api from '../services/api';
import { Send, Sparkles, Copy, Check, RefreshCw, AlertCircle, ArrowRight, Wand2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const PromptOptimizer = () => {
    const [query, setQuery] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [copied, setCopied] = useState(false);

    const handleOptimize = async (e) => {
        e.preventDefault();
        if (!query.trim()) return;

        setLoading(true);
        setError('');
        // Don't clear result immediately to allow comparison if needed, or clear it if you want a fresh state.
        // Let's clear it to show loading state clearly.
        setResult(null);

        try {
            const response = await api.post('/api/prompt/optimize', {
                query: query,
                use_context: true
            });

            if (response.data.success) {
                setResult(response.data);
            } else {
                setError(response.data.error || 'Optimization failed');
            }
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to connect to server');
        } finally {
            setLoading(false);
        }
    };

    const copyToClipboard = () => {
        if (result?.optimized_prompt) {
            navigator.clipboard.writeText(result.optimized_prompt);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    return (
        <div className="h-[calc(100vh-8rem)] flex flex-col">
            <div className="mb-6 flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                        <Wand2 className="w-6 h-6 text-primary-400" />
                        Prompt Optimizer
                    </h1>
                    <p className="text-slate-400 text-sm">Refine your medical queries with AI precision</p>
                </div>
            </div>

            <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 min-h-0">
                {/* Left Panel: Input */}
                <div className="flex flex-col bg-surface rounded-2xl border border-white/10 overflow-hidden shadow-xl">
                    <div className="p-4 border-b border-white/5 bg-surfaceHighlight/30 flex items-center justify-between">
                        <span className="text-sm font-medium text-slate-300">Input Query</span>
                        <span className="text-xs text-slate-500">{query.length} chars</span>
                    </div>

                    <div className="flex-1 relative">
                        <textarea
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="Describe the patient case, symptoms, or medical question here..."
                            className="w-full h-full bg-transparent p-6 text-base text-slate-200 placeholder-slate-600 resize-none focus:outline-none focus:bg-surfaceHighlight/10 transition-colors"
                        />

                        <div className="absolute bottom-6 right-6">
                            <button
                                onClick={handleOptimize}
                                disabled={loading || !query.trim()}
                                className="flex items-center gap-2 px-6 py-3 bg-primary-600 hover:bg-primary-500 text-white font-medium rounded-xl shadow-lg shadow-primary-500/20 transition-all transform hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                            >
                                {loading ? (
                                    <RefreshCw className="w-5 h-5 animate-spin" />
                                ) : (
                                    <>
                                        Optimize <ArrowRight className="w-5 h-5" />
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Right Panel: Output */}
                <div className="flex flex-col bg-surface rounded-2xl border border-white/10 overflow-hidden shadow-xl relative">
                    <div className="p-4 border-b border-white/5 bg-surfaceHighlight/30 flex items-center justify-between">
                        <span className="text-sm font-medium text-slate-300">Optimized Result</span>
                        {result && (
                            <button
                                onClick={copyToClipboard}
                                className="flex items-center gap-1.5 text-xs font-medium text-primary-400 hover:text-primary-300 transition-colors"
                            >
                                {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                                {copied ? 'Copied' : 'Copy'}
                            </button>
                        )}
                    </div>

                    <div className="flex-1 p-6 relative overflow-y-auto custom-scrollbar">
                        <AnimatePresence mode="wait">
                            {loading ? (
                                <motion.div
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    className="absolute inset-0 flex flex-col items-center justify-center text-center p-8"
                                >
                                    <div className="w-12 h-12 border-4 border-primary-500/20 border-t-primary-500 rounded-full animate-spin mb-4" />
                                    <p className="text-slate-400 font-medium">Analyzing medical context...</p>
                                </motion.div>
                            ) : error ? (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="flex flex-col items-center justify-center h-full text-center text-red-400"
                                >
                                    <AlertCircle className="w-10 h-10 mb-3 opacity-50" />
                                    <p>{error}</p>
                                </motion.div>
                            ) : result ? (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="space-y-6"
                                >
                                    <div className="prose prose-invert max-w-none">
                                        <p className="text-lg leading-relaxed text-slate-200 whitespace-pre-wrap">
                                            {result.optimized_prompt}
                                        </p>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4 pt-6 border-t border-white/5">
                                        <div className="p-3 rounded-lg bg-surfaceHighlight/30 border border-white/5">
                                            <p className="text-xs text-slate-500 mb-1">Specialty Detected</p>
                                            <p className="text-sm font-medium text-primary-400 capitalize">
                                                {result.medical_specialty?.replace('_', ' ') || 'General'}
                                            </p>
                                        </div>
                                        <div className="p-3 rounded-lg bg-surfaceHighlight/30 border border-white/5">
                                            <p className="text-xs text-slate-500 mb-1">Optimization Score</p>
                                            <p className="text-sm font-medium text-emerald-400">
                                                {result.metrics?.improvement_score
                                                    ? `+${Math.round(result.metrics.improvement_score * 100)}% Quality`
                                                    : 'High'}
                                            </p>
                                        </div>
                                    </div>
                                </motion.div>
                            ) : (
                                <div className="flex flex-col items-center justify-center h-full text-center text-slate-600">
                                    <Sparkles className="w-12 h-12 mb-4 opacity-20" />
                                    <p>Enter a query on the left to generate<br />an optimized medical prompt.</p>
                                </div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PromptOptimizer;
