import React, { useState, useRef } from 'react';
import api from '../../services/api';
import { Activity, MessageSquare, Send, Sparkles } from 'lucide-react';

const DoctorAIChat = () => {
    const [chatHistory, setChatHistory] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const chatEndRef = useRef(null);

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = { role: 'user', content: input };
        setChatHistory(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const response = await api.post('/api/consultation/chat', {
                query: userMsg.content
                // No patient_id needed for general queries
            });

            const aiMsg = { role: 'assistant', content: response.data.response };
            setChatHistory(prev => [...prev, aiMsg]);
        } catch (error) {
            console.error("Chat error", error);
            const errorMsg = { role: 'assistant', content: "I apologize, but I encountered an error processing your request. Please try again." };
            setChatHistory(prev => [...prev, errorMsg]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="h-[calc(100vh-6rem)] flex flex-col glass-panel rounded-2xl overflow-hidden">
            <div className="p-6 border-b border-white/10 bg-surfaceHighlight/30 flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-bold text-white flex items-center gap-3">
                        <Sparkles className="w-6 h-6 text-primary-400" />
                        MedGemma AI Assistant
                    </h2>
                    <p className="text-sm text-slate-400 mt-1">Ask general medical questions, research conditions, or get second opinions.</p>
                </div>
                <span className="text-xs px-3 py-1.5 bg-primary-500/10 text-primary-400 rounded-full border border-primary-500/20 font-medium">
                    General Medical Mode
                </span>
            </div>

            <div className="flex-1 p-6 overflow-y-auto space-y-6 custom-scrollbar bg-surface/50">
                {chatHistory.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-full text-slate-500 opacity-50 space-y-4">
                        <div className="w-20 h-20 rounded-full bg-primary-500/10 flex items-center justify-center">
                            <Activity className="w-10 h-10 text-primary-400" />
                        </div>
                        <div className="text-center max-w-md">
                            <h3 className="text-lg font-medium text-white mb-2">How can I help you today?</h3>
                            <p>Ask about drug interactions, latest treatment protocols, differential diagnoses, or medical research.</p>
                        </div>
                    </div>
                )}

                {chatHistory.map((msg, idx) => (
                    <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[80%] p-5 rounded-2xl shadow-lg ${msg.role === 'user'
                            ? 'bg-gradient-to-br from-primary-600 to-indigo-600 text-white rounded-tr-none'
                            : 'bg-surfaceHighlight border border-white/10 text-slate-200 rounded-tl-none'
                            }`}>
                            <p className="text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                        </div>
                    </div>
                ))}

                {loading && (
                    <div className="flex justify-start">
                        <div className="bg-surfaceHighlight border border-white/10 p-4 rounded-2xl rounded-tl-none">
                            <div className="flex gap-1.5">
                                <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" />
                                <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce delay-100" />
                                <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce delay-200" />
                            </div>
                        </div>
                    </div>
                )}
                <div ref={chatEndRef} />
            </div>

            <div className="p-6 bg-surfaceHighlight/30 border-t border-white/10">
                <form onSubmit={handleSendMessage} className="relative max-w-4xl mx-auto">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Type your medical question here..."
                        className="w-full bg-surface border border-white/10 rounded-xl pl-6 pr-14 py-4 text-base text-white focus:outline-none focus:border-primary-500/50 focus:ring-1 focus:ring-primary-500/50 shadow-inner transition-all placeholder-slate-500"
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || loading}
                        className="absolute right-2 top-1/2 -translate-y-1/2 p-2.5 bg-primary-500 text-white rounded-lg hover:bg-primary-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-primary-500/20"
                    >
                        <Send className="w-5 h-5" />
                    </button>
                </form>
                <p className="text-center text-xs text-slate-500 mt-3">
                    AI responses are for informational purposes only and should be verified by a medical professional.
                </p>
            </div>
        </div>
    );
};

export default DoctorAIChat;
