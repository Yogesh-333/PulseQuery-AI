import React, { useState, useRef } from 'react';
import api from '../services/api';
import { Upload, FileText, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const DocumentUpload = () => {
    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [status, setStatus] = useState(null); // 'success' | 'error'
    const [message, setMessage] = useState('');
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setStatus(null);
            setMessage('');
        }
    };

    const handleUpload = async (e) => {
        e.preventDefault();
        if (!file) return;

        setUploading(true);
        setStatus(null);
        setMessage('');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await api.post('/api/rag/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            if (response.data.success) {
                setStatus('success');
                setMessage(`Successfully processed ${response.data.chunks_created} chunks from ${response.data.original_filename}`);
                setFile(null);
                if (fileInputRef.current) fileInputRef.current.value = '';
            } else {
                setStatus('error');
                setMessage(response.data.error || 'Upload failed');
            }
        } catch (error) {
            setStatus('error');
            setMessage(error.response?.data?.error || 'Server connection failed');
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="max-w-3xl mx-auto space-y-6">
            <div>
                <h2 className="text-2xl font-bold text-white mb-2 flex items-center gap-2">
                    <Upload className="w-6 h-6 text-primary-400" />
                    Document Upload
                </h2>
                <p className="text-slate-400">Ingest medical documents into the RAG knowledge base</p>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8">
                <div
                    className={`border-2 border-dashed rounded-xl p-12 text-center transition-all ${file ? 'border-primary-500 bg-primary-500/5' : 'border-slate-700 hover:border-slate-600 hover:bg-slate-800/50'
                        }`}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => {
                        e.preventDefault();
                        const droppedFile = e.dataTransfer.files[0];
                        if (droppedFile) setFile(droppedFile);
                    }}
                >
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                        className="hidden"
                        accept=".txt,.pdf,.md"
                    />

                    <div className="flex flex-col items-center gap-4">
                        <div className={`w-16 h-16 rounded-full flex items-center justify-center ${file ? 'bg-primary-500/20 text-primary-400' : 'bg-slate-800 text-slate-400'
                            }`}>
                            {file ? <FileText className="w-8 h-8" /> : <Upload className="w-8 h-8" />}
                        </div>

                        {file ? (
                            <div>
                                <p className="text-lg font-medium text-white mb-1">{file.name}</p>
                                <p className="text-sm text-slate-500">{(file.size / 1024).toFixed(2)} KB</p>
                                <button
                                    onClick={() => {
                                        setFile(null);
                                        if (fileInputRef.current) fileInputRef.current.value = '';
                                    }}
                                    className="text-sm text-red-400 hover:text-red-300 mt-2"
                                >
                                    Remove file
                                </button>
                            </div>
                        ) : (
                            <div>
                                <p className="text-lg font-medium text-white mb-2">
                                    Drag and drop your file here, or{' '}
                                    <button
                                        onClick={() => fileInputRef.current?.click()}
                                        className="text-primary-400 hover:text-primary-300 font-bold"
                                    >
                                        browse
                                    </button>
                                </p>
                                <p className="text-sm text-slate-500">Supports PDF, TXT, MD</p>
                            </div>
                        )}
                    </div>
                </div>

                <div className="mt-6 flex justify-end">
                    <button
                        onClick={handleUpload}
                        disabled={!file || uploading}
                        className="flex items-center gap-2 px-6 py-3 bg-primary-600 hover:bg-primary-500 text-white rounded-xl font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-primary-500/20"
                    >
                        {uploading ? (
                            <>
                                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                Processing...
                            </>
                        ) : (
                            <>
                                <Upload className="w-4 h-4" />
                                Upload Document
                            </>
                        )}
                    </button>
                </div>
            </div>

            <AnimatePresence>
                {status && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 10 }}
                        className={`rounded-xl p-4 border flex items-start gap-3 ${status === 'success'
                                ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                                : 'bg-red-500/10 border-red-500/20 text-red-400'
                            }`}
                    >
                        {status === 'success' ? <CheckCircle className="w-5 h-5 shrink-0" /> : <XCircle className="w-5 h-5 shrink-0" />}
                        <div>
                            <h4 className="font-bold text-sm mb-1">
                                {status === 'success' ? 'Upload Successful' : 'Upload Failed'}
                            </h4>
                            <p className="text-sm opacity-90">{message}</p>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default DocumentUpload;
