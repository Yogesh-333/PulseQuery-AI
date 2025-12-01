import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const RoleRoute = ({ children, roles = [] }) => {
    const { user, loading } = useAuth();

    if (loading) {
        return <div className="min-h-screen bg-background flex items-center justify-center">
            <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
        </div>;
    }

    if (!user) {
        return <Navigate to="/login" />;
    }

    if (roles.length > 0 && !roles.includes(user.role)) {
        // Redirect to appropriate dashboard based on their actual role
        if (user.role === 'admin') return <Navigate to="/admin" />;
        if (user.role === 'doctor') return <Navigate to="/doctor" />;
        if (user.role === 'lab') return <Navigate to="/lab" />;
        if (user.role === 'patient') return <Navigate to="/portal" />;
        return <Navigate to="/" />;
    }

    return children;
};

export default RoleRoute;
