import React, { createContext, useState, useEffect, useContext } from 'react';
import api from '../services/api';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const storedUser = localStorage.getItem('user');
        if (storedUser) {
            setUser(JSON.parse(storedUser));
        }
        setLoading(false);
    }, []);

    const login = async (userId, password) => {
        try {
            const response = await api.post('/api/auth/login', {
                user_id: userId,
                password: password,
            });

            console.log('Login response:', response.data); // DEBUG
            if (response.data.success) {
                const userData = {
                    ...response.data.user,
                    session_id: response.data.session_id,
                };
                setUser(userData);
                localStorage.setItem('user', JSON.stringify(userData));
                return { success: true, user: userData };
            }
            return { success: false, error: response.data.error };
        } catch (error) {
            return {
                success: false,
                error: error.response?.data?.error || 'Login failed',
            };
        }
    };

    const logout = async () => {
        try {
            await api.post('/api/auth/logout');
        } catch (error) {
            console.error('Logout failed', error);
        } finally {
            setUser(null);
            localStorage.removeItem('user');
        }
    };

    return (
        <AuthContext.Provider value={{ user, login, logout, loading }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
