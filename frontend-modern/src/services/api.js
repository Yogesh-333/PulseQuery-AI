import axios from 'axios';

const api = axios.create({
    baseURL: 'http://localhost:5000', // Flask backend URL
    headers: {
        'Content-Type': 'application/json',
    },
});

// Add a request interceptor to include the session ID if available
api.interceptors.request.use(
    (config) => {
        const user = JSON.parse(localStorage.getItem('user'));
        if (user && user.session_id) {
            config.headers['X-Session-ID'] = user.session_id;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

export default api;
