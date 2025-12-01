import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Login from './pages/Login';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import PromptOptimizer from './pages/PromptOptimizer';
import DocumentUpload from './pages/DocumentUpload';
import RoleRoute from './components/RoleRoute';
import DoctorDashboard from './pages/doctor/DoctorDashboard';
import ConsultationView from './pages/doctor/ConsultationView';
import DoctorAIChat from './pages/doctor/DoctorAIChat';
import AdminDashboard from './pages/admin/AdminDashboard';
import LabDashboard from './pages/lab/LabDashboard';
import PatientPortal from './pages/patient/PatientPortal';

// Protected Route Wrapper
const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" />;
  }

  return children;
};

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/login" element={<Login />} />

          {/* Main Layout Routes */}
          <Route path="/" element={<RoleRoute><Layout /></RoleRoute>}>
            <Route index element={<Dashboard />} />
            <Route path="optimize" element={<PromptOptimizer />} />
            <Route path="upload" element={<DocumentUpload />} />

            {/* Doctor Routes */}
            <Route path="doctor" element={<RoleRoute roles={['doctor']}><DoctorDashboard /></RoleRoute>} />
            <Route path="doctor/patient/:id" element={<RoleRoute roles={['doctor']}><ConsultationView /></RoleRoute>} />
            <Route path="doctor/ai-ask" element={<RoleRoute roles={['doctor']}><DoctorAIChat /></RoleRoute>} />

            {/* Admin Routes */}
            <Route path="admin" element={<RoleRoute roles={['admin']}><AdminDashboard /></RoleRoute>} />

            {/* Lab Routes */}
            <Route path="lab" element={<RoleRoute roles={['lab']}><LabDashboard /></RoleRoute>} />

            {/* Patient Routes */}
            <Route path="portal" element={<RoleRoute roles={['patient']}><PatientPortal /></RoleRoute>} />
          </Route>
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
