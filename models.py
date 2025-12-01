from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import hashlib

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # admin, doctor, lab, patient
    name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100))
    email = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    address = db.Column(db.String(200))
    
    def set_password(self, password):
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()
        
    def check_password(self, password):
        return self.password_hash == hashlib.sha256(password.encode()).hexdigest()
        
    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "name": self.name,
            "department": self.department,
            "email": self.email,
            "phone": self.phone,
            "address": self.address
        }

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    contact = db.Column(db.String(50))
    medical_history = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    vitals = db.relationship('Vital', backref='patient', lazy=True)
    lab_requests = db.relationship('LabRequest', backref='patient', lazy=True)
    appointments = db.relationship('Appointment', backref='patient', lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "contact": self.contact,
            "medical_history": self.medical_history
        }

class Vital(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)
    bp = db.Column(db.String(20))  # Blood Pressure (e.g., "120/80")
    heart_rate = db.Column(db.Integer)
    temperature = db.Column(db.Float)
    weight = db.Column(db.Float)
    
    def to_dict(self):
        return {
            "id": self.id,
            "recorded_at": self.recorded_at.isoformat(),
            "bp": self.bp,
            "heart_rate": self.heart_rate,
            "temperature": self.temperature,
            "weight": self.weight
        }

class LabRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    test_type = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, completed
    result = db.Column(db.Text)
    requested_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    doctor = db.relationship('User', foreign_keys=[doctor_id])

    def to_dict(self):
        return {
            "id": self.id,
            "patient_name": self.patient.name,
            "doctor_name": self.doctor.name,
            "test_type": self.test_type,
            "status": self.status,
            "result": self.result,
            "requested_at": self.requested_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Can be null if not assigned yet
    date = db.Column(db.DateTime, nullable=False)
    reason = db.Column(db.String(200))
    status = db.Column(db.String(20), default='scheduled') # scheduled, completed, cancelled
    
    doctor = db.relationship('User', foreign_keys=[doctor_id])

    def to_dict(self):
        return {
            "id": self.id,
            "patient_name": self.patient.name,
            "doctor_name": self.doctor.name if self.doctor else "Unassigned",
            "date": self.date.isoformat(),
            "reason": self.reason,
            "status": self.status
        }
