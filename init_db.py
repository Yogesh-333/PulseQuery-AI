from app import create_app
from models import db, User, Patient, Vital, LabRequest, Appointment
from datetime import datetime, timedelta

app = create_app()

with app.app_context():
    db.create_all()
    
    # Check if users exist
    if not User.query.filter_by(username='admin1').first():
        print("Creating default users...")
        
        # Admin
        admin = User(username='admin1', role='admin', name='Admin Eve Wilson', department='IT')
        admin.set_password('admin123')
        db.session.add(admin)
        
        # Doctor
        doctor = User(username='doctor1', role='doctor', name='Dr. Alice Johnson', department='Cardiology')
        doctor.set_password('password123')
        db.session.add(doctor)
        
        # Lab
        lab = User(username='lab1', role='lab', name='Lab Technician Max', department='Pathology')
        lab.set_password('lab123')
        db.session.add(lab)
        
        # Patient User (for login)
        patient_user = User(username='patient1', role='patient', name='John Doe', department=None)
        patient_user.set_password('patient123')
        db.session.add(patient_user)
        
        db.session.commit()
        
        # Create Patient Record
        print("Creating dummy patient data...")
        patient = Patient(name='John Doe', age=45, gender='Male', contact='555-0123', medical_history='Hypertension, Type 2 Diabetes')
        db.session.add(patient)
        db.session.commit()
        
        # Create Vitals
        vitals = Vital(patient_id=patient.id, bp='130/85', heart_rate=78, temperature=36.6, weight=82.5)
        db.session.add(vitals)
        
        # Create Lab Request
        lab_req = LabRequest(patient_id=patient.id, doctor_id=doctor.id, test_type='Complete Blood Count', status='pending')
        db.session.add(lab_req)
        
        # Create Appointment
        appt = Appointment(patient_id=patient.id, doctor_id=doctor.id, date=datetime.now() + timedelta(days=2), reason='Regular Checkup')
        db.session.add(appt)
        
        db.session.commit()
        print("✅ Database initialized with default data!")
    else:
        print("ℹ️ Database already initialized.")
