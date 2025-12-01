import random
from datetime import datetime, timedelta
from app import create_app
from models import db, User, Patient, Vital, LabRequest, Appointment

def seed_database():
    app = create_app()
    with app.app_context():
        print("🗑️ Clearing existing data...")
        # Delete in reverse order of dependencies
        db.session.query(LabRequest).delete()
        db.session.query(Vital).delete()
        db.session.query(Appointment).delete()
        db.session.query(Patient).delete()
        db.session.query(User).delete()
        db.session.commit()
        print("✅ Data cleared.")

        print("🌱 Seeding Users...")
        # 1. Admin
        admin = User(
            username='admin1',
            role='admin',
            name='Admin User',
            department='Administration',
            email='admin@pulsequery.com',
            phone='555-0100',
            address='100 Admin Blvd'
        )
        admin.set_password('admin123')
        db.session.add(admin)

        # 2. Doctors
        specialties = ['Cardiology', 'Neurology', 'Pediatrics', 'Oncology', 'General Practice']
        doctors = []
        for i, spec in enumerate(specialties, 1):
            doc = User(
                username=f'doctor{i}',
                role='doctor',
                name=f'Dr. {["Smith", "Jones", "Chen", "Patel", "Wilson"][i-1]}',
                department=spec,
                email=f'doctor{i}@pulsequery.com',
                phone=f'555-020{i}',
                address=f'{i} Medical Plaza'
            )
            doc.set_password('password123')
            db.session.add(doc)
            doctors.append(doc)

        # 3. Lab Tech
        lab_tech = User(
            username='lab1',
            role='lab',
            name='Lab Technician',
            department='Pathology',
            email='lab@pulsequery.com',
            phone='555-0300',
            address='500 Lab Lane'
        )
        lab_tech.set_password('password123')
        db.session.add(lab_tech)

        db.session.commit()
        print(f"✅ Created {len(doctors) + 2} users.")

        print("🌱 Seeding Patients (Users + Records)...")
        patient_names = [
            ("John Doe", "Male"), ("Jane Smith", "Female"), ("Michael Johnson", "Male"),
            ("Emily Davis", "Female"), ("Robert Brown", "Male"), ("Sarah Wilson", "Female"),
            ("David Miller", "Male"), ("Jennifer Taylor", "Female"), ("James Anderson", "Male"),
            ("Lisa Thomas", "Female")
        ]
        
        patients = []
        conditions = [
            "Hypertension, High Cholesterol",
            "Type 2 Diabetes",
            "Migraine, Vertigo",
            "Asthma, Seasonal Allergies",
            "Healthy, Annual Checkup",
            "Post-surgery recovery",
            "Arthritis",
            "Anxiety, Insomnia"
        ]

        for i, (name, gender) in enumerate(patient_names, 1):
            # Create User for Patient
            first_name = name.split()[0].lower()
            last_name = name.split()[1].lower()
            username = f"{first_name}{last_name}"
            
            user = User(
                username=username,
                role='patient',
                name=name,
                department=None,
                email=f"{username}@example.com",
                phone=f'555-10{i:02d}',
                address=f'{i*10} Patient Rd'
            )
            user.set_password('password123')
            db.session.add(user)

            # Create Patient Record
            p = Patient(
                name=name,
                age=random.randint(20, 80),
                gender=gender,
                contact=f'555-10{i:02d}',
                medical_history=random.choice(conditions)
            )
            db.session.add(p)
            patients.append(p)
        
        db.session.commit()
        print(f"✅ Created {len(patients)} patients (User + Record).")

        print("🌱 Seeding Vitals, Labs, and Appointments...")
        
        # Refresh doctors and patients to get IDs
        doctors = User.query.filter_by(role='doctor').all()
        patients = Patient.query.all()

        for p in patients:
            # Vitals (3-5 records per patient)
            for _ in range(random.randint(3, 5)):
                v = Vital(
                    patient_id=p.id,
                    recorded_at=datetime.utcnow() - timedelta(days=random.randint(0, 30)),
                    bp=f"{random.randint(110, 140)}/{random.randint(70, 90)}",
                    heart_rate=random.randint(60, 100),
                    temperature=round(random.uniform(36.5, 37.5), 1),
                    weight=round(random.uniform(60, 100), 1)
                )
                db.session.add(v)

            # Appointments (1-2 per patient)
            doc = random.choice(doctors)
            appt = Appointment(
                patient_id=p.id,
                doctor_id=doc.id,
                date=datetime.utcnow() + timedelta(days=random.randint(-10, 10)),
                reason="Follow-up",
                status=random.choice(['scheduled', 'completed', 'cancelled'])
            )
            db.session.add(appt)

            # Lab Requests (1-2 per patient)
            lab = LabRequest(
                patient_id=p.id,
                doctor_id=doc.id,
                test_type=random.choice(['Blood Count', 'Lipid Panel', 'MRI', 'X-Ray']),
                status=random.choice(['pending', 'completed']),
                requested_at=datetime.utcnow() - timedelta(days=random.randint(1, 5))
            )
            if lab.status == 'completed':
                lab.result = "Normal limits" if random.random() > 0.3 else "Abnormal findings detected"
                lab.completed_at = datetime.utcnow()
            
            db.session.add(lab)

        db.session.commit()
        print("✅ Seeding complete!")

if __name__ == "__main__":
    seed_database()
