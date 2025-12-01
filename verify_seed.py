from app import create_app
from models import db, User, Patient, Vital, LabRequest, Appointment

def verify_data():
    app = create_app()
    with app.app_context():
        print(f"Users: {User.query.count()}")
        print(f"Patients: {Patient.query.count()}")
        print(f"Vitals: {Vital.query.count()}")
        print(f"Lab Requests: {LabRequest.query.count()}")
        print(f"Appointments: {Appointment.query.count()}")

if __name__ == "__main__":
    verify_data()
