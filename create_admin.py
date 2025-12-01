from app import create_app
from models import db, User

def create_admin_user():
    app = create_app()
    with app.app_context():
        # Check if admin exists
        admin = User.query.filter_by(username='admin_manual').first()
        if admin:
            print("Admin user 'admin_manual' already exists.")
            # Reset password just in case
            admin.set_password('admin123')
            db.session.commit()
            print("Password reset to 'admin123'.")
        else:
            admin = User(
                username='admin_manual',
                role='admin',
                name='Manual Admin',
                department='IT',
                email='admin@example.com',
                phone='555-555-5555',
                address='123 Admin St'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Admin user 'admin_manual' created successfully.")

if __name__ == "__main__":
    create_admin_user()
