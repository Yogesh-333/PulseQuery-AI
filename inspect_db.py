from app import app, db
from models import User

def inspect_users():
    with app.app_context():
        users = User.query.all()
        print(f"Found {len(users)} users:")
        for user in users:
            print(f" - ID: {user.id}, Username: {user.username}, Role: {user.role}, Name: {user.name}")

if __name__ == "__main__":
    inspect_users()
