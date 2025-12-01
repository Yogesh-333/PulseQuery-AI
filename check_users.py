
import sqlite3
import os

def check_users():
    print(f"CWD: {os.getcwd()}")
    db_path = 'pulsequery.db'
    print(f"DB Path: {os.path.abspath(db_path)}")
    print(f"DB Exists: {os.path.exists(db_path)}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # List tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables:", tables)
        
        cursor.execute("PRAGMA table_info(user)")
        columns = cursor.fetchall()
        print("Columns in user table:")
        for col in columns:
            print(col)
            
        cursor.execute("SELECT * FROM user")
        users = cursor.fetchall()
        print("Users in DB (user table):")
        for user in users:
            print(user)
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_users()
