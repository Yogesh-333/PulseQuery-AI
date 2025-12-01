import sqlite3

def inspect_sqlite():
    try:
        conn = sqlite3.connect('pulsequery.db')
        cursor = conn.cursor()
        
        print("--- Users ---")
        cursor.execute("SELECT id, username, role, name FROM user")
        users = cursor.fetchall()
        for user in users:
            print(user)
            
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_sqlite()
