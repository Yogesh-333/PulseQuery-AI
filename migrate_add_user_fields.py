import sqlite3
import os

DB_PATH = 'instance/pulsequery.db'

def migrate_db():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check existing columns
        cursor.execute("PRAGMA table_info(user)")
        columns = [col[1] for col in cursor.fetchall()]
        
        new_columns = {
            'email': 'VARCHAR(120)',
            'phone': 'VARCHAR(20)',
            'address': 'VARCHAR(200)'
        }
        
        for col_name, col_type in new_columns.items():
            if col_name not in columns:
                print(f"🔄 Adding column '{col_name}'...")
                cursor.execute(f"ALTER TABLE user ADD COLUMN {col_name} {col_type}")
                print(f"✅ Column '{col_name}' added.")
            else:
                print(f"ℹ️ Column '{col_name}' already exists.")
                
        conn.commit()
        print("✅ Migration complete.")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_db()
