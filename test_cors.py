
try:
    from flask import Flask
    from flask_cors import CORS
    print("SUCCESS: flask and flask_cors imported")
except Exception as e:
    print(f"FAILURE: {e}")
