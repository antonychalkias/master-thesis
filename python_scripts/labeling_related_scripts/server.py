from flask import Flask, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS on all routes

BASE_DIR = "/Users/chalkiasantonios/Desktop/master-thesis"

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory(BASE_DIR, filename)

if __name__ == "__main__":
    app.run(port=8000, debug=True)
# This code serves files from the specified base directory using Flask.
# It enables CORS for all routes, allowing cross-origin requests.
# The server listens on port 8000 and serves files from the base directory when accessed via a URL.
# The `serve_file` function uses Flask's `send_from_directory` to serve files from the specified base directory.