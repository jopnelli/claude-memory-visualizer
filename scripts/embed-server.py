#!/usr/bin/env python3
"""
Simple embedding server that uses the same model as claude-memory.
Runs on localhost:5001 and provides embeddings via HTTP.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from sentence_transformers import SentenceTransformer

# Load model once at startup (uses cached version from ~/.cache/huggingface)
print("Loading embedding model...")
model = SentenceTransformer('all-mpnet-base-v2')
print("Model loaded!")

class EmbeddingHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """Handle embedding request"""
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)

        text = data.get('text', '')
        if not text:
            self.send_error(400, 'Missing text field')
            return

        # Generate embedding
        embedding = model.encode(text, normalize_embeddings=True).tolist()

        # Send response
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({'embedding': embedding}).encode())

    def log_message(self, format, *args):
        # Quieter logging
        pass

if __name__ == '__main__':
    port = 5001
    server = HTTPServer(('localhost', port), EmbeddingHandler)
    print(f"Embedding server running on http://localhost:{port}")
    print("Using model: all-mpnet-base-v2 (same as claude-memory)")
    server.serve_forever()
