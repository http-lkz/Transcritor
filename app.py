from flask import Flask, request, jsonify, render_template
import os
import socket
from werkzeug.utils import secure_filename
import torch
from transformers import pipeline
import librosa
import numpy as np
from pathlib import Path

# Configurações do servidor
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = r'P:\Eduardo Silveira\uploads'  # Caminho absoluto na rede
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limite de 50MB

# Garante que a pasta de uploads existe
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.lower().endswith(tuple(app.config['ALLOWED_EXTENSIONS']))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de arquivo não permitido'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=device
        )
        
        audio, sr = librosa.load(filepath, sr=16000)
        result = model(
            audio.astype(np.float32),
            generate_kwargs={"language": "portuguese", "task": "transcribe"},
            chunk_length_s=30
        )
        
        os.remove(filepath)
        return jsonify({
            'text': result['text'],
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    print("\n=== Servidor de Transcrição de Áudio ===")
    print(f"Endereços de acesso:")
    print(f"→ Este computador: http://localhost:5000")
    print(f"→ Na rede corporativa: http://10.10.40.62:5000")
    print("\nAguardando conexões... (Ctrl+C para encerrar)\n")
    
    # Permite conexões de qualquer host na rede
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)