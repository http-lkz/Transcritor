from flask import Flask, request, jsonify, render_template
import os
import socketio  # Adicionado para notificações em tempo real
from werkzeug.utils import secure_filename
import torch
from transformers import pipeline
import librosa
import numpy as np
from pathlib import Path
from datetime import datetime

# Configurações do servidor
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = r'P:\Eduardo Silveira\uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['SECRET_KEY'] = 'sua_chave_secreta_aqui'  # Necessário para SocketIO

# Configuração do Socket.IO
sio = socketio.Server(async_mode='threading')
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

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
        
        # Envia notificação de conclusão para todos os clientes
        sio.emit('play_sound', {'message': 'Transcrição concluída!'}, namespace='/')
        
        return jsonify({
            'text': result['text'],
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/generate_password', methods=['GET'])
def generate_password():
    now = datetime.now()
    day = now.day
    month = now.month
    hour = now.hour
    
    p1 = 20 + day
    p2 = 11 + month
    p3 = 40 + hour
    
    password = f"{p1:02d}{p2:02d}{p3:02d}"
    return jsonify({'password': password})

# Nova rota para acionar notificações
@app.route('/trigger_notification', methods=['POST'])
def trigger_notification():
    try:
        # Envia notificação para todos os clientes conectados
        sio.emit('play_sound', {'message': 'Notificação do servidor!'}, namespace='/')
        return jsonify({'status': 'success', 'message': 'Notificação enviada'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@sio.on('connect')
def handle_connect(sid, environ):
    print(f'Cliente conectado: {sid}')

@sio.on('disconnect')
def handle_disconnect(sid):
    print(f'Cliente desconectado: {sid}')

if __name__ == '__main__':
    print("\n=== Servidor de Transcrição de Áudio ===")
    print("Recursos implementados:")
    print("- Transcrição de áudio com Whisper")
    print("- Geração de senhas dinâmicas")
    print("- Sistema de notificação sonora em tempo real")
    print("\nEndereços de acesso:")
    print("→ Este computador: http://localhost:5000")
    print("→ Na rede corporativa: http://10.10.40.62:5000")
    print("\nAguardando conexões... (Ctrl+C para encerrar)\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
