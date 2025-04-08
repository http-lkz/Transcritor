# 🎙️ Descritor de Áudios

Sistema web para **transcrição automática de áudios** com notificações em tempo real, usando o modelo Whisper da OpenAI.

## 🚀 Funcionalidades

- Upload de arquivos de áudio (.wav, .mp3, .ogg)
- Transcrição automática usando o modelo `openai/whisper-small`
- Notificação sonora de conclusão via WebSocket (Socket.IO)
- Geração de senhas dinâmicas baseadas na data e hora

## 📦 Tecnologias e Dependências

- Python 3.8+
- [Flask](https://flask.palletsprojects.com/)
- [Socket.IO](https://python-socketio.readthedocs.io/)
- [Transformers](https://huggingface.co/docs/transformers/index)
- [Torch](https://pytorch.org/)
- [Librosa](https://librosa.org/)
- [NumPy](https://numpy.org/)

### Instalação

```bash
pip install flask python-socketio[client] torch transformers librosa numpy
