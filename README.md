# J.A.R.V.I.S. - Just A Rather Very Intelligent System

A complete local AI assistant built with Python, featuring voice interaction, multimodal vision, web search, and smart home integration. No API keys required - everything runs locally on your machine.

## Features

### 🎤 Voice Pipeline
- **Wake Word Detection**: Responds to "Hey Jarvis" using OpenWakeWord
- **Speech-to-Text**: Real-time transcription with faster-whisper
- **Text-to-Speech**: Natural voice synthesis with Coqui XTTS v2

### 🧠 Brain & Memory
- **Local LLM**: Powered by Ollama + Llama 3.1 8B (8GB RAM required)
- **Two-Layer Memory**: Vector storage for conversations + structured facts
- **Context Awareness**: Remembers previous interactions and user preferences

### 🔧 Agent & Tools
- **Tool Calling**: 15+ built-in tools (web search, weather, file operations, etc.)
- **Smart Home**: Home Assistant integration for device control
- **Code Execution**: Run Python code and shell commands safely

### 👁️ Vision & Multimodal
- **Image Analysis**: Describe photos and analyze visual content with LLaVA
- **Camera Integration**: Real-time environment monitoring
- **OCR Support**: Extract text from images

### 🌐 API Backend
- **REST API**: Full HTTP endpoints for all functionality
- **WebSocket Streaming**: Real-time chat with streaming responses
- **Frontend Ready**: Easy integration with web/mobile apps

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama & Models
```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.1:8b          # Main LLM (8GB RAM)
ollama pull llava:7b             # Vision model (4GB RAM)
```

### 3. Run J.A.R.V.I.S.
```bash
# Text mode (testing)
python run.py --mode text

# Voice mode (full assistant)
python run.py --mode voice --location "New York, USA"

# API mode (for frontend)
python run.py --mode api
```

## Configuration

### Command Line Options
- `--mode`: `text`, `voice`, or `api`
- `--location`: Your location for weather (e.g., "New York, USA")
- `--ha-url`: Home Assistant URL (optional)
- `--ha-token`: Home Assistant access token (optional)

### Environment Variables
Create a `.env` file for sensitive data:
```
OPENWEATHER_API_KEY=your_key_here
HOME_ASSISTANT_URL=http://localhost:8123
HOME_ASSISTANT_TOKEN=your_token_here
```

## Architecture

```
jarvis/
├── voice/           # Voice pipeline components
│   ├── wake_word.py # Wake word detection
│   ├── stt.py       # Speech-to-text
│   ├── tts.py       # Text-to-speech
│   └── pipeline.py  # Voice orchestration
├── brain/           # LLM and prompts
│   ├── llm_client.py
│   └── prompts.py
├── memory/          # Vector + structured storage
│   └── manager.py
├── agent/           # ReAct agent loop
│   ├── agent.py
│   ├── tools.py
│   └── tool_executor.py
├── vision/          # Computer vision
│   └── vision_module.py
├── api/             # FastAPI backend
│   └── server.py
└── run.py          # Main entry point
```

## API Endpoints

### REST API
- `GET /health` - Health check
- `POST /chat` - Text chat
- `POST /voice/chat` - Voice interaction
- `GET /memory` - Retrieve memories
- `POST /vision/analyze` - Analyze image

### WebSocket
- `ws://localhost:8000/ws/chat` - Streaming chat

## Tool Capabilities

J.A.R.V.I.S. can perform these actions:

1. **Web Search**: DuckDuckGo search with summaries
2. **Weather**: Current conditions and forecasts
3. **File Operations**: Read/write files, list directories
4. **Code Execution**: Run Python/shell commands
5. **Home Assistant**: Control smart home devices
6. **Vision**: Analyze images and camera feed
7. **Memory**: Store and recall facts
8. **Time/Date**: Current time and calendar
9. **Calculator**: Mathematical computations
10. **System Info**: Hardware and OS details

## Hardware Requirements

- **RAM**: 16GB minimum (24GB recommended)
- **Storage**: 20GB free space for models
- **Microphone**: For voice input
- **Speakers/Headphones**: For voice output
- **Camera**: Optional for vision features

## Troubleshooting

### Common Issues

**Ollama not found**
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve
```

**Audio device errors**
```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

**Memory issues**
- Reduce ChromaDB collection size in memory/manager.py
- Use smaller LLM models if RAM is limited

### Logs
All components use prefixed logging:
- `[WakeWord]`: Wake word detection
- `[STT]`: Speech-to-text
- `[TTS]`: Text-to-speech
- `[Agent]`: Agent reasoning
- `[Run]`: Main application

## Development

### Testing
```bash
# Test individual components
python -c "from jarvis.brain.llm_client import LLMClient; print('LLM OK')"

# Run API server for testing
python run.py --mode api
```

### Extending J.A.R.V.I.S.

**Add new tools**:
1. Define tool schema in `jarvis/agent/tools.py`
2. Implement in `jarvis/agent/tool_executor.py`
3. Add to agent prompt if needed

**Add new voice features**:
1. Extend `jarvis/voice/` modules
2. Update `jarvis/voice/pipeline.py`

**Add new API endpoints**:
1. Add routes in `jarvis/api/server.py`
2. Update OpenAPI documentation

## License

MIT License - feel free to modify and distribute.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Built with ❤️ using Python 3.8+**