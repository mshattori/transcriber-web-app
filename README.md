# Transcriber Web App

A production-ready web application for transcribing large audio files using OpenAI's Whisper API, with support for translation and interactive chat functionality. Built with Gradio for an intuitive user interface.

## Features

- **Large File Support**: Automatically chunks audio files up to 1 hour in length to bypass OpenAI's 25MB API limit
- **Real-time Progress**: Live progress updates during transcription with chunk-by-chunk processing
- **Multi-language Translation**: Structured JSON-based translation with OpenAI's structured outputs
- **Interactive Chat**: Context-aware chat interface using transcription as context
- **Job Management**: Persistent job history with downloadable results
- **Browser Settings**: Automatic settings persistence using localStorage
- **Comprehensive Error Handling**: User-friendly error messages with detailed logging
- **Memory Efficient**: Uses `tempfile.SpooledTemporaryFile` for large file processing
- **Environment-based Configuration**: Support for production, testing, and mock UI modes
- **Testable Architecture**: Complete separation of UI and business logic with mock handlers

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key
- FFmpeg (for audio processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd transcriber-web-app
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the application**
   - Copy `sample.env` to `.env` and add your OpenAI API key
   - The application uses `src/config.yaml` for default settings
   - You can also add API key through the web interface settings modal

### Running the Application

```bash
# Activate virtual environment and start the application
source venv/bin/activate

# Production mode (default)
cd src && python app.py

# Or with explicit environment setting
APP_ENV=prod python src/app.py

# Mock UI mode (for UI testing without API calls)
APP_ENV=mock-ui python src/app.py

# Test mode (real APIs with test configuration)
APP_ENV=test python src/app.py
```

The application will start on `http://localhost:7860` by default.

## Usage

1. **Upload Audio File**: Support for MP3, WAV, M4A, FLAC, OGG formats
2. **Configure Settings**: Set API key, models, language preferences
3. **Start Transcription**: Real-time progress with chunk processing
4. **Optional Translation**: Enable translation to target language
5. **Interactive Chat**: Ask questions about the transcribed content
6. **Download Results**: Get transcripts and translations as files

## Architecture

### Design Principles

The application follows a handler-based architecture with complete separation of UI and business logic:

- **UI/Logic Separation**: Gradio UI event handlers are completely separate from business logic
- **Testable Architecture**: Business logic is testable with dependency injection and mock handlers
- **Environment-based Configuration**: Support for production, testing, and mock UI modes
- **Modular Design**: Clear separation of concerns with dedicated handlers for different features

### Project Structure

```
transcriber-web-app/
├── src/                    # Source code
│   ├── app.py             # Main Gradio application with environment support
│   ├── handlers/          # Business logic handlers (separated from UI)
│   │   ├── audio_handler.py     # Audio processing logic
│   │   ├── chat_handler.py      # Chat functionality
│   │   ├── history_handler.py   # Job history management
│   │   └── settings_handler.py  # Settings management
│   ├── config/            # Configuration management
│   │   ├── app_config.py        # Application configuration
│   │   └── test_config.py       # Test-specific configuration
│   ├── transcribe.py      # Audio transcription with chunking
│   ├── llm.py             # LLM utilities and translation
│   ├── util.py            # Audio processing utilities
│   ├── errors.py          # Error handling system
│   └── config.yaml        # Configuration file
├── tests/                 # Unit tests
│   ├── data/             # Test data and configuration
│   │   ├── test_audio.mp3      # English sample audio file
│   │   └── test_config.yaml    # Test configuration
│   ├── unit/             # Unit tests for handlers
│   ├── integration/      # Integration tests
│   └── ui/              # UI-only tests with mocks
├── data/                  # Job persistence (auto-created)
│   └── {YYYY-MM-DD}/     # Date-based organization
│       └── {job_id}/     # Individual job folders
├── sample.env            # Environment configuration template
├── requirements.txt       # Python dependencies
├── PLANNING.md           # Architecture and implementation plan
└── README.md             # This file
```

### Core Components

#### 1. UI/Logic Separation (`handlers/`)
- **AudioHandler**: Business logic for audio processing and transcription
- **ChatHandler**: Context-aware chat functionality
- **HistoryHandler**: Job history and transcript loading
- **SettingsHandler**: Configuration management and validation
- **Mock Handlers**: Testing implementations for UI-only testing

#### 2. Environment Configuration (`config/`)
- **AppConfig**: Environment-based configuration with .env override
- **TestConfig**: Test-specific settings with mock data support
- **Environment modes**: prod, test, mock-ui for different use cases

#### 3. Audio Processing (`util.py`)
- **Memory-efficient chunking**: Uses `tempfile.SpooledTemporaryFile`
- **Overlap handling**: 2-second overlap between chunks for seamless merging
- **Format validation**: Supports multiple audio formats with size warnings
- **Naming convention**: `chunk_01.mp3`, `chunk_02.mp3`, etc.

#### 4. Transcription Engine (`transcribe.py`)
- **4-step processing**: Split → Transcribe → Real-time updates → Translation
- **Async processing**: With exponential backoff retry logic
- **Progress callbacks**: Real-time UI updates via `gr.Progress`
- **CLI interface**: For testing and batch processing

#### 5. LLM Integration (`llm.py`)
- **Structured outputs**: JSON-based translation with OpenAI's structured responses
- **Chat context injection**: 3-message pattern (system → context → question)
- **Translation workflow**: Parse → Translate → Reconstruct
- **Language mapping**: Configurable language code mappings

#### 6. Web Interface (`app.py`)
- **Environment-based handlers**: Production/test/mock mode support
- **Single-column layout**: File upload → Controls → Progress → Results → Chat
- **Browser state management**: Settings persistence in localStorage
- **Job history modal**: Access to previous transcriptions
- **Error handling**: User-friendly error messages throughout

#### 7. Error Management (`errors.py`)
- **Custom exception hierarchy**: Typed errors for different failure modes
- **User-friendly messages**: Technical errors converted to actionable messages
- **Safe execution**: Automatic error catching and conversion utilities
- **Comprehensive validation**: API keys, file formats, input parameters

### Data Flow

1. **File Upload** → Validation → Size/format checks
2. **Audio Processing** → Chunking → Temporary file management
3. **Transcription** → Parallel chunk processing → Progress updates
4. **Translation** (optional) → JSON parsing → Structured translation
5. **Results** → File persistence → Download preparation
6. **Chat** → Context injection → Interactive Q&A

### Configuration

#### Configuration Override Pattern

The application uses a two-tier configuration system for secure credential management:

```python
# Load base config from YAML
base_config = load_yaml("config.yaml")

# Override with environment variables for sensitive data
if os.getenv("OPENAI_API_KEY"):
    base_config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
```

This allows secure credential management while maintaining default settings in version control.

#### Environment Configuration
- **`.env` file**: API keys and sensitive configuration
- **`APP_ENV` variable**: Controls application mode (prod/test/mock-ui)
- **Environment override**: .env values override config.yaml defaults

#### Application Configuration (`config.yaml`)
- **Audio models**: Available Whisper models
- **Language models**: Available GPT models for translation/chat
- **System message**: Default prompt for LLM interactions
- **Language mappings**: Translation target languages
- **Processing settings**: Chunk duration, file size limits

#### Test Configuration (`tests/data/test_config.yaml`)
- **Test-specific settings**: Smaller limits, mock delays
- **Test audio files**: Reference to `test_audio.mp3` (English sample)

## Development

### Running Tests

#### Test Setup
1. **Test audio file**: Place `test_audio.mp3` (English sample) in `tests/data/`
2. **Environment**: Copy `sample.env` to `.env` and add your OpenAI API key for integration tests
3. **Mock tests**: UI-only tests use mock handlers and don't require API keys

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run unit tests with mock handlers (no API required)
APP_ENV=mock-ui python -m pytest tests/unit/ -v

# Run integration tests with real APIs (requires API key)
APP_ENV=test python -m pytest tests/integration/ -v

# Run UI tests with mocks
APP_ENV=mock-ui python -m pytest tests/ui/ -v

# Run specific test file
python -m pytest tests/test_util.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Structure

- **Unit tests** (`tests/unit/`): Tests for individual handlers and modules
- **Integration tests** (`tests/integration/`): End-to-end tests with real APIs
- **UI tests** (`tests/ui/`): UI-only tests using mock handlers
- **Test data** (`tests/data/`): Test audio files and configuration
- **Mocking**: External APIs (OpenAI, file system) are mocked for unit tests
- **Fixtures**: Shared test data and mock objects in `conftest.py`
- **Test categories**: Handler logic, API interactions, file processing, error handling

### Code Quality

```bash
# Linting and formatting
ruff check src/ tests/
ruff format src/ tests/

# Type checking
mypy src/

# All quality checks
ruff check src/ && mypy src/
```

### CLI Interfaces

Each module provides a CLI interface for testing:

```bash
# Test transcription with test audio
python src/transcribe.py tests/data/test_audio.mp3 --model whisper-1

# Test translation
python src/llm.py --help

# Test configuration loading
python -c "import sys; sys.path.append('src'); from config import AppConfig; print(AppConfig().get_all())"

# Test handlers directly
python -c "
import sys; sys.path.append('src');
from handlers import MockAudioHandler;
handler = MockAudioHandler();
print('Mock handler ready for testing')
"
```

### Development Guidelines

1. **Architecture**: Separate UI logic from business logic using handlers
2. **Environment**: Use environment variables for configuration (`APP_ENV`)
3. **Import Structure**: Use absolute imports within the `src/` package
4. **Error Handling**: Always use custom exceptions from `errors.py`
5. **Configuration**: Load settings via `AppConfig` with .env override support
6. **Testing**: Write tests for new functionality with appropriate mocking
7. **Documentation**: Include docstrings using Google style format

### Adding New Features

1. **Handler Pattern**: Create both real and mock implementations
2. **Environment Support**: Ensure new features work in all modes (prod/test/mock-ui)
3. **Error Handling**: Add new error types to `errors.py` if needed
4. **Configuration**: Update `config.yaml` and environment settings
5. **Testing**: Add unit tests (mocked), integration tests (real APIs), and UI tests
6. **Documentation**: Update this README, PLANNING.md, and add docstrings

## API Integration

### OpenAI Configuration

The application integrates with:
- **Whisper API**: For speech-to-text transcription
- **GPT Models**: For translation and chat functionality
- **Structured Outputs**: For JSON-formatted translation results

### Rate Limiting

- **Exponential backoff**: Automatic retry with increasing delays
- **Error handling**: Graceful degradation for API failures
- **User feedback**: Clear error messages for API issues

## Deployment

### Production Considerations

1. **Environment Configuration**: Use `.env` file with `APP_ENV=prod`
2. **API Keys**: Store OpenAI API key securely in environment variables
3. **File Storage**: Configure persistent storage for job data in `/data` directory
4. **Monitoring**: Add logging for production debugging
5. **Security**: Validate file uploads and sanitize inputs
6. **Performance**: Monitor memory usage for large files
7. **Testing**: Use `APP_ENV=mock-ui` for UI testing without API calls

### Docker Support

```dockerfile
# Example Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY sample.env .env

# Set environment variables
ENV APP_ENV=prod
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python", "src/app.py"]
```

### Environment Setup Examples

```bash
# Production deployment
echo "OPENAI_API_KEY=your_key_here" > .env
echo "APP_ENV=prod" >> .env
python src/app.py

# Development with mocks
echo "APP_ENV=mock-ui" > .env
python src/app.py  # No API key required

# Testing setup
echo "OPENAI_API_KEY=your_key_here" > .env
echo "APP_ENV=test" >> .env
echo "TEST_AUDIO_FILE=tests/data/test_audio.mp3" >> .env
python src/app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and code quality checks succeed
5. Submit a pull request

## License

[Add appropriate license information]

## Support

For issues and questions:
1. Check the error messages in the web interface
2. Review the console output for detailed error information
3. Consult the test files for usage examples
4. Open an issue with reproduction steps