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
   - The application uses `src/config.yaml` for default settings
   - Add your OpenAI API key through the web interface settings modal

### Running the Application

```bash
# Activate virtual environment and start the application
source venv/bin/activate
cd src && python app.py
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

### Project Structure

```
transcriber-web-app/
├── src/                    # Source code
│   ├── app.py             # Main Gradio application
│   ├── transcribe.py      # Audio transcription with chunking
│   ├── llm.py             # LLM utilities and translation
│   ├── util.py            # Audio processing utilities
│   ├── errors.py          # Error handling system
│   └── config.yaml        # Configuration file
├── tests/                 # Unit tests
├── data/                  # Job persistence (auto-created)
│   └── {YYYY-MM-DD}/     # Date-based organization
│       └── {job_id}/     # Individual job folders
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Core Components

#### 1. Audio Processing (`util.py`)
- **Memory-efficient chunking**: Uses `tempfile.SpooledTemporaryFile`
- **Overlap handling**: 2-second overlap between chunks for seamless merging
- **Format validation**: Supports multiple audio formats with size warnings
- **Naming convention**: `chunk_01.mp3`, `chunk_02.mp3`, etc.

#### 2. Transcription Engine (`transcribe.py`)
- **4-step processing**: Split → Transcribe → Real-time updates → Translation
- **Async processing**: With exponential backoff retry logic
- **Progress callbacks**: Real-time UI updates via `gr.Progress`
- **CLI interface**: For testing and batch processing

#### 3. LLM Integration (`llm.py`)
- **Structured outputs**: JSON-based translation with OpenAI's structured responses
- **Chat context injection**: 3-message pattern (system → context → question)
- **Translation workflow**: Parse → Translate → Reconstruct
- **Language mapping**: Configurable language code mappings

#### 4. Web Interface (`app.py`)
- **Single-column layout**: File upload → Controls → Progress → Results → Chat
- **Browser state management**: Settings persistence in localStorage
- **Job history modal**: Access to previous transcriptions
- **Error handling**: User-friendly error messages throughout

#### 5. Error Management (`errors.py`)
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

The `config.yaml` file defines:
- **Audio models**: Available Whisper models
- **Language models**: Available GPT models for translation/chat
- **System message**: Default prompt for LLM interactions
- **Language mappings**: Translation target languages
- **Processing settings**: Chunk duration, file size limits

## Development

### Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_util.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Structure

- **Unit tests**: 95 tests covering all modules
- **Mocking**: External APIs (OpenAI, file system) are mocked
- **Fixtures**: Shared test data and mock objects in `conftest.py`
- **Test categories**: Data models, API interactions, file processing, error handling

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
# Test transcription
python src/transcribe.py --help

# Test translation
python src/llm.py --help

# Test configuration
python -c "import sys; sys.path.append('src'); import util; print(util.load_config())"
```

### Development Guidelines

1. **Import Structure**: Use absolute imports within the `src/` package
2. **Error Handling**: Always use custom exceptions from `errors.py`
3. **Configuration**: Load settings via `util.load_config()`
4. **Testing**: Write tests for new functionality with appropriate mocking
5. **Documentation**: Include docstrings using Google style format

### Adding New Features

1. **Module Structure**: Follow the existing pattern of separation of concerns
2. **Error Handling**: Add new error types to `errors.py` if needed
3. **Configuration**: Update `config.yaml` for new settings
4. **Testing**: Add corresponding unit tests with mocks
5. **Documentation**: Update this README and add docstrings

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

1. **Environment Variables**: Store API keys securely
2. **File Storage**: Configure persistent storage for job data
3. **Monitoring**: Add logging for production debugging
4. **Security**: Validate file uploads and sanitize inputs
5. **Performance**: Monitor memory usage for large files

### Docker Support

```dockerfile
# Example Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
EXPOSE 7860

CMD ["python", "src/app.py"]
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