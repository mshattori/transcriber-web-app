# Test Data Directory

This directory contains test data files for the transcriber web app testing.

## Test Audio Files

### Available Test Files:
- `test_audio.mp3` - English sample audio file (provided)

### Creating Test Audio Files:

You can create simple test audio files using various methods:

#### Method 1: Using ffmpeg (recommended)
```bash
# Create a 10-second sine wave audio file
ffmpeg -f lavfi -i "sine=frequency=440:duration=10" -ar 22050 test_audio_small.wav

# Create a 30-second white noise audio file
ffmpeg -f lavfi -i "anoisesrc=duration=30:color=white" -ar 22050 test_audio_medium.wav
```

#### Method 2: Using online text-to-speech
1. Visit a text-to-speech service (e.g., Google TTS, Amazon Polly)
2. Generate audio from sample text in different languages
3. Download as WAV files

#### Method 3: Record your own
1. Use any audio recording software
2. Record short samples in different languages
3. Save as WAV format

### Test Audio Content Suggestions:

**English sample text:**
"Hello, this is a test audio file for the transcriber web application. The quick brown fox jumps over the lazy dog."

**Japanese sample text:**  
"こんにちは、これは文字起こしウェブアプリケーションのテスト音声ファイルです。日本語の音声認識をテストしています。"

## Configuration Files

- `test_config.yaml` - Test-specific configuration settings
- Contains model lists, language settings, and test parameters

## Usage in Tests

The main test file `test_audio.mp3` is used by default for all tests. Additional test files are optional and tests will gracefully skip if not found.

Example test usage:
```python
from src.config.test_config import TestConfig

config = TestConfig()
# Uses test_audio.mp3 by default
main_audio = config.get_test_audio_file()  # tests/data/test_audio.mp3

# Optional additional files
small_audio = config.get_test_audio_file("small")  # tests/data/test_audio_small.wav
```

### Test Configuration
The `test_config.yaml` file references `test_audio.mp3` as the primary test file. All handlers and integration tests are designed to work with this single English sample audio file.