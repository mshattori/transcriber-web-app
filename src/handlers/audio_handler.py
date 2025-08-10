"""
Audio processing handler - manages audio transcription and translation.

Separates audio business logic from UI event handlers.
"""

import os
import json
import uuid
from typing import Dict, Any, Tuple, Optional, NamedTuple
from datetime import datetime

from transcribe import transcribe_chunked
from llm import translate_transcript_full
from util import validate_audio_file, estimate_processing_time, create_job_directory
from errors import ValidationError, get_user_friendly_message, safe_execute


class ProcessingResult(NamedTuple):
    """Result of audio processing."""
    transcript: str
    translation: str
    job_id: str
    settings_used: Dict[str, Any]


class AudioHandler:
    """Real audio processing handler."""
    
    def __init__(self):
        self.current_job_id: Optional[str] = None
        self.current_transcript: Optional[str] = None
        self.current_translation: Optional[str] = None
    
    def validate_audio(self, file_path: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validate audio file format, size, and properties.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, error_message, file_info)
        """
        return validate_audio_file(file_path)
    
    def validate_settings(self, settings: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate user settings.
        
        Args:
            settings: Settings dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        from app import validate_settings as _validate_settings
        return _validate_settings(settings)
    
    def estimate_processing_time(self, file_size_mb: float, chunk_minutes: int) -> Dict[str, Any]:
        """
        Estimate processing time based on file size and chunk duration.
        
        Args:
            file_size_mb: File size in megabytes
            chunk_minutes: Duration of each chunk in minutes
            
        Returns:
            Dictionary with time estimates
        """
        return estimate_processing_time(file_size_mb, chunk_minutes)
    
    async def process_audio(
        self, 
        audio_file: str,
        settings: Dict[str, Any],
        progress_callback=None
    ) -> ProcessingResult:
        """
        Process audio file with transcription and optional translation.
        
        Args:
            audio_file: Path to audio file
            settings: Processing settings
            progress_callback: Optional progress callback function
            
        Returns:
            ProcessingResult with transcript, translation, job_id, and settings
        """
        # Validate inputs
        if not audio_file:
            raise ValidationError("No audio file provided", field="audio_file")
        
        # Validate settings
        is_valid, error_msg = self.validate_settings(settings)
        if not is_valid:
            raise ValidationError(error_msg, field="settings")
        
        # Validate audio file
        is_valid_file, error_msg, file_info = self.validate_audio(audio_file)
        if not is_valid_file:
            raise ValidationError(f"Invalid audio file: {error_msg}", field="audio_file")
        
        # Create job directory
        job_id = str(uuid.uuid4())[:8]
        job_dir = create_job_directory(job_id)
        self.current_job_id = job_id
        
        # Estimate processing time
        time_estimates = self.estimate_processing_time(
            file_info['size_mb'], 
            settings['chunk_minutes']
        )
        
        if progress_callback:
            progress_callback(
                0.1, 
                f"Starting transcription ({time_estimates['estimated_chunks']} chunks expected)"
            )
        
        # Step 1-4: Transcription with chunked processing
        transcript_result = await transcribe_chunked(
            audio_path=audio_file,
            api_key=settings["api_key"],
            model=settings["audio_model"],
            language=settings["default_language"],
            chunk_minutes=settings["chunk_minutes"],
            temperature=0.0,
            include_timestamps=True,
            progress_callback=lambda p, m: progress_callback(0.1 + p * 0.6, m) if progress_callback else None
        )
        
        transcript_text = transcript_result.text
        self.current_transcript = transcript_text
        
        # Save original transcript
        def _save_transcript():
            transcript_path = os.path.join(job_dir, "transcript.txt")
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            return transcript_path
        
        safe_execute(_save_transcript, error_context="saving transcript")
        
        translation_text = ""
        
        # Translation if enabled
        if settings.get("translation_enabled", False):
            if progress_callback:
                progress_callback(0.7, "Starting translation...")
            
            translation_result = await translate_transcript_full(
                api_key=settings["api_key"],
                model=settings["language_model"],
                transcript_text=transcript_text,
                target_language=settings["default_translation_language"],
                temperature=0.3,
                progress_callback=lambda p, m: progress_callback(0.7 + p * 0.25, m) if progress_callback else None
            )
            
            translation_text = translation_result.translated_text
            self.current_translation = translation_text
            
            # Save translation
            def _save_translation():
                lang_code = settings["default_translation_language"].lower()[:2]
                translation_path = os.path.join(job_dir, f"transcript.{lang_code}.txt")
                with open(translation_path, 'w', encoding='utf-8') as f:
                    f.write(translation_text)
            
            safe_execute(_save_translation, error_context="saving translation")
        
        # Save job metadata
        def _save_metadata():
            job_metadata = {
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "original_filename": os.path.basename(audio_file),
                "file_info": file_info,
                "settings": settings,
                "transcript_stats": {
                    "word_count": transcript_result.word_count,
                    "duration": transcript_result.total_duration,
                    "processing_time": transcript_result.processing_time
                },
                "translation_enabled": settings.get("translation_enabled", False)
            }
            
            metadata_path = os.path.join(job_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(job_metadata, f, indent=2, ensure_ascii=False)
        
        safe_execute(_save_metadata, error_context="saving job metadata")
        
        if progress_callback:
            progress_callback(1.0, "Processing completed!")
        
        return ProcessingResult(
            transcript=transcript_text,
            translation=translation_text,
            job_id=job_id,
            settings_used=settings
        )


class MockAudioHandler:
    """Mock audio processing handler for UI testing."""
    
    def __init__(self):
        self.current_job_id: Optional[str] = "mock-job-123"
        self.current_transcript: Optional[str] = "Mock transcript content"
        self.current_translation: Optional[str] = "Mock translation content"
    
    def validate_audio(self, file_path: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Mock audio validation - always returns valid."""
        return True, None, {
            'size_mb': 2.5,
            'duration_seconds': 150.0,
            'format': 'mp3',
            'sample_rate': 44100,
            'needs_warning': False
        }
    
    def validate_settings(self, settings: Dict[str, Any]) -> Tuple[bool, str]:
        """Mock settings validation - always returns valid."""
        return True, ""
    
    def estimate_processing_time(self, file_size_mb: float, chunk_minutes: int) -> Dict[str, Any]:
        """Mock processing time estimation."""
        return {
            'estimated_chunks': 2,
            'upload_time_seconds': 1.0,
            'processing_time_seconds': 5.0,
            'total_time_seconds': 6.0,
            'total_time_minutes': 0.1
        }
    
    async def process_audio(
        self, 
        audio_file: str,
        settings: Dict[str, Any],
        progress_callback=None
    ) -> ProcessingResult:
        """Mock audio processing - returns instant results."""
        import asyncio
        
        # Simulate progress updates
        if progress_callback:
            progress_callback(0.1, "Starting mock transcription...")
            await asyncio.sleep(0.1)
            
            progress_callback(0.5, "Processing mock chunks...")
            await asyncio.sleep(0.1)
            
            if settings.get("translation_enabled", False):
                progress_callback(0.7, "Starting mock translation...")
                await asyncio.sleep(0.1)
            
            progress_callback(1.0, "Mock processing completed!")
        
        # Generate mock content
        mock_transcript = f"""# 00:00:00 --> 00:02:30
This is a mock transcript generated for UI testing purposes.
The original audio file was: {os.path.basename(audio_file)}

# 00:02:30 --> 00:05:00  
This mock includes timestamp formatting and multiple segments
to simulate real transcription output for testing purposes.
"""
        
        mock_translation = ""
        if settings.get("translation_enabled", False):
            mock_translation = f"""# 00:00:00 --> 00:02:30
これはUIテスト用のモック翻訳です。
元の音声ファイルは: {os.path.basename(audio_file)}

# 00:02:30 --> 00:05:00
このモックには、テスト用の実際の翻訳出力をシミュレートするために
タイムスタンプフォーマットと複数のセグメントが含まれています。
"""
        
        return ProcessingResult(
            transcript=mock_transcript,
            translation=mock_translation,
            job_id=self.current_job_id,
            settings_used=settings
        )