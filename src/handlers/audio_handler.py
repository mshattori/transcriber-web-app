"""
Audio processing handler - manages audio transcription and translation.

Separates audio business logic from UI event handlers.
"""

import os
import uuid
from typing import Any, NamedTuple

from ..errors import ValidationError, safe_execute
from ..file_manager import save_job_metadata, save_transcription_files
from ..integrated_display import get_display_content_for_ui
from ..llm import translate_transcript_full
from ..transcribe import transcribe_chunked
from ..util import create_job_directory, estimate_processing_time, validate_audio_file


class ProcessingResult(NamedTuple):
    """Result of audio processing."""
    transcript: str
    translation: str
    job_id: str
    settings_used: dict[str, Any]
    display_text: str  # 新規追加: UI表示用テキスト


class AudioHandler:
    """Real audio processing handler."""

    def __init__(self):
        self.current_job_id: str | None = None
        self.current_transcript: str | None = None
        self.current_translation: str | None = None

    def get_display_content(self) -> tuple[str, str, bool]:
        """
        表示用コンテンツの取得
        
        Returns:
            Tuple of (transcript, translation, translation_available)
        """
        return (
            self.current_transcript or "",
            self.current_translation or "",
            bool(self.current_translation)
        )

    def get_ui_display_text(self) -> str:
        """
        UI表示用テキストの取得
        
        Returns:
            統合表示形式のテキスト
        """
        return get_display_content_for_ui(
            self.current_transcript or "",
            self.current_translation or ""
        )

    def validate_audio(self, file_path: str) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Validate audio file format, size, and properties.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, error_message, file_info)
        """
        return validate_audio_file(file_path)

    def validate_settings(self, settings: dict[str, Any]) -> tuple[bool, str]:
        """
        Validate user settings.
        
        Args:
            settings: Settings dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        from app import validate_settings as _validate_settings
        return _validate_settings(settings)

    def estimate_processing_time(self, file_size_mb: float, chunk_minutes: int) -> dict[str, Any]:
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
        settings: dict[str, Any],
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
            progress_callback=lambda p, m: progress_callback(0.1 + p * 0.6, m) if progress_callback else None,
            job_dir=job_dir
        )

        transcript_text = transcript_result.text
        self.current_transcript = transcript_text

        translation_text = ""

        # Translation if enabled
        translation_error = None
        if settings.get("translation_enabled", False):
            if progress_callback:
                progress_callback(0.7, "Starting translation...")

            try:
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

            except Exception as e:
                # Handle translation failure gracefully
                from errors import handle_translation_failure
                transcript_text, translation_text, translation_error = handle_translation_failure(
                    transcript_text, e
                )
                self.current_translation = translation_text

                if progress_callback:
                    progress_callback(0.95, "Translation failed, but transcription completed successfully")

        # Save all file formats (transcript, translation, integrated)
        def _save_all_files():
            saved_files = save_transcription_files(
                job_dir,
                transcript_text,
                translation_text,
                settings
            )
            return saved_files

        saved_files = safe_execute(_save_all_files, error_context="saving transcription files")

        # Save job metadata
        def _save_metadata():
            transcript_stats = {
                "word_count": transcript_result.word_count,
                "duration": transcript_result.total_duration,
                "processing_time": transcript_result.processing_time
            }

            return save_job_metadata(
                job_dir,
                job_id,
                os.path.basename(audio_file),
                file_info,
                settings,
                transcript_stats,
                saved_files
            )

        safe_execute(_save_metadata, error_context="saving job metadata")

        # Generate display text for UI with error handling
        display_text = transcript_text  # Default fallback
        try:
            import logging
            logging.info(f"Generating UI display text - transcript length: {len(transcript_text)}, translation length: {len(translation_text)}")

            display_text = get_display_content_for_ui(transcript_text, translation_text)

            logging.info(f"Generated UI display text - length: {len(display_text)}")
            if len(display_text) < 200:  # Log short content for debugging
                logging.info(f"UI display text content: {repr(display_text[:200])}")

        except Exception as e:
            # Handle integrated display generation failure
            from errors import handle_integrated_display_failure
            display_text, display_error = handle_integrated_display_failure(
                transcript_text, translation_text, e
            )
            # Log the error but don't fail the entire process
            import logging
            logging.error(f"Integrated display generation failed: {str(e)}", exc_info=True)

        if progress_callback:
            if translation_error:
                progress_callback(1.0, "Transcription completed! Translation failed - see results for details.")
            else:
                progress_callback(1.0, "Processing completed!")

        return ProcessingResult(
            transcript=transcript_text,
            translation=translation_text,
            job_id=job_id,
            settings_used=settings,
            display_text=display_text
        )


class MockAudioHandler:
    """Mock audio processing handler for UI testing."""

    def __init__(self):
        self.current_job_id: str | None = "mock-job-123"
        self.current_transcript: str | None = "# 00:00:00 --> 00:02:30\nMock transcript content"
        self.current_translation: str | None = "# 00:00:00 --> 00:02:30\nMock translation content"

    def get_display_content(self) -> tuple[str, str, bool]:
        """
        表示用コンテンツの取得（モック）
        
        Returns:
            Tuple of (transcript, translation, translation_available)
        """
        return (
            self.current_transcript or "",
            self.current_translation or "",
            bool(self.current_translation)
        )

    def get_ui_display_text(self) -> str:
        """
        UI表示用テキストの取得（モック）
        
        Returns:
            統合表示形式のテキスト
        """
        return get_display_content_for_ui(
            self.current_transcript or "",
            self.current_translation or ""
        )

    def validate_audio(self, file_path: str) -> tuple[bool, str | None, dict[str, Any]]:
        """Mock audio validation - always returns valid."""
        return True, None, {
            'size_mb': 2.5,
            'duration_seconds': 150.0,
            'format': 'mp3',
            'sample_rate': 44100,
            'needs_warning': False
        }

    def validate_settings(self, settings: dict[str, Any]) -> tuple[bool, str]:
        """Mock settings validation - always returns valid."""
        return True, ""

    def estimate_processing_time(self, file_size_mb: float, chunk_minutes: int) -> dict[str, Any]:
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
        settings: dict[str, Any],
        progress_callback=None
    ) -> ProcessingResult:
        """Mock audio processing - returns instant results and creates mock files."""
        import asyncio
        import uuid

        from file_manager import save_job_metadata, save_transcription_files
        from util import create_job_directory

        # Generate unique job ID
        job_id = f"mock-{uuid.uuid4().hex[:8]}"

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

        # Create job directory and save files
        job_dir = create_job_directory(job_id)

        # Save transcription files (3 formats)
        saved_files = save_transcription_files(
            job_dir,
            mock_transcript,
            mock_translation,
            settings
        )

        # Save metadata
        file_info = {
            'size_mb': 2.5,
            'duration_seconds': 150.0,
            'format': 'mp3',
            'sample_rate': 44100
        }

        transcript_stats = {
            'total_chunks': 2,
            'total_duration': 150.0,
            'processing_time': 1.0
        }

        save_job_metadata(
            job_dir,
            job_id,
            os.path.basename(audio_file),
            file_info,
            settings,
            transcript_stats,
            saved_files
        )

        # Generate display text for UI
        display_text = get_display_content_for_ui(mock_transcript, mock_translation)

        # Update current job ID for download functionality
        self.current_job_id = job_id

        return ProcessingResult(
            transcript=mock_transcript,
            translation=mock_translation,
            job_id=job_id,
            settings_used=settings,
            display_text=display_text
        )
