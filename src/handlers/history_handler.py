"""
History handler - manages job history and transcript loading.

Separates history management business logic from UI event handlers.
"""

import json
import os
from typing import Any


class HistoryHandler:
    """Real history handler."""

    def __init__(self):
        # Get the absolute path of the project root directory (one level up from src)
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        self.data_dir = os.path.join(self.project_root, "data")

    def get_job_history(self) -> list[list[str]]:
        """
        Get list of previous jobs for history view.
        
        Returns:
            List of job records [job_id, timestamp, filename, duration, language, status]
        """
        jobs = []

        if not os.path.exists(self.data_dir):
            return []

        try:
            for date_folder in sorted(os.listdir(self.data_dir), reverse=True):
                date_path = os.path.join(self.data_dir, date_folder)
                if not os.path.isdir(date_path):
                    continue

                for job_folder in os.listdir(date_path):
                    job_path = os.path.join(date_path, job_folder)
                    metadata_path = os.path.join(job_path, "metadata.json")

                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, encoding='utf-8') as f:
                                metadata = json.load(f)

                            jobs.append([
                                metadata.get("job_id", job_folder),
                                metadata.get("timestamp", ""),
                                metadata.get("original_filename", ""),
                                f"{metadata.get('file_info', {}).get('duration_seconds', 0):.1f}s",
                                metadata.get("settings", {}).get("default_language", ""),
                                "Completed"
                            ])
                        except (KeyError, ValueError, json.JSONDecodeError):
                            continue
        except OSError:
            pass

        return jobs

    def load_job_transcript(self, job_id: str) -> tuple[str, str]:
        """
        Load display content and translation for a specific job using new file management.
        
        Args:
            job_id: Job ID to load
            
        Returns:
            Tuple of (display_content, translation)
        """
        from ..errors import handle_file_read_failure
        from ..file_manager import get_display_content_from_job, load_job_files

        if not job_id:
            return "", ""

        try:
            # Find job directory using our own data directory
            job_dir = self._find_job_directory(job_id)
            if not job_dir:
                return "Job not found - files may have been deleted or moved", ""

            # Get display content (integrated display if available, otherwise transcript)
            display_content = get_display_content_from_job(job_dir)

            # Get individual files for reference
            transcript, translation, _ = load_job_files(job_dir)

            return display_content, translation

        except Exception as e:
            import logging
            logging.error(f"Error loading job transcript for {job_id}: {e}")
            error_content, _ = handle_file_read_failure(
                f"job_{job_id}", e, "Failed to load job content - files may be corrupted or inaccessible"
            )
            return error_content, ""

    def load_job_content(self, job_id: str) -> tuple[str, str, dict[str, Any]]:
        """
        Load job content and metadata for display purposes.
        
        Args:
            job_id: Job ID to load
            
        Returns:
            Tuple of (display_content, translation, metadata)
        """
        from ..errors import handle_file_read_failure
        from ..file_manager import (
            get_display_content_from_job,
            load_job_files,
            load_job_metadata,
        )

        if not job_id:
            return "", "", {}

        try:
            # Find job directory using our own data directory
            job_dir = self._find_job_directory(job_id)
            if not job_dir:
                error_content, _ = handle_file_read_failure(
                    f"job_{job_id}", FileNotFoundError("Job directory not found"),
                    "Job not found - files may have been deleted or moved"
                )
                return error_content, "", {"error": "job_not_found"}

            # Get display content (integrated display if available, otherwise transcript)
            display_content = get_display_content_from_job(job_dir)

            # Get individual files
            transcript, translation, integrated_display = load_job_files(job_dir)

            # Get metadata with error handling
            try:
                metadata = load_job_metadata(job_dir)
            except Exception as e:
                import logging
                logging.warning(f"Failed to load metadata for job {job_id}: {e}")
                metadata = {"error": "metadata_load_failed"}

            # Ensure translation availability is correctly detected
            if not metadata.get("translation_available") and translation:
                metadata["translation_available"] = True

            return display_content, translation, metadata

        except Exception as e:
            import logging
            logging.error(f"Error loading job content for {job_id}: {e}")
            error_content, _ = handle_file_read_failure(
                f"job_{job_id}", e, "Failed to load job content - files may be corrupted or inaccessible"
            )
            return error_content, "", {"error": "content_load_failed"}

    def get_job_history_with_translation_info(self) -> list[list[str]]:
        """
        Get job history with translation information for better display.
        
        Returns:
            List of job records with translation status
        """
        jobs = []

        if not os.path.exists(self.data_dir):
            return []

        try:
            for date_folder in sorted(os.listdir(self.data_dir), reverse=True):
                date_path = os.path.join(self.data_dir, date_folder)
                if not os.path.isdir(date_path):
                    continue

                for job_folder in os.listdir(date_path):
                    job_path = os.path.join(date_path, job_folder)
                    metadata_path = os.path.join(job_path, "metadata.json")

                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, encoding='utf-8') as f:
                                metadata = json.load(f)

                            # Check for translation availability by examining files if metadata is incomplete
                            translation_available = metadata.get("translation_available", False)
                            if not translation_available:
                                translation_available = self._check_translation_files_exist(job_path)

                            # Determine language display
                            language = metadata.get("settings", {}).get("default_language", "")
                            if metadata.get("translation_enabled", False) and translation_available:
                                target_lang = metadata.get("settings", {}).get("default_translation_language", "")
                                if target_lang:
                                    target_lang = target_lang[:2].lower()
                                else:
                                    target_lang = "ja"  # Default to Japanese
                                language = f"{language}+{target_lang}"

                            jobs.append([
                                metadata.get("job_id", job_folder),
                                metadata.get("timestamp", ""),
                                metadata.get("original_filename", ""),
                                f"{metadata.get('file_info', {}).get('duration_seconds', 0):.1f}s",
                                language,
                                "Completed"
                            ])
                        except (KeyError, ValueError, json.JSONDecodeError):
                            continue
        except OSError:
            pass

        return jobs

    def _check_translation_files_exist(self, job_dir: str) -> bool:
        """
        Check if translation files exist in the job directory.
        
        Args:
            job_dir: Job directory path
            
        Returns:
            True if translation files are found
        """
        try:
            for file in os.listdir(job_dir):
                if (file.startswith("transcript.") and
                    file.endswith(".txt") and
                    file != "transcript.txt" and
                    file != "transcript_integrated.txt"):
                    return True
            return False
        except OSError:
            return False

    def has_translation_available(self, job_id: str) -> bool:
        """
        Check if a specific job has translation available.
        
        Args:
            job_id: Job ID to check
            
        Returns:
            True if translation is available for this job
        """
        if not job_id:
            return False

        try:
            job_dir = self._find_job_directory(job_id)
            if not job_dir:
                return False

            # Check metadata first
            metadata_path = os.path.join(job_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, encoding='utf-8') as f:
                    metadata = json.load(f)
                    if metadata.get("translation_available"):
                        return True

            # Fallback to file system check
            return self._check_translation_files_exist(job_dir)

        except Exception:
            return False

    def get_job_details(self, job_id: str) -> dict[str, Any]:
        """
        Get detailed information about a specific job.
        
        Args:
            job_id: Job ID to get details for
            
        Returns:
            Dictionary containing job details
        """
        if not job_id:
            return {}

        try:
            job_dir = self._find_job_directory(job_id)
            if not job_dir:
                return {}

            # Load metadata
            metadata_path = os.path.join(job_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, encoding='utf-8') as f:
                    metadata = json.load(f)
                return metadata

            return {}

        except Exception as e:
            print(f"Error loading job details: {e}")
            return {}

    def _find_job_directory(self, job_id: str) -> str | None:
        """
        Find job directory by searching through date-based folders in our data directory.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Path to job directory if found, None otherwise
        """
        if not os.path.exists(self.data_dir):
            return None

        # Search through date folders
        for date_folder in os.listdir(self.data_dir):
            potential_path = os.path.join(self.data_dir, date_folder, job_id)
            if os.path.exists(potential_path):
                return potential_path

        return None


class MockHistoryHandler:
    """Mock history handler for UI testing."""

    def __init__(self):
        self.mock_jobs = [
            ["mock-001", "2024-08-10T10:30:00", "sample_audio.mp3", "120.0s", "auto", "Completed"],
            ["mock-002", "2024-08-10T14:15:00", "meeting_record.wav", "180.5s", "en", "Completed"],
            ["mock-003", "2024-08-09T16:45:00", "interview.m4a", "95.2s", "ja", "Completed"],
        ]

    def get_job_history(self) -> list[list[str]]:
        """
        Mock job history - returns predefined job list.
        
        Returns:
            List of mock job records
        """
        return self.mock_jobs.copy()

    def load_job_transcript(self, job_id: str) -> tuple[str, str]:
        """
        Mock transcript loading - returns mock integrated display content.
        
        Args:
            job_id: Job ID to load
            
        Returns:
            Tuple of (mock_display_content, mock_translation)
        """
        from ..integrated_display import format_integrated_display

        if not job_id:
            return "", ""

        # Mock transcript content based on job ID
        mock_data = {
            "mock-001": {
                "transcript": """# 00:00:00 --> 00:01:00
This is a mock transcript for job mock-001.
It contains sample content for testing the history loading functionality.

# 00:01:00 --> 00:02:00
The transcript includes multiple segments with timestamps
to demonstrate the full transcript viewing experience.""",
                "translation": """# 00:00:00 --> 00:01:00
これはジョブ mock-001 のモック転写です。
履歴読み込み機能をテストするためのサンプルコンテンツが含まれています。

# 00:01:00 --> 00:02:00
転写には、完全な転写表示体験を実証するために、
タイムスタンプ付きの複数のセグメントが含まれています。"""
            },
            "mock-002": {
                "transcript": """# 00:00:00 --> 00:01:30
Mock transcript for meeting recording job mock-002.
This demonstrates loading different transcripts for different jobs.

# 00:01:30 --> 00:03:00
Each job has its own unique transcript content
that can be loaded and displayed independently.""",
                "translation": ""
            },
            "mock-003": {
                "transcript": """# 00:00:00 --> 00:00:45
日本語のモック転写 - ジョブ mock-003
これは日本語音声のテスト用コンテンツです。

# 00:00:45 --> 00:01:35
各ジョブには独自の転写コンテンツがあり、
独立して読み込みと表示が可能です。""",
                "translation": """# 00:00:00 --> 00:00:45
Mock Japanese transcript - Job mock-003
This is test content for Japanese audio.

# 00:00:45 --> 00:01:35
Each job has its own transcript content
that can be loaded and displayed independently."""
            }
        }

        if job_id in mock_data:
            data = mock_data[job_id]
            transcript = data["transcript"]
            translation = data["translation"]

            # Generate integrated display if translation exists
            if translation:
                display_content = format_integrated_display(transcript, translation)
            else:
                display_content = transcript

            return display_content, translation
        else:
            # Default mock content for unknown job IDs with proper timestamp format
            transcript = f"""# 00:00:00 --> 00:01:00
Mock transcript for job {job_id}.
This is default mock content for testing purposes."""
            translation = f"""# 00:00:00 --> 00:01:00
Mock translation for job {job_id}.
これはテスト用のデフォルトモック翻訳です。"""
            display_content = format_integrated_display(transcript, translation)
            return display_content, translation

    def load_job_content(self, job_id: str) -> tuple[str, str, dict[str, Any]]:
        """
        Mock job content loading with metadata.
        
        Args:
            job_id: Job ID to load
            
        Returns:
            Tuple of (display_content, translation, metadata)
        """
        display_content, translation = self.load_job_transcript(job_id)

        # Mock metadata
        metadata = {
            "translation_enabled": bool(translation),
            "translation_available": bool(translation),
            "display_preferences": {
                "default_mode": "integrated",
                "last_used_mode": "integrated"
            }
        }

        return display_content, translation, metadata

    def get_job_history_with_translation_info(self) -> list[list[str]]:
        """
        Mock job history with translation information.
        
        Returns:
            List of mock job records with translation status
        """
        return [
            ["mock-001", "2024-08-10T10:30:00", "sample_audio.mp3", "120.0s", "auto+ja", "Completed"],
            ["mock-002", "2024-08-10T14:15:00", "meeting_record.wav", "180.5s", "en", "Completed"],
            ["mock-003", "2024-08-09T16:45:00", "interview.m4a", "95.2s", "ja+en", "Completed"],
        ]

    def has_translation_available(self, job_id: str) -> bool:
        """
        Mock translation availability check.
        
        Args:
            job_id: Job ID to check
            
        Returns:
            True if mock job has translation
        """
        # Mock jobs with translation
        jobs_with_translation = {"mock-001", "mock-003"}
        return job_id in jobs_with_translation

    def get_job_details(self, job_id: str) -> dict[str, Any]:
        """
        Mock job details - returns mock metadata.
        
        Args:
            job_id: Job ID to get details for
            
        Returns:
            Mock job details dictionary
        """
        mock_details = {
            "mock-001": {
                "job_id": "mock-001",
                "timestamp": "2024-08-10T10:30:00",
                "original_filename": "sample_audio.mp3",
                "file_info": {"duration_seconds": 120.0},
                "translation_enabled": True,
                "translation_available": True,
                "settings": {
                    "default_language": "auto",
                    "default_translation_language": "Japanese"
                }
            },
            "mock-002": {
                "job_id": "mock-002",
                "timestamp": "2024-08-10T14:15:00",
                "original_filename": "meeting_record.wav",
                "file_info": {"duration_seconds": 180.5},
                "translation_enabled": False,
                "translation_available": False,
                "settings": {
                    "default_language": "en"
                }
            },
            "mock-003": {
                "job_id": "mock-003",
                "timestamp": "2024-08-09T16:45:00",
                "original_filename": "interview.m4a",
                "file_info": {"duration_seconds": 95.2},
                "translation_enabled": True,
                "translation_available": True,
                "settings": {
                    "default_language": "ja",
                    "default_translation_language": "English"
                }
            }
        }

        return mock_details.get(job_id, {
            "job_id": job_id,
            "timestamp": "2024-08-10T12:00:00",
            "original_filename": "unknown.mp3",
            "file_info": {"duration_seconds": 60.0},
            "translation_enabled": False,
            "translation_available": False,
            "settings": {"default_language": "auto"}
        })

    def _check_translation_files_exist(self, job_dir: str) -> bool:
        """
        Mock translation file check - not used in mock but needed for interface compatibility.
        
        Args:
            job_dir: Job directory path
            
        Returns:
            Always False for mock
        """
        return False
