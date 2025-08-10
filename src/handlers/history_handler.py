"""
History handler - manages job history and transcript loading.

Separates history management business logic from UI event handlers.
"""

import os
import json
from typing import List, Tuple


class HistoryHandler:
    """Real history handler."""
    
    def __init__(self):
        # Get the absolute path of the project root directory (one level up from src)
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        self.data_dir = os.path.join(self.project_root, "data")
    
    def get_job_history(self) -> List[List[str]]:
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
                            with open(metadata_path, 'r', encoding='utf-8') as f:
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
    
    def load_job_transcript(self, job_id: str) -> Tuple[str, str]:
        """
        Load transcript and translation for a specific job.
        
        Args:
            job_id: Job ID to load
            
        Returns:
            Tuple of (transcript, translation)
        """
        if not job_id:
            return "", ""
        
        if not os.path.exists(self.data_dir):
            return "", ""
        
        # Find job by ID across all date folders
        for date_folder in os.listdir(self.data_dir):
            date_path = os.path.join(self.data_dir, date_folder)
            if not os.path.isdir(date_path):
                continue
                
            job_path = os.path.join(date_path, job_id)
            if os.path.exists(job_path):
                transcript_path = os.path.join(job_path, "transcript.txt")
                transcript = ""
                if os.path.exists(transcript_path):
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        transcript = f.read()
                
                # Look for translation file
                translation = ""
                for file in os.listdir(job_path):
                    if file.startswith("transcript.") and file.endswith(".txt") and file != "transcript.txt":
                        translation_path = os.path.join(job_path, file)
                        with open(translation_path, 'r', encoding='utf-8') as f:
                            translation = f.read()
                        break
                
                return transcript, translation
        
        return "", ""


class MockHistoryHandler:
    """Mock history handler for UI testing."""
    
    def __init__(self):
        self.mock_jobs = [
            ["mock-001", "2024-08-10T10:30:00", "sample_audio.mp3", "120.0s", "auto", "Completed"],
            ["mock-002", "2024-08-10T14:15:00", "meeting_record.wav", "180.5s", "en", "Completed"],
            ["mock-003", "2024-08-09T16:45:00", "interview.m4a", "95.2s", "ja", "Completed"],
        ]
    
    def get_job_history(self) -> List[List[str]]:
        """
        Mock job history - returns predefined job list.
        
        Returns:
            List of mock job records
        """
        return self.mock_jobs.copy()
    
    def load_job_transcript(self, job_id: str) -> Tuple[str, str]:
        """
        Mock transcript loading - returns mock transcripts.
        
        Args:
            job_id: Job ID to load
            
        Returns:
            Tuple of (mock_transcript, mock_translation)
        """
        if not job_id:
            return "", ""
        
        # Mock transcript content based on job ID
        mock_transcripts = {
            "mock-001": (
                """# 00:00:00 --> 00:01:00
This is a mock transcript for job mock-001.
It contains sample content for testing the history loading functionality.

# 00:01:00 --> 00:02:00
The transcript includes multiple segments with timestamps
to demonstrate the full transcript viewing experience.""",
                """# 00:00:00 --> 00:01:00
これはジョブ mock-001 のモック転写です。
履歴読み込み機能をテストするためのサンプルコンテンツが含まれています。

# 00:01:00 --> 00:02:00
転写には、完全な転写表示体験を実証するために、
タイムスタンプ付きの複数のセグメントが含まれています。"""
            ),
            "mock-002": (
                """# 00:00:00 --> 00:01:30
Mock transcript for meeting recording job mock-002.
This demonstrates loading different transcripts for different jobs.

# 00:01:30 --> 00:03:00
Each job has its own unique transcript content
that can be loaded and displayed independently.""",
                ""
            ),
            "mock-003": (
                """# 00:00:00 --> 00:00:45
日本語のモック転写 - ジョブ mock-003
これは日本語音声のテスト用コンテンツです。

# 00:00:45 --> 00:01:35
各ジョブには独自の転写コンテンツがあり、
独立して読み込みと表示が可能です。""",
                """# 00:00:00 --> 00:00:45
Mock Japanese transcript - Job mock-003
This is test content for Japanese audio.

# 00:00:45 --> 00:01:35
Each job has its own transcript content
that can be loaded and displayed independently."""
            )
        }
        
        if job_id in mock_transcripts:
            return mock_transcripts[job_id]
        else:
            # Default mock content for unknown job IDs
            return (
                f"Mock transcript for job {job_id}.\nThis is default mock content for testing purposes.",
                f"Mock translation for job {job_id}.\nこれはテスト用のデフォルトモック翻訳です。"
            )