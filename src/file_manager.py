"""
File management utilities for transcription and translation files.

Handles saving transcription, translation, and integrated display files
in the proper format and structure.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from integrated_display import format_integrated_display


def save_transcription_files(
    job_dir: str, 
    transcript: str, 
    translation: str = "", 
    settings: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    転写、翻訳、統合表示の3形式で保存
    
    Args:
        job_dir: ジョブディレクトリのパス
        transcript: 転写テキスト
        translation: 翻訳テキスト（オプション）
        settings: 設定辞書（オプション）
        
    Returns:
        保存されたファイルのパス辞書
        
    Raises:
        FileError: ファイル保存に失敗した場合
    """
    from errors import FileError, safe_execute
    
    if settings is None:
        settings = {}
    
    saved_files = {}
    
    # 1. 元の転写ファイル（常に保存）
    def _save_transcript():
        transcript_path = os.path.join(job_dir, "transcript.txt")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        return transcript_path
    
    try:
        transcript_path = safe_execute(_save_transcript, error_context="saving transcript file")
        saved_files['transcript'] = transcript_path
    except Exception as e:
        raise FileError(f"Failed to save transcript file: {str(e)}", 
                       file_path="transcript.txt", operation="save")
    
    # 2. 翻訳ファイル（翻訳が有効で成功した場合のみ）
    if translation and settings.get("translation_enabled", False):
        lang_code = settings.get("default_translation_language", "").lower()[:2]
        if not lang_code:
            lang_code = "ja"  # デフォルトは日本語
        
        def _save_translation():
            translation_path = os.path.join(job_dir, f"transcript.{lang_code}.txt")
            with open(translation_path, 'w', encoding='utf-8') as f:
                f.write(translation)
            return translation_path
        
        try:
            translation_path = safe_execute(_save_translation, error_context="saving translation file")
            saved_files['translation'] = translation_path
        except Exception as e:
            # Log warning but don't fail the entire process
            import logging
            logging.warning(f"Failed to save translation file: {str(e)}")
    
    # 3. 統合表示ファイル（翻訳がある場合のみ、エラー翻訳でも保存）
    if translation:
        def _save_integrated():
            import logging
            logging.info(f"Generating integrated display - transcript length: {len(transcript)}, translation length: {len(translation)}")
            
            integrated_display = format_integrated_display(transcript, translation)
            
            logging.info(f"Generated integrated display - length: {len(integrated_display)}")
            if len(integrated_display) < 100:  # Log short content for debugging
                logging.info(f"Integrated display content: {repr(integrated_display)}")
            
            integrated_path = os.path.join(job_dir, "transcript_integrated.txt")
            with open(integrated_path, 'w', encoding='utf-8') as f:
                f.write(integrated_display)
            
            logging.info(f"Saved integrated display to: {integrated_path}")
            return integrated_path
        
        try:
            integrated_path = safe_execute(_save_integrated, error_context="saving integrated display file")
            saved_files['integrated'] = integrated_path
        except Exception as e:
            # Log warning but don't fail the entire process
            import logging
            logging.error(f"Failed to save integrated display file: {str(e)}", exc_info=True)
    
    return saved_files


def save_job_metadata(
    job_dir: str,
    job_id: str,
    original_filename: str,
    file_info: Dict[str, Any],
    settings: Dict[str, Any],
    transcript_stats: Optional[Dict[str, Any]] = None,
    saved_files: Optional[Dict[str, str]] = None
) -> str:
    """
    ジョブメタデータを保存
    
    Args:
        job_dir: ジョブディレクトリのパス
        job_id: ジョブID
        original_filename: 元のファイル名
        file_info: ファイル情報
        settings: 設定辞書
        transcript_stats: 転写統計（オプション）
        saved_files: 保存されたファイルのパス辞書（オプション）
        
    Returns:
        メタデータファイルのパス
    """
    if transcript_stats is None:
        transcript_stats = {}
    
    if saved_files is None:
        saved_files = {}
    
    # ファイル名から相対パスを生成
    files_info = {}
    for file_type, file_path in saved_files.items():
        if file_path:
            files_info[file_type] = os.path.basename(file_path)
    
    metadata = {
        "job_id": job_id,
        "timestamp": datetime.now().isoformat(),
        "original_filename": original_filename,
        "file_info": file_info,
        "settings": settings,
        "transcript_stats": transcript_stats,
        "translation_enabled": settings.get("translation_enabled", False),
        "translation_available": bool(saved_files.get('translation')),
        "files": files_info
    }
    
    metadata_path = os.path.join(job_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return metadata_path


def load_job_files(job_dir: str) -> Tuple[str, str, str]:
    """
    ジョブディレクトリから各形式のファイルを読み込み
    
    Args:
        job_dir: ジョブディレクトリのパス
        
    Returns:
        (transcript, translation, integrated_display) のタプル
    """
    from errors import handle_file_read_failure
    
    transcript = ""
    translation = ""
    integrated_display = ""
    
    # 転写ファイルの読み込み（必須）
    transcript_path = os.path.join(job_dir, "transcript.txt")
    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
        except Exception as e:
            transcript, file_error = handle_file_read_failure(
                transcript_path, e, "Failed to load transcript file"
            )
            import logging
            logging.error(f"Failed to read transcript file: {str(e)}")
    
    # 翻訳ファイルの読み込み（言語コードを自動検出）
    try:
        for file in os.listdir(job_dir):
            if (file.startswith("transcript.") and 
                file.endswith(".txt") and 
                file != "transcript.txt" and 
                file != "transcript_integrated.txt"):
                translation_path = os.path.join(job_dir, file)
                try:
                    with open(translation_path, 'r', encoding='utf-8') as f:
                        translation = f.read()
                    break
                except Exception as e:
                    translation, file_error = handle_file_read_failure(
                        translation_path, e, ""
                    )
                    import logging
                    logging.warning(f"Failed to read translation file {file}: {str(e)}")
    except OSError as e:
        import logging
        logging.warning(f"Failed to list job directory files: {str(e)}")
    
    # 統合表示ファイルの読み込み
    integrated_path = os.path.join(job_dir, "transcript_integrated.txt")
    if os.path.exists(integrated_path):
        try:
            with open(integrated_path, 'r', encoding='utf-8') as f:
                integrated_display = f.read()
        except Exception as e:
            # Fallback to generating integrated display from transcript and translation
            try:
                if transcript and translation:
                    integrated_display = format_integrated_display(transcript, translation)
                else:
                    integrated_display = transcript
            except Exception as display_error:
                integrated_display = transcript
                import logging
                logging.warning(f"Failed to read and regenerate integrated display: {str(e)}, {str(display_error)}")
    
    return transcript, translation, integrated_display


def load_job_metadata(job_dir: str) -> Dict[str, Any]:
    """
    ジョブメタデータを読み込み
    
    Args:
        job_dir: ジョブディレクトリのパス
        
    Returns:
        メタデータ辞書
    """
    metadata_path = os.path.join(job_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def get_display_content_from_job(job_dir: str) -> str:
    """
    ジョブディレクトリから表示用コンテンツを取得
    
    Args:
        job_dir: ジョブディレクトリのパス
        
    Returns:
        表示用テキスト
    """
    from errors import handle_file_read_failure
    
    # 統合表示ファイルがあれば優先
    integrated_path = os.path.join(job_dir, "transcript_integrated.txt")
    if os.path.exists(integrated_path):
        try:
            with open(integrated_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            import logging
            logging.warning(f"Failed to read integrated display file: {str(e)}")
            # Fall through to transcript file
    
    # なければ原文のみ
    transcript_path = os.path.join(job_dir, "transcript.txt")
    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            content, file_error = handle_file_read_failure(
                transcript_path, e, "No content available - file read failed"
            )
            import logging
            logging.error(f"Failed to read transcript file: {str(e)}")
            return content
    
    return "No content available - files not found"


def create_download_package(job_dir: str, job_id: str) -> str:
    """
    ダウンロード用ファイルパッケージの作成（3形式対応）
    
    Args:
        job_dir: ジョブディレクトリのパス
        job_id: ジョブID
        
    Returns:
        ダウンロード用ファイルのパス
    """
    import tempfile
    import zipfile
    
    # 利用可能なファイルを確認
    available_files = []
    
    # 原文（常にある）
    transcript_path = os.path.join(job_dir, "transcript.txt")
    if os.path.exists(transcript_path):
        available_files.append(("transcript.txt", transcript_path))
    
    # 翻訳（ある場合）
    for file in os.listdir(job_dir):
        if (file.startswith("transcript.") and 
            file.endswith(".txt") and 
            file != "transcript.txt" and 
            file != "transcript_integrated.txt"):
            available_files.append((file, os.path.join(job_dir, file)))
    
    # 統合表示（ある場合）
    integrated_path = os.path.join(job_dir, "transcript_integrated.txt")
    if os.path.exists(integrated_path):
        available_files.append(("transcript_integrated.txt", integrated_path))
    
    if len(available_files) == 1:
        # 単一ファイルの場合は直接提供
        return available_files[0][1]
    else:
        # 複数ファイルの場合はZIP
        return _create_zip_from_files(available_files, job_id)


def _create_zip_from_files(files: List[Tuple[str, str]], job_id: str) -> str:
    """
    複数ファイルからZIPを作成
    
    Args:
        files: (filename, filepath) のタプルのリスト
        job_id: ジョブID
        
    Returns:
        ZIPファイルのパス
    """
    import tempfile
    import zipfile
    
    temp_zip_path = tempfile.mktemp(suffix=f"_{job_id}.zip")
    
    with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename, filepath in files:
            zipf.write(filepath, filename)
    
    return temp_zip_path