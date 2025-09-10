"""
Integrated display functionality for transcription and translation.

Provides functions to generate integrated display format that combines
transcription and translation with proper formatting and visual separation.
"""

import re


def format_integrated_display(transcript: str, translation: str = "") -> str:
    """
    統合表示形式のテキスト生成
    
    Args:
        transcript: 元の転写テキスト
        translation: 翻訳テキスト（オプション）
        
    Returns:
        統合表示形式のテキスト
        
    Raises:
        IntegratedDisplayError: 統合表示生成に失敗した場合
    """
    from .errors import IntegratedDisplayError

    try:
        if not translation:
            # 翻訳なしの場合は転写のみ
            return transcript

        # タイムスタンプベースでセクション分割
        transcript_sections = parse_timestamp_sections(transcript)
        translation_sections = parse_timestamp_sections(translation)

        # セクション数を合わせる（翻訳の方が少ない場合に対応）
        max_sections = max(len(transcript_sections), len(translation_sections))

        result = []

        for i in range(max_sections):
            # 転写セクションの取得
            if i < len(transcript_sections):
                t_section = transcript_sections[i]
                timestamp = t_section['timestamp']
                transcript_content = t_section['content']
            else:
                timestamp = ""
                transcript_content = ""

            # 翻訳セクションの取得
            if i < len(translation_sections):
                translation_content = translation_sections[i]['content']
            else:
                translation_content = ""

            # セクションの構築
            if timestamp:
                result.append(timestamp)
                result.append("")  # 空行

            if transcript_content:
                result.append(transcript_content)
                result.append("")  # 空行

            if translation_content:
                result.append("─" * 20 + " Translation " + "─" * 20)  # ラベル付き区切り線
                result.append("")  # 空行
                result.append(translation_content)
                result.append("")  # 空行

            # セクション間の区切り
            if i < max_sections - 1:
                result.append("")  # セクション間の空行

        return '\n'.join(result)

    except Exception as e:
        raise IntegratedDisplayError(
            f"Failed to generate integrated display: {str(e)}",
            transcript=transcript,
            translation=translation
        )


def parse_timestamp_sections(text: str) -> list[dict[str, str]]:
    """
    タイムスタンプベースでテキストをセクション分割
    
    Args:
        text: 分割対象のテキスト
        
    Returns:
        セクションのリスト（各セクションはtimestampとcontentを含む辞書）
        
    Raises:
        IntegratedDisplayError: セクション分割に失敗した場合
    """
    from .errors import IntegratedDisplayError

    try:
        sections = []
        lines = text.split('\n')
        current_section = None

        # タイムスタンプパターン: # MM:SS --> MM:SS または # HH:MM:SS --> HH:MM:SS
        timestamp_pattern = r'^# (\d{2}:\d{2}(?::\d{2})? --> \d{2}:\d{2}(?::\d{2})?)'

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if re.match(timestamp_pattern, line):
                # 前のセクションを保存
                if current_section and current_section['content']:
                    sections.append(current_section)

                # 新しいセクション開始
                current_section = {
                    'timestamp': line,
                    'content': ''
                }
            elif current_section:
                # コンテンツの追加
                if current_section['content']:
                    current_section['content'] += '\n'
                current_section['content'] += line

        # 最後のセクションを保存
        if current_section and current_section['content']:
            sections.append(current_section)

        return sections

    except Exception as e:
        safe_text = ""
        if text is not None:
            safe_text = text if len(text) < 1000 else text[:1000] + "..."

        raise IntegratedDisplayError(
            f"Failed to parse timestamp sections: {str(e)}",
            transcript=safe_text,
            translation=""
        )


def validate_integrated_display(transcript: str, translation: str) -> str:
    """
    統合表示の妥当性検証
    
    Args:
        transcript: 転写テキスト
        translation: 翻訳テキスト
        
    Returns:
        検証済みの表示テキスト
    """
    import logging

    logging.debug(f"validate_integrated_display - transcript: {bool(transcript)}, translation: {bool(translation)}")

    if not transcript:
        logging.debug("validate_integrated_display - no transcript, returning empty")
        return ""

    if translation:
        logging.debug("validate_integrated_display - has translation, calling format_integrated_display")
        try:
            result = format_integrated_display(transcript, translation)
            logging.debug(f"validate_integrated_display - format result: {len(result) if result else 0} chars")
            return result
        except Exception as e:
            logging.error(f"validate_integrated_display - format_integrated_display failed: {str(e)}", exc_info=True)
            return transcript
    else:
        logging.debug("validate_integrated_display - no translation, returning transcript")
        return transcript


def get_display_content_for_ui(transcript: str, translation: str = "") -> str:
    """
    UI表示用のコンテンツを取得
    
    Args:
        transcript: 転写テキスト
        translation: 翻訳テキスト（オプション）
        
    Returns:
        UI表示用のテキスト
    """
    import logging

    try:
        logging.debug(f"get_display_content_for_ui called - transcript: {len(transcript) if transcript else 0} chars, translation: {len(translation) if translation else 0} chars")

        result = validate_integrated_display(transcript, translation)

        logging.debug(f"get_display_content_for_ui result - {len(result) if result else 0} chars")

        return result
    except Exception as e:
        # Fallback to transcript only if integrated display fails
        logging.error(f"Failed to generate UI display content: {str(e)}", exc_info=True)
        return transcript if transcript else "No content available"
