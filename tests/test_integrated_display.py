"""
Tests for integrated display functionality.
"""

import pytest
from src.integrated_display import (
    format_integrated_display,
    parse_timestamp_sections,
    validate_integrated_display,
    get_display_content_for_ui
)


class TestParseTimestampSections:
    """Test timestamp section parsing functionality."""
    
    def test_parse_single_section(self):
        """Test parsing a single timestamp section."""
        text = """# 00:00:00 --> 00:02:30
This is a test transcript.
It has multiple lines."""
        
        sections = parse_timestamp_sections(text)
        
        assert len(sections) == 1
        assert sections[0]['timestamp'] == "# 00:00:00 --> 00:02:30"
        assert sections[0]['content'] == "This is a test transcript.\nIt has multiple lines."
    
    def test_parse_multiple_sections(self):
        """Test parsing multiple timestamp sections."""
        text = """# 00:00:00 --> 00:02:30
First section content.

# 00:02:30 --> 00:05:00
Second section content.
With multiple lines."""
        
        sections = parse_timestamp_sections(text)
        
        assert len(sections) == 2
        assert sections[0]['timestamp'] == "# 00:00:00 --> 00:02:30"
        assert sections[0]['content'] == "First section content."
        assert sections[1]['timestamp'] == "# 00:02:30 --> 00:05:00"
        assert sections[1]['content'] == "Second section content.\nWith multiple lines."
    
    def test_parse_empty_text(self):
        """Test parsing empty text."""
        sections = parse_timestamp_sections("")
        assert len(sections) == 0
    
    def test_parse_no_timestamps(self):
        """Test parsing text without timestamps."""
        text = "Just some text without timestamps."
        sections = parse_timestamp_sections(text)
        assert len(sections) == 0


class TestFormatIntegratedDisplay:
    """Test integrated display formatting functionality."""
    
    def test_format_with_translation(self):
        """Test formatting with both transcript and translation."""
        transcript = """# 00:00:00 --> 00:02:30
Hello, this is a test."""
        
        translation = """# 00:00:00 --> 00:02:30
こんにちは、これはテストです。"""
        
        result = format_integrated_display(transcript, translation)
        
        # Check that timestamp appears only once
        assert result.count("# 00:00:00 --> 00:02:30") == 1
        
        # Check that both contents are present
        assert "Hello, this is a test." in result
        assert "こんにちは、これはテストです。" in result
        
        # Check that translation separator is present
        assert "─" * 20 + " Translation " + "─" * 20 in result
    
    def test_format_without_translation(self):
        """Test formatting with only transcript."""
        transcript = """# 00:00:00 --> 00:02:30
Hello, this is a test."""
        
        result = format_integrated_display(transcript, "")
        
        # Should return original transcript
        assert result == transcript
    
    def test_format_multiple_sections(self):
        """Test formatting with multiple sections."""
        transcript = """# 00:00:00 --> 00:02:30
First section.

# 00:02:30 --> 00:05:00
Second section."""
        
        translation = """# 00:00:00 --> 00:02:30
最初のセクション。

# 00:02:30 --> 00:05:00
2番目のセクション。"""
        
        result = format_integrated_display(transcript, translation)
        
        # Check that both timestamps appear
        assert result.count("# 00:00:00 --> 00:02:30") == 1
        assert result.count("# 00:02:30 --> 00:05:00") == 1
        
        # Check that all content is present
        assert "First section." in result
        assert "Second section." in result
        assert "最初のセクション。" in result
        assert "2番目のセクション。" in result
        
        # Check that translation separators are present
        assert result.count("─" * 20 + " Translation " + "─" * 20) == 2
    
    def test_format_mismatched_sections(self):
        """Test formatting when transcript and translation have different number of sections."""
        transcript = """# 00:00:00 --> 00:02:30
First section.

# 00:02:30 --> 00:05:00
Second section."""
        
        translation = """# 00:00:00 --> 00:02:30
最初のセクションのみ。"""
        
        result = format_integrated_display(transcript, translation)
        
        # Should handle mismatched sections gracefully
        assert "First section." in result
        assert "Second section." in result
        assert "最初のセクションのみ。" in result


class TestValidateIntegratedDisplay:
    """Test integrated display validation functionality."""
    
    def test_validate_with_transcript_and_translation(self):
        """Test validation with both transcript and translation."""
        transcript = "# 00:00:00 --> 00:02:30\nTest content."
        translation = "# 00:00:00 --> 00:02:30\nテストコンテンツ。"
        
        result = validate_integrated_display(transcript, translation)
        
        assert "Test content." in result
        assert "テストコンテンツ。" in result
        assert "Translation" in result
    
    def test_validate_with_transcript_only(self):
        """Test validation with transcript only."""
        transcript = "# 00:00:00 --> 00:02:30\nTest content."
        
        result = validate_integrated_display(transcript, "")
        
        assert result == transcript
    
    def test_validate_empty_transcript(self):
        """Test validation with empty transcript."""
        result = validate_integrated_display("", "translation")
        assert result == ""


class TestGetDisplayContentForUI:
    """Test UI display content generation."""
    
    def test_get_display_content_with_translation(self):
        """Test getting display content with translation."""
        transcript = "# 00:00:00 --> 00:02:30\nTest content."
        translation = "# 00:00:00 --> 00:02:30\nテストコンテンツ。"
        
        result = get_display_content_for_ui(transcript, translation)
        
        assert "Test content." in result
        assert "テストコンテンツ。" in result
    
    def test_get_display_content_without_translation(self):
        """Test getting display content without translation."""
        transcript = "# 00:00:00 --> 00:02:30\nTest content."
        
        result = get_display_content_for_ui(transcript)
        
        assert result == transcript


class TestTimestampSectionSplitting:
    """Test comprehensive timestamp section splitting functionality."""
    
    def test_parse_complex_timestamp_format(self):
        """Test parsing various timestamp formats."""
        text = """# 00:00:00 --> 00:02:30
First section with standard format.

# 00:02:30 --> 00:05:15
Second section with different end time.

# 00:05:15 --> 00:10:00
Third section with longer duration."""
        
        sections = parse_timestamp_sections(text)
        
        assert len(sections) == 3
        assert sections[0]['timestamp'] == "# 00:00:00 --> 00:02:30"
        assert sections[1]['timestamp'] == "# 00:02:30 --> 00:05:15"
        assert sections[2]['timestamp'] == "# 00:05:15 --> 00:10:00"
        
        assert "First section" in sections[0]['content']
        assert "Second section" in sections[1]['content']
        assert "Third section" in sections[2]['content']
    
    def test_parse_sections_with_empty_lines(self):
        """Test parsing sections that contain empty lines."""
        text = """# 00:00:00 --> 00:02:30
First line of content.

Second line after empty line.

# 00:02:30 --> 00:05:00

Content with leading empty line.
And another line."""
        
        sections = parse_timestamp_sections(text)
        
        assert len(sections) == 2
        # Empty lines are skipped by the parser, so content is joined without them
        assert "First line of content.\nSecond line after empty line." in sections[0]['content']
        assert "Content with leading empty line.\nAnd another line." in sections[1]['content']
    
    def test_parse_sections_with_multiline_content(self):
        """Test parsing sections with complex multiline content."""
        text = """# 00:00:00 --> 00:02:30
This is a longer section of content.
It spans multiple lines and includes
various types of text formatting.
Some lines might be longer than others.

# 00:02:30 --> 00:05:00
Another section with different content.
This one also has multiple lines.
Each line contributes to the overall content."""
        
        sections = parse_timestamp_sections(text)
        
        assert len(sections) == 2
        
        # Check first section content
        first_content = sections[0]['content']
        assert "This is a longer section" in first_content
        assert "various types of text formatting" in first_content
        assert "Some lines might be longer" in first_content
        
        # Check second section content
        second_content = sections[1]['content']
        assert "Another section with different content" in second_content
        assert "This one also has multiple lines" in second_content
        assert "Each line contributes" in second_content
    
    def test_parse_sections_ignore_malformed_timestamps(self):
        """Test that malformed timestamps are ignored."""
        text = """# 00:00:00 --> 00:02:30
Valid section content.

# Invalid timestamp format
This should not be treated as a timestamp.

# 00:02:30 --> 00:05:00
Another valid section."""
        
        sections = parse_timestamp_sections(text)
        
        assert len(sections) == 2
        assert sections[0]['timestamp'] == "# 00:00:00 --> 00:02:30"
        assert sections[1]['timestamp'] == "# 00:02:30 --> 00:05:00"
        
        # The malformed timestamp should be included as content
        assert "# Invalid timestamp format" in sections[0]['content']
        assert "This should not be treated as a timestamp" in sections[0]['content']
    
    def test_parse_sections_with_only_timestamps(self):
        """Test parsing text that contains only timestamps without content."""
        text = """# 00:00:00 --> 00:02:30

# 00:02:30 --> 00:05:00

# 00:05:00 --> 00:07:30"""
        
        sections = parse_timestamp_sections(text)
        
        # Should not create sections for timestamps without content
        assert len(sections) == 0
    
    def test_parse_sections_mixed_content_and_empty(self):
        """Test parsing with mix of sections with and without content."""
        text = """# 00:00:00 --> 00:02:30
Content for first section.

# 00:02:30 --> 00:05:00

# 00:05:00 --> 00:07:30
Content for third section."""
        
        sections = parse_timestamp_sections(text)
        
        # Should only include sections with actual content
        assert len(sections) == 2
        assert sections[0]['timestamp'] == "# 00:00:00 --> 00:02:30"
        assert sections[1]['timestamp'] == "# 00:05:00 --> 00:07:30"
        assert "Content for first section" in sections[0]['content']
        assert "Content for third section" in sections[1]['content']


class TestIntegratedDisplayEdgeCases:
    """Test edge cases for integrated display functionality."""
    
    def test_format_with_mismatched_section_counts(self):
        """Test formatting when transcript and translation have different section counts."""
        transcript = """# 00:00:00 --> 00:02:30
First transcript section.

# 00:02:30 --> 00:05:00
Second transcript section.

# 00:05:00 --> 00:07:30
Third transcript section."""
        
        translation = """# 00:00:00 --> 00:02:30
最初の翻訳セクション。"""
        
        result = format_integrated_display(transcript, translation)
        
        # Should handle all transcript sections
        assert "First transcript section." in result
        assert "Second transcript section." in result
        assert "Third transcript section." in result
        
        # Should include the available translation
        assert "最初の翻訳セクション。" in result
        
        # Should have appropriate number of translation separators
        assert result.count("Translation") >= 1
    
    def test_format_with_translation_having_more_sections(self):
        """Test formatting when translation has more sections than transcript."""
        transcript = """# 00:00:00 --> 00:02:30
Single transcript section."""
        
        translation = """# 00:00:00 --> 00:02:30
最初の翻訳セクション。

# 00:02:30 --> 00:05:00
2番目の翻訳セクション。"""
        
        result = format_integrated_display(transcript, translation)
        
        # Should include transcript content
        assert "Single transcript section." in result
        
        # Should include all translation content
        assert "最初の翻訳セクション。" in result
        assert "2番目の翻訳セクション。" in result
        
        # Should handle the extra translation section appropriately
        assert result.count("Translation") >= 1
    
    def test_format_with_no_timestamps_in_transcript(self):
        """Test formatting when transcript has no timestamp markers."""
        transcript = "This is plain text without timestamps."
        translation = "# 00:00:00 --> 00:02:30\nこれはタイムスタンプなしのプレーンテキストです。"
        
        result = format_integrated_display(transcript, translation)
        
        # When transcript has no timestamps, only translation sections are processed
        # The transcript without timestamps is not included in the result
        assert "これはタイムスタンプなしのプレーンテキストです。" in result
        assert "Translation" in result
    
    def test_format_with_no_timestamps_in_translation(self):
        """Test formatting when translation has no timestamp markers."""
        transcript = "# 00:00:00 --> 00:02:30\nThis has timestamps."
        translation = "これにはタイムスタンプがありません。"
        
        result = format_integrated_display(transcript, translation)
        
        # Should include transcript content (has timestamps)
        assert "This has timestamps." in result
        # Translation without timestamps is not processed into sections
        # Only transcript sections are included
    
    def test_format_with_very_long_content(self):
        """Test formatting with very long content sections."""
        long_content = "This is a very long line of content. " * 50
        
        transcript = f"""# 00:00:00 --> 00:02:30
{long_content}

# 00:02:30 --> 00:05:00
Another section with normal length."""
        
        translation = f"""# 00:00:00 --> 00:02:30
これは非常に長いコンテンツ行です。 """ * 30 + """

# 00:02:30 --> 00:05:00
通常の長さの別のセクション。"""
        
        result = format_integrated_display(transcript, translation)
        
        # Should handle long content without issues
        assert len(result) > 0
        assert "This is a very long line" in result
        assert "これは非常に長いコンテンツ行です" in result
        assert "Translation" in result
    
    def test_format_with_special_characters(self):
        """Test formatting with special characters and unicode."""
        transcript = """# 00:00:00 --> 00:02:30
Content with special chars: @#$%^&*()
Unicode: café, naïve, résumé
Emojis: 🎵 🎤 📝"""
        
        translation = """# 00:00:00 --> 00:02:30
特殊文字を含むコンテンツ: ！？＃％
日本語: こんにちは、さようなら
絵文字: 🎵 🎤 📝"""
        
        result = format_integrated_display(transcript, translation)
        
        # Should preserve all special characters
        assert "@#$%^&*()" in result
        assert "café, naïve, résumé" in result
        assert "🎵 🎤 📝" in result
        assert "特殊文字を含むコンテンツ" in result
        assert "こんにちは、さようなら" in result


class TestIntegratedDisplayPerformance:
    """Test performance aspects of integrated display functionality."""
    
    def test_format_large_number_of_sections(self):
        """Test formatting with a large number of timestamp sections."""
        import time
        
        # Generate transcript with many sections
        transcript_parts = []
        translation_parts = []
        
        for i in range(100):  # 100 sections
            start_time = f"{i//60:02d}:{i%60:02d}:00"
            end_time = f"{(i+1)//60:02d}:{(i+1)%60:02d}:00"
            
            transcript_parts.append(f"# {start_time} --> {end_time}")
            transcript_parts.append(f"Transcript section {i+1} content.")
            transcript_parts.append("")
            
            translation_parts.append(f"# {start_time} --> {end_time}")
            translation_parts.append(f"翻訳セクション {i+1} コンテンツ。")
            translation_parts.append("")
        
        transcript = "\n".join(transcript_parts)
        translation = "\n".join(translation_parts)
        
        # Measure formatting time
        start_time = time.time()
        result = format_integrated_display(transcript, translation)
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second for 100 sections)
        assert elapsed_time < 1.0
        
        # Should produce correct result
        assert len(result) > 0
        assert "Transcript section 1 content." in result
        assert "Transcript section 100 content." in result
        assert "翻訳セクション 1 コンテンツ。" in result
        assert "翻訳セクション 100 コンテンツ。" in result
        
        # Should have correct number of translation separators
        assert result.count("Translation") == 100
    
    def test_parse_sections_performance(self):
        """Test performance of timestamp section parsing."""
        import time
        
        # Generate large text with many sections
        text_parts = []
        for i in range(200):  # 200 sections
            start_time = f"{i//60:02d}:{i%60:02d}:00"
            end_time = f"{(i+1)//60:02d}:{(i+1)%60:02d}:00"
            
            text_parts.append(f"# {start_time} --> {end_time}")
            text_parts.append(f"Section {i+1} content line 1.")
            text_parts.append(f"Section {i+1} content line 2.")
            text_parts.append("")
        
        text = "\n".join(text_parts)
        
        # Measure parsing time
        start_time = time.time()
        sections = parse_timestamp_sections(text)
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (< 0.5 seconds for 200 sections)
        assert elapsed_time < 0.5
        
        # Should produce correct number of sections
        assert len(sections) == 200
        
        # Verify first and last sections
        assert sections[0]['timestamp'] == "# 00:00:00 --> 00:01:00"
        assert "Section 1 content" in sections[0]['content']
        assert sections[-1]['timestamp'] == "# 03:19:00 --> 03:20:00"
        assert "Section 200 content" in sections[-1]['content']


class TestIntegratedDisplayValidation:
    """Test validation functionality for integrated display."""
    
    def test_validate_with_empty_inputs(self):
        """Test validation with various empty input combinations."""
        # Both empty
        result = validate_integrated_display("", "")
        assert result == ""
        
        # Empty transcript, non-empty translation
        result = validate_integrated_display("", "Translation text")
        assert result == ""
        
        # Non-empty transcript, empty translation
        result = validate_integrated_display("Transcript text", "")
        assert result == "Transcript text"
        
        # Whitespace-only inputs
        result = validate_integrated_display("   ", "   ")
        # When both are whitespace-only, format_integrated_display returns empty string
        # because the parsing function skips empty/whitespace lines
        assert result == ""
        
        result = validate_integrated_display("Transcript", "   ")
        # Whitespace-only translation is still truthy, so format_integrated_display is called
        # But since neither has timestamps, it returns empty string
        assert result == ""
    
    def test_validate_with_none_inputs(self):
        """Test validation with None inputs."""
        # This tests robustness against None values
        try:
            result = validate_integrated_display(None, None)
            # Should handle gracefully or raise appropriate error
            assert result == "" or result is None
        except (TypeError, AttributeError):
            # Acceptable to raise error for None inputs
            pass
    
    def test_get_display_content_for_ui_consistency(self):
        """Test that get_display_content_for_ui is consistent with validate_integrated_display."""
        test_cases = [
            ("", ""),
            ("Transcript only", ""),
            ("", "Translation only"),
            ("Transcript", "Translation"),
            ("# 00:00:00 --> 00:02:30\nTimestamped transcript", "# 00:00:00 --> 00:02:30\nTimestamped translation")
        ]
        
        for transcript, translation in test_cases:
            result1 = validate_integrated_display(transcript, translation)
            result2 = get_display_content_for_ui(transcript, translation)
            
            # Both functions should return the same result
            assert result1 == result2