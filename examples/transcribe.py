# minimal_transcribe.py
import openai
from pathlib import Path

def transcribe(
    audio_path: str,
    api_key: str,
    model: str = "whisper-1",
    language: str = "auto"
) -> str:
    """
    単一の音声ファイルを OpenAI Speech-to-Text API で文字起こししてプレーンテキストを返す。
    - language="auto" なら自動判定。
    """
    openai.api_key = api_key

    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(p)

    with p.open("rb") as f:
        resp = openai.audio.transcriptions.create(
            model=model,
            file=f,
            language=None if language == "auto" else language,
            response_format="text"   # 文字列で受け取る
        )
    return resp  # str
