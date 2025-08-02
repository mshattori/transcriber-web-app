from typing import List, Dict, Tuple, Any
import openai


def chat_completion(
    api_key: str,
    model: str,
    message: str,
    system_message: str = None,
    history: List[Dict[str, str]] | None = None,
    temperature: float = 0.7,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    - 戻り値: (アシスタント応答, 更新後の history)
    """
    openai.api_key = api_key

    if history is None or len(history) == 0:
        # 初回呼び出し: system + user
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": message})
        new_history: List[Dict[str, str]] = []
    else:
        # 2 回目以降
        messages = history + [{"role": "user", "content": message}]
        new_history = history.copy()

    resp = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    assistant_reply = resp.choices[0].message.content.strip()

    # 最新の user/assistant を履歴に追加
    new_history.append({"role": "user", "content": message})
    new_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply, new_history


def structured_completion(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: Dict[str, Any],
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """
    JSON での構造化出力を 1 ショットで取得。
      - json_schema: {"type": "object", "properties": {...}, "required": [...]}
      - 戻り値: パース済み Python dict
    """
    openai.api_key = api_key

    resp = openai.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object", "schema": json_schema},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return resp.choices[0].message.model_dump()["content"]