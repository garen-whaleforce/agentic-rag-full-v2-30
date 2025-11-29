from typing import Dict

from storage import get_prompt, set_prompt, get_all_prompts


def get_prompt_override(key: str, default: str) -> str:
    """
    從 DB 讀取 override（prompt_configs），如果沒有值就回傳 default。
    """
    content = get_prompt(key)
    return content if content not in (None, "") else default


def get_all_prompt_overrides() -> Dict[str, str]:
    """
    回傳 DB 中所有 key -> content 的 mapping。
    """
    return get_all_prompts()


def save_prompt_override(key: str, content: str) -> None:
    """
    更新（或新增）某個 prompt 的 override。
    """
    set_prompt(key, content)
