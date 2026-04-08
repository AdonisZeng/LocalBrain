"""
Shared utilities for document processing.
"""

from typing import List


def is_valid_text(text: str) -> bool:
    """Check if text contains valid content (printable ratio > 80%)."""
    if not text or not text.strip():
        return False

    printable_chars = sum(1 for c in text if c.isprintable() or c in '\n\r\t')
    total_chars = len(text)

    if total_chars == 0:
        return False

    return (printable_chars / total_chars) > 0.8


def detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet."""
    try:
        import chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read(2048)  # 2KB sample sufficient for detection
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            return encoding if encoding else 'utf-8'
    except ImportError:
        return 'utf-8'
    except Exception:
        return 'utf-8'
