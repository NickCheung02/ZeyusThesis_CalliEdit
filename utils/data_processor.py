"""Prepare inscription conditions for full-canvas landscape generation."""
import math
from models.text_embedder import TextEmbedder


def validate_texts(texts, width, height):
    if not isinstance(texts, list):
        raise ValueError('题字必须是 JSON 数组')
    if len(texts) > 7:
        raise ValueError('最多支持 7 个题字框')
    result = []
    for text in texts:
        if not isinstance(text, dict):
            raise ValueError('每个题字框必须包含 content 和 pos')
        content, pos = text.get('content'), text.get('pos')
        if not isinstance(content, str) or not content.strip() or len(content) > 16:
            raise ValueError('每框必须包含 1–16 个字符')
        if not isinstance(pos, (list, tuple)) or len(pos) != 4:
            raise ValueError('pos 必须包含四个坐标 [x1, y1, x2, y2]')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in pos):
            raise ValueError('坐标必须是有限数值')
        x1, y1, x2, y2 = pos
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('题字框必须位于画布内且具有正面积')
        result.append({'content': content, 'pos': list(pos)})
    return result


class UserInputProcessor:
    def __init__(self):
        self.text_embedder = TextEmbedder()

    def __call__(self, texts, prompt, width=1024, height=1024):
        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError('画布宽高必须为正数且是 16 的倍数')
        texts = validate_texts(texts, width, height)
        # The pretrained text coordinate encoder uses a 1024 × 1024 coordinate system.
        normalized = [{'content': t['content'], 'pos': [
            t['pos'][0] * 1024 / width, t['pos'][1] * 1024 / height,
            t['pos'][2] * 1024 / width, t['pos'][3] * 1024 / height,
        ]} for t in texts]
        return {'prompt': prompt, 'text_embeds': self.text_embedder(normalized).unsqueeze(0)}
