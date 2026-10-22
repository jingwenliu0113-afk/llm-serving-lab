"""A stand-in for the transformers tokenizer behind capserve.tokens.LocalTokenizer (decision 038 §4) that counts like
the fake vLLM's /tokenize: whitespace-separated words of the message contents (no template tokens) or of the prompt.
extra adds that many tokens to every chat count (a local tokenizer that disagrees with vLLM); fail makes the template
raise, as Qwen3.5's raise_exception does."""

from __future__ import annotations


class WordsHF:
    def __init__(self, *, extra: int = 0, fail: bool = False):
        self.extra = extra
        self.fail = fail
        self.calls = 0

    def apply_chat_template(self, messages, **kw):
        self.calls += 1
        if self.fail:
            raise ValueError("System message must be at the beginning.")
        return " ".join(m["content"] for m in messages) + " x" * self.extra

    def encode(self, text, *, add_special_tokens):
        self.calls += 1
        return text.split()
