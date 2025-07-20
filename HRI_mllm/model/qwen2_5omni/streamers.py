from __future__ import annotations

import asyncio
from queue import Queue
from typing import TYPE_CHECKING, Optional
from transformers import TextStreamer
from jiwer import wer

if TYPE_CHECKING:
    from transformers.models.auto import AutoTokenizer


class QwenTextStreamer(TextStreamer):
    def __init__(self, tokenizer: AutoTokenizer, skip_prompt: bool = False, **decode_kwargs):
        super(QwenTextStreamer, self).__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_list = []
        self.tg = None

    def check_tts_quality(self):
        tg_text = list(filter(lambda x: x != "", [ti.mark for ti in self.tg[0]]))
        tg_text = [text.strip().lower() for text in tg_text]
        pred_text = list(filter(lambda x: x != "", self.text_list))
        pred_text = [text.strip().lower() for text in pred_text]

        wer_result = wer(" ".join(tg_text), " ".join(pred_text[-213:]))
        return wer_result < 0.1

    def clear_text_list(self):
        self.text_list = []

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.text_list.append(printable_text)

        self.on_finalized_text(printable_text)


class QwenMotionAdaptorStreamer(TextStreamer):
    def __init__(self, tokenizer: AutoTokenizer, skip_prompt: bool = False, **decode_kwargs):
        super(QwenMotionAdaptorStreamer, self).__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_list = []
        self.token_list = []
        self.tg = None

    def check_tts_quality(self):
        return True

    def clear_text_list(self):
        self.text_list = []
        self.token_list = []

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        self.token_list.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.text_list.append(printable_text)

        self.on_finalized_text(printable_text)        