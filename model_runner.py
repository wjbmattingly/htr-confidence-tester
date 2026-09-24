"""Loads the HTR model once and runs repeated sampled generations (CUDA, Metal or CPU)."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from PIL import Image

# Qwen 3.5's linear-attention layers have no optimized macOS kernels; a few ops
# may be missing on MPS too, so let those fall back to CPU instead of crashing.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DEFAULT_REPO = "wjbmattingly/comma-qwen-3.5-0.8b-full-33k"
PROMPT_PATH = Path(__file__).parent / "prompt.txt"


def _default_device() -> str:
    if os.environ.get("HTR_DEVICE"):
        return os.environ["HTR_DEVICE"]
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # The hybrid gated-delta-rule layers run via a reference PyTorch
        # implementation on macOS (the fused Triton kernels are CUDA-only);
        # on CPU that is unusably slow, on Metal it is workable.
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class ModelManager:
    """Lazy singleton around the model + processor.

    The README for the model is explicit about three things this class honors:
    the shipped prompt.txt must be the text input, `enable_thinking=False` is
    mandatory, and the processor must come from the finetune repo (its config
    caps visual tokens at 2048/page to match training).
    """

    def __init__(self, repo: str | None = None):
        self.repo = repo or os.environ.get("HTR_MODEL_REPO", DEFAULT_REPO)
        self.device = _default_device()
        self._lock = threading.Lock()
        self.model = None
        self.processor = None
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8")
        self.state = "not_loaded"  # not_loaded | loading | ready | error
        self.error: str | None = None

    def ensure_loaded(self):
        # Held for the whole load: jobs submitted together must not import
        # transformers / load weights concurrently (half-initialized modules).
        with self._lock:
            if self.state == "ready":
                return
            self.state = "loading"
            try:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor

                torch.set_num_threads(max(1, (os.cpu_count() or 4)))
                processor = AutoProcessor.from_pretrained(self.repo)
                # float32 on CPU: bf16 matmuls fall back to slow paths on many Macs
                dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
                model = AutoModelForImageTextToText.from_pretrained(self.repo, dtype=dtype)
                model.to(self.device)
                model.eval()
                self.processor, self.model = processor, model
                self.state = "ready"
                self.error = None
            except Exception as e:  # surface load failures to the UI
                self.state = "error"
                self.error = f"{type(e).__name__}: {e}"
                raise

    def transcribe_once(
        self,
        image: Image.Image,
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        greedy: bool = False,
        max_new_tokens: int = 3072,
        seed: int | None = None,
        on_token=None,
        should_stop=None,
    ) -> str:
        """One generation pass. Streams tokens to on_token(text_so_far, n_tokens)."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        assert self.state == "ready"
        processor, model = self.processor, self.model

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": self.prompt}],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = processor(text=[text], images=[[image]], return_tensors="pt").to(
            model.device
        )

        if seed is not None:
            torch.manual_seed(seed)

        stopping = None
        if should_stop is not None:

            class _Cancel(StoppingCriteria):
                def __call__(self, input_ids, scores, **kw):
                    return bool(should_stop())

            stopping = StoppingCriteriaList([_Cancel()])

        streamer = TextIteratorStreamer(
            processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            **inputs,
            repetition_penalty=1.1,
            max_new_tokens=max_new_tokens,
            eos_token_id=[processor.tokenizer.eos_token_id],
            pad_token_id=processor.tokenizer.pad_token_id,
            streamer=streamer,
        )
        if greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        if stopping is not None:
            gen_kwargs["stopping_criteria"] = stopping

        errors: list[Exception] = []

        def _generate():
            try:
                with torch.inference_mode():
                    model.generate(**gen_kwargs)
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=_generate, daemon=True)
        t.start()

        pieces: list[str] = []
        n = 0
        for piece in streamer:
            pieces.append(piece)
            n += 1
            if on_token is not None and n % 5 == 0:
                on_token("".join(pieces), n)
        t.join()
        if errors:
            raise errors[0]

        out = "".join(pieces).strip()
        if on_token is not None:
            on_token(out, n)
        return out
