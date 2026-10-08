# ruff: noqa: E501
"""N11: the prompt enhancer against transformers' Gemma-4 on the CPU.

ref  OUT.npz "prompt"       (conda vllm-mlx, through gpu_run.py --need-gb 30: the
                            fp32 model is 20 GiB of RAM) upstream's exact recipe:
                            Gemma4ForConditionalGeneration, apply_chat_template with
                            the T2V system prompt, generate(do_sample=False,
                            no_repeat_ngram_size=5, max_new_tokens=600). Saves the
                            prompt ids, the generated ids, the text, and the first
                            step's logits.
ours OUT.npz REF.npz        (venv-slimserve, --need-gb 24) our PromptEnhancer on the
                            same prompt: compares prompt ids, first-step logits,
                            and the generated ids token for token.
"""

import sys
import time

import numpy as np

ROOT = "/Users/seangherardi/models/ltx-2.5/official/prompt_enhancer/gemma-4-E2B-it"
REPO = "/Users/seangherardi/Code/slimserve/SlimServe-ltx25"


def ref(out, prompt):
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    sys.path.append(REPO)
    from slimserve.video.ltx25.enhancer import system_prompt

    tk = AutoTokenizer.from_pretrained(ROOT)
    msgs = [
        {"role": "system", "content": system_prompt("t2v")},
        {"role": "user", "content": f"user prompt: {prompt}"},
    ]
    text = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tk(text, return_tensors="pt")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        ROOT, dtype=torch.float32
    ).eval()
    with torch.inference_mode():
        first = model(**ids, logits_to_keep=1).logits[0, -1].float().numpy()
        t0 = time.time()
        gen = model.generate(
            **ids, do_sample=False, no_repeat_ngram_size=5, max_new_tokens=600
        )
    new = gen[0][ids["input_ids"].shape[1] :].numpy()
    print(f"generated {len(new)} tokens in {time.time() - t0:.0f}s")
    print(tk.decode(new, skip_special_tokens=True))
    np.savez(
        out,
        prompt_ids=ids["input_ids"][0].numpy(),
        first_logits=first,
        new_ids=new,
        text=np.array(tk.decode(new, skip_special_tokens=True)),
    )


def ours(out, ref_path):
    import mlx.core as mx

    sys.path.append(REPO)
    from slimserve.video.ltx25.enhancer import PromptEnhancer, chat_text, system_prompt

    r = np.load(ref_path)
    prompt_ids = r["prompt_ids"].tolist()
    e = PromptEnhancer().load()
    text = e.tokenizer.decode(prompt_ids[1:], skip_special_tokens=False)
    user = text.split("<|turn>user\n", 1)[1].split("<turn|>", 1)[0]
    ids = e.tokenize(chat_text(system_prompt("t2v"), user))
    print("prompt ids equal:", ids == prompt_ids, len(ids), len(prompt_ids))
    cache = [None] * len(e.cfg["layer_types"])
    logits = e._logits(e._forward(ids, 0, cache))
    mx.eval(logits)
    ours_first = np.array(logits).astype(np.float64)
    ref_first = r["first_logits"].astype(np.float64)
    rel = np.linalg.norm(ours_first - ref_first) / np.linalg.norm(ref_first)
    print(
        f"first logits rel-L2 {rel:.2e} max-abs {np.abs(ours_first - ref_first).max():.2e}"
        f" argmax {int(ours_first.argmax())} vs {int(ref_first.argmax())}"
    )
    t0 = time.time()
    new = e.generate(ids)
    print(f"generated {len(new)} tokens in {time.time() - t0:.1f}s")
    ref_new = r["new_ids"].tolist()
    eos = e.gen["eos_token_id"]
    if ref_new and ref_new[-1] in eos:  # generate() keeps the stop token; ours does not
        ref_new = ref_new[:-1]
    common = 0
    for a, b in zip(new, ref_new):
        if a != b:
            break
        common += 1
    print(
        f"ref {len(ref_new)} tokens; identical prefix {common}; equal: {new == ref_new}"
    )
    if new != ref_new:
        print("ours:", e.tokenizer.decode(new, skip_special_tokens=True))
        print("ref :", str(r["text"]))
    np.savez(
        out, new_ids=np.array(new), prompt_ids=np.array(ids), first_logits=ours_first
    )


def teacher_ref(out, ref_path):
    """Logits at every generated position of the reference sequence (one fp32
    forward over prompt + generation)."""
    import torch
    from transformers import Gemma4ForConditionalGeneration

    r = np.load(ref_path)
    seq = np.concatenate([r["prompt_ids"], r["new_ids"]])
    n_prompt = len(r["prompt_ids"])
    model = Gemma4ForConditionalGeneration.from_pretrained(
        ROOT, dtype=torch.float32
    ).eval()
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor(seq)[None]).logits[0].float().numpy()
    # logits[i] predicts token i + 1: positions n_prompt - 1 .. end - 1 predict the generation
    np.savez(out, logits=logits[n_prompt - 1 : -1], seq=seq, n_prompt=n_prompt)


def teacher_ours(ref_logits_path):
    """Our logits at the same positions (one prefill over the reference
    sequence); per-position error, and the reference's top-2 margin wherever
    our argmax differs."""
    import mlx.core as mx

    sys.path.append(REPO)
    from slimserve.video.ltx25.enhancer import PromptEnhancer

    t = np.load(ref_logits_path)
    seq, n_prompt, ref = (
        t["seq"].tolist(),
        int(t["n_prompt"]),
        t["logits"].astype(np.float64),
    )
    e = PromptEnhancer().load()
    cache = [None] * len(e.cfg["layer_types"])
    h = e._forward(seq, 0, cache)
    x = h[0, n_prompt - 1 : -1].astype(e.O)
    logits = (x @ e.w["embed_tokens.weight"].astype(e.O).T).astype(mx.float32)
    cap = e.cfg["final_logit_softcapping"]
    logits = cap * mx.tanh(logits / cap)
    mx.eval(logits)
    ours = np.array(logits).astype(np.float64)
    rel = np.linalg.norm(ours - ref, axis=1) / np.linalg.norm(ref, axis=1)
    print(
        f"{len(rel)} positions: rel-L2 median {np.median(rel):.2e} max {rel.max():.2e}"
    )
    flips = np.nonzero(ours.argmax(1) != ref.argmax(1))[0]
    print(f"argmax differs at {len(flips)} positions")
    for i in flips[:10]:
        top2 = np.sort(ref[i])[-2:]
        print(
            f"  gen step {i}: ref top-2 margin {top2[1] - top2[0]:.3f} logits;"
            f" ours picks {int(ours[i].argmax())} (ref {int(ref[i].argmax())})"
        )


if __name__ == "__main__":
    {
        "ref": ref,
        "ours": ours,
        "teacher-ref": teacher_ref,
        "teacher-ours": teacher_ours,
    }[sys.argv[1]](*sys.argv[2:])
