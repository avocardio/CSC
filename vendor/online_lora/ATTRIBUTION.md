# Vendored from Online-LoRA (WACV 2025)

Source: https://github.com/Christina200/Online-LoRA-official
Paper: https://arxiv.org/abs/2411.05663
Authors: Xiwen Wei, Guihong Li, Radu Marculescu

We vendor the `Disjoint/` directory verbatim from commit master @ 2026-04-29
to compare CSC's bit-depth signal against their MAS-style omega importance
weights. Modifications below are non-trivial and affect engine.py and lora.py:

  - Add `--importance` flag with options {mas, bd}. When `bd`, the omega
    arrays are populated from per-channel accumulated bit-depth of CSC
    quantizers attached to the LoRA wnew layers (broadcast over fan-in).
  - Wrap `wnew_a_*` and `wnew_b_*` Linear layers with our QuantizedLinear
    when `--csc` flag is set. The compression objective γ·Q is added to
    the SAC actor loss.
  - All other logic preserved 1:1.

Original code is BSD-licensed (per their environment.yml MIT-style). We
keep their copyright headers intact.
