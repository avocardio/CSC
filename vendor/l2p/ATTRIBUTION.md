# Vendored from L2P (CVPR 2022)

Source: https://github.com/JH-LEE-KR/l2p-pytorch
Paper: Wang et al., "Learning to Prompt for Continual Learning", CVPR 2022.

We vendor unmodified for the comparison table in our paper. Their code
trains a fixed pretrained ViT-B/16 with a learnable prompt pool; the
backbone is frozen, only prompts + classification head update.
