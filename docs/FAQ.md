# Frequently Asked Questions

## What is Qythera?
Qythera is an open-source production superintelligence platform built on the Vaelon model architecture.

## What hardware do I need?
- 7B model: 1x RTX 4090 or better
- 13B model: 2x RTX 4090 or 1x A100
- 70B model: 4x A100 80GB

## Can I run it on CPU?
Yes, with reduced performance. Use quantized models (INT4) for better CPU performance.

## How do I fine-tune?
Use the training pipeline: `qythera train --config training/configs/7b_lora.yaml`

## Does it support multiple languages?
Yes, Qythera supports 12+ languages including English, Spanish, French, German, Japanese, Chinese, and more.

## Is there a mobile app?
Yes, React Native support is included for iOS and Android.

## How do I contribute?
See CONTRIBUTING.md in the docs/ directory.
