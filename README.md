# localllm
Test different open source local LLM on sentiment analysis of stock-related news

# Installation Instructions

llama cpp submodule that is installed during `pip install llama-cpp-python` isn't enabled for Qwen3 models. As such, we have to pip install from source instead of from `pypi`:

1. Git clone `llama-cpp-python` and changed to `llama-cpp-python` directory:

```bash
git clone --recursive https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
```

Note:
- `--recursive` is required to ensure all commits are cloned.

2. Update llama cpp to Qwen3-compatible version:

```bash
cd vendor/llama.cpp
git fetch origin
git checkout b5092
cd ../..
```

Note:
- Commit `b5092` is with Qwen3 support.
- Need to git fetch to get latest updates.

3. Enabled Qwen3 in build

```bash
CMAKE_ARGS="-DLLAMA_QWEN=ON" pip install -v .
```

4. Remove `llama-cpp-python` repo once we are able to load GGUF file:

```bash
cd ..
rm -rf llama-cpp-python
```


