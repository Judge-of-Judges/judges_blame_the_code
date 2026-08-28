"""Ask an endpoint what it actually serves.

    python -m harness.list_models --contains qwen
    python -m harness.list_models --endpoint OLLAMA_BASE_URL

Model ids churn between releases and providers. Look them up rather than
trusting the strings in config.SPECS.
"""
import argparse
from . import providers

ENDPOINTS = {
    "openrouter": ("OPENROUTER_BASE_URL", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "ollama": ("OLLAMA_BASE_URL", "OLLAMA_API_KEY", "http://localhost:11434/v1"),
    "together": ("TOGETHER_BASE_URL", "TOGETHER_API_KEY", "https://api.together.xyz/v1"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="openrouter", choices=list(ENDPOINTS))
    ap.add_argument("--contains", default="")
    args = ap.parse_args()
    import os
    url_env, key_env, default = ENDPOINTS[args.endpoint]
    os.environ.setdefault(url_env, default)
    try:
        ids = providers.list_models(url_env, key_env, args.contains)
    except Exception as e:
        raise SystemExit(f"{type(e).__name__}: {e}")
    print(f"{len(ids)} models at {os.environ[url_env]} matching {args.contains!r}:")
    for i in ids:
        print(" ", i)

if __name__ == "__main__":
    main()
