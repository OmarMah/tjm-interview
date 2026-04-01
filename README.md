# tjm-interview

Current project status:
- `pyproject.toml` for `uv`/packaging/dev tooling
- `config/` for global and target-specific TOML settings
- `src/tjm_automation/automation/desktop.py` for Windows desktop actions
- `src/tjm_automation/grounding/` for pluggable target grounding backends
- `src/tjm_automation/targets/` for thin target-specific adapters such as Notepad

Vision LLM setup:

```powershell
Copy-Item .env.example .env
```

Set these env values in `.env`:
- `TJM_VLM_BASE_URL`
- `TJM_VLM_API_KEY`
- `TJM_VLM_MODEL`

The grounding backend uses the same OpenAI-compatible request format for OpenAI, Gemini, and Groq. Switching providers should only require changing those env values.

Useful commands:

```powershell
python -m tjm_automation check-config
python -m tjm_automation show-config
python -m tjm_automation version
python -m tjm_automation capture-screenshot --show-desktop
python -m tjm_automation check-window --title Notepad
python -m tjm_automation cursor-position
python -m tjm_automation click-point 500 400 --double
python -m tjm_automation detect-target --show-desktop
python -m tjm_automation launch-target
python -m tjm_automation write-sample
```
