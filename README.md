# Aura Lily

Aura Lily is an ESP32 device bridge for running a local or remote Hermes agent
through the public `hermes` CLI.

The Lily release has one job: let an ESP32 voice/device front end hand a task
to Hermes Agent and receive a result back, without patching Hermes source
files.

## What This Is

- ESP32-S3 firmware for the Aura device path.
- A small standalone Hermes CLI bridge in `integrations/hermes_lily_cli/`.
- An optional Lily-native Aura persona gateway in
  `integrations/aura_persona_gateway/`.
- Shared protocol and provider configuration helpers.
- A release allowlist tool that exports only the Lily-safe files.

## What This Is Not

Aura Lily does not include the closed desktop Aura app.

It also does not include:

- Aura's old full life/persona/world engine.
- social, moments, reading, film, image generation, or companion simulation.
- the Tauri desktop app.
- the old LazyCat/WebApp deployment path.
- any patch or source modification to Hermes itself.

## Requirements

- Python 3.11+
- An installed `hermes` command available on `PATH`
- Hermes already configured with at least one working provider/model
- ESP-IDF for firmware builds
- Docker Compose for reproducible local deployment

Check Hermes first:

```bash
hermes status
hermes -z "请只回复：Hermes 可用。"
```

## Quick Start

Run a Lily bridge task:

```bash
python -m integrations.hermes_lily_cli.cli \
  --provider deepseek \
  --model deepseek-v4-pro \
  "你能帮我做什么"
```

Or send a JSON request:

```bash
echo '{"goal":"查一下今天上海天气","provider":"deepseek","model":"deepseek-v4-pro"}' \
  | python -m integrations.hermes_lily_cli.cli --json-input
```

Run the local ESP32/Mini HTTP bridge:

```bash
python -m integrations.hermes_lily_cli.server \
  --host 0.0.0.0 \
  --port 8765 \
  --provider deepseek \
  --model deepseek-v4-pro \
  --max-concurrency 1 \
  --queue-timeout 30
```

Then send one turn:

```bash
curl -s http://127.0.0.1:8765/turn \
  -H 'content-type: application/json' \
  -d '{"goal":"你能帮我做什么"}'
```

The bridge returns JSON:

```json
{
  "ok": true,
  "status": "completed",
  "response": "...",
  "request_id": "lily-...",
  "latency_ms": 1234,
  "evidence": {
    "returncode": 0,
    "command": ["hermes", "-z", "<prompt>", "..."]
  }
}
```

The command evidence hides the user prompt and scrubs common secret patterns.

## Aura Persona Gateway

Clean Lily can optionally add Aura's persona/state layer before sending a turn
to Hermes. This is disabled by default so the plain Hermes bridge stays small
and predictable.

Enable it with local runtime volumes:

```bash
AURA_PERSONA_ENABLED=1 docker compose up --build aura-lily
```

The gateway reads private files from the existing ignored volumes:

```text
.docker/aura-persona/persona/soul.md  # optional; empty by default
.docker/aura-companion/companion.db
```

No Soul file is bundled or imported from Hermes. Until the user explicitly
creates `persona/soul.md`, the Soul content remains empty. When configured, the
gateway can use that Soul together with current relationship/state values, recent useful IM
history, today's plan, the latest moment, location/weather deixis resolution,
ESP32 voice-turn policy, proactive-message signals, spending signals, and a
new Lily debug event. It intentionally does not import the old Aura engine,
agent runtime, friend/social ring, or cultural book chunks.

Explicit persona endpoint:

```bash
curl -s http://127.0.0.1:8765/persona/turn \
  -H 'content-type: application/json' \
  -d '{"goal":"你那边天气怎么样"}'
```

Persona and Lily admin/config endpoints require a local admin login. Use
`AURA_LILY_ADMIN_USER` and `AURA_LILY_ADMIN_PASSWORD`; the old
`AURA_LILY_ADMIN_TOKEN`/`AURA_PERSONA_ADMIN_TOKEN` variables remain accepted
only as compatibility credentials. The admin login is not a Hermes provider API
key.

After the HTTP bridge is running, open the built-in local admin page:

```text
http://127.0.0.1:8765/admin
```

Log in on that page to manage Hermes model selection, Aura main model
selection, ASR/TTS speech models, persona switches, Lily/user locations, the
main soul text, and relationship state. Hermes upstream settings configure the
Hermes execution bridge. Aura runtime settings decide whether the persona layer
reuses the Hermes main model or uses an independent Aura model; local short
replies are only voice shortcuts. ASR defaults to a local model but can be
switched to an API provider. The page writes runtime files under the mounted
private `.docker/` volumes. LLM/TTS/ASR API keys stay private by default and
are revealed only from authenticated admin secret endpoints. The admin keeps a
small history of non-secret LLM/TTS/ASR settings so previous providers/models
can be reapplied without retyping.

```bash
curl -s http://127.0.0.1:8765/persona/health
curl -s http://127.0.0.1:8765/persona/config \
  -u "$AURA_LILY_ADMIN_USER:$AURA_LILY_ADMIN_PASSWORD"
```

Run a full local smoke check:

```bash
python -m integrations.hermes_lily_cli.smoke_test \
  --provider deepseek \
  --model deepseek-v4-pro \
  --timeout 90
```

The smoke test checks both CLI execution and the HTTP `/health` + `/turn`
contract against an ephemeral local server.

## Docker Local Deployment

Docker is the reproducible runtime boundary for Aura Lily. It pins the Python
version, package dependencies, entrypoints, ports, and volume layout so a cloned
project starts the same way on every machine. It does not magically configure a
user's provider keys, Hermes account, or LAN IP; those remain local settings.

You can start with defaults:

```bash
docker compose up --build
```

Docker Compose automatically reads `.env` when it exists. For real
provider/model settings, create a local environment file:

```bash
cp .env.example .env
```

Then edit `.env` and restart. To run only the HTTP bridge:

```bash
docker compose up --build aura-lily
```

Check it:

```bash
curl -s http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8765/turn \
  -H 'content-type: application/json' \
  -d '{"goal":"请回复：Aura Lily Docker OK"}'
```

The compose file also includes `aura-lily-gateway` on port `8787`. The gateway
accepts ESP32 WebSocket audio and supports StepFun Plan-covered speech paths:

- StepFun Step Plan Realtime (`stepaudio-2.5-realtime`) opens one upstream
  WebSocket at recording start, forwards audio chunks immediately, and sends
  upstream `response.audio.delta` frames directly to the ESP32 as ATTS audio.
  This is an experimental latency comparison route: it bypasses Aura/Lily
  semantic and persona handling, and is disabled unless
  `AURA_STEPFUN_REALTIME_DIRECT_REPLY_ENABLED=1`.
- StepFun Step Plan ASR (`stepaudio-2.5-asr`) is HTTP+SSE and can be used as the
  subscription-covered production route. It submits audio after recording stops,
  then routes the transcript through Aura/Lily LLM and StepFun streaming TTS.

The older StepFun realtime ASR preset (`stepaudio-2.5-asr-stream`) uses the
non-Plan `/v1` realtime ASR route, but only for transcription. This is the
Xiaozhi-style semantic streaming path: audio is streamed to ASR, the transcript
still goes through Aura/Lily, and StepFun streaming TTS starts as soon as Aura
emits the first short segment. Confirm the account's billing and permission
behavior before using it as the main path.

To verify the same WebSocket path that the firmware uses, run the gateway and
benchmark Opus frames against port `8787`:

```bash
python3 tools/voice_latency_benchmark.py \
  --mode voice-ws \
  --audio-format opus \
  --audio-ms 1300 \
  --frame-ms 60 \
  --realtime-upload
```

Watch `first_audio_after_stop_ms` and
`timing.realtime_first_audio_after_response_ms`. The first value is the user
experience after recording stops; the second isolates StepFun's response audio
startup after the gateway sends `response.create`.

Firmware builds also log `VOICE_TIMING` over serial for physical-device timing.
For one turn, compare `listen_start_sent`, `upload_first_packet`,
`upload_stop_sent`, `first_tts_frame`, `first_pcm_queued`,
`speaker_first_write`, and `playback_done`. That separates microphone upload,
StepFun response startup, device buffering, and actual I2S speaker output.

For a physical ESP32, set the server URL in the provisioning page to your
computer's LAN or Tailscale address, for example:

```text
ws://192.168.1.23:8787/ws
```

Do not use `127.0.0.1` from the ESP32; that points back to the device itself.

### Hermes In Docker

Your existing macOS `hermes` install is fine for fast local development. The
Docker image installs a Linux-compatible `hermes-agent` package by default. A
Linux container cannot run your macOS virtualenv entrypoint directly.

Compose mounts local, private runtime directories into the container:

```text
.docker/hermes-home     -> /home/aura/.hermes
.docker/aura-companion  -> /data/aura-companion
.docker/aura-persona    -> /data/aura-persona
```

Put private runtime state there, such as provider config, optional Aura persona
text assets, and `companion.db`. The `.docker/` directory is
intentionally ignored by Git and is not part of the release export. Public
source should only document how to mount persona/config files, not include a
private persona or state database by default.

The included `Dockerfile` can also install Hermes from a source tree inside the
build context:

```bash
HERMES_AGENT_SOURCE=vendor/hermes-agent docker compose build
```

If `hermes` is not available or not configured inside the container, `/health`
still works, while `/turn` returns a clear process or provider configuration
failure.

## Provider Configuration

Aura Lily does not store provider keys itself. Configure providers through
Hermes, then call Hermes through this bridge.

Typical flow:

```bash
hermes model
hermes status
python -m integrations.hermes_lily_cli.cli "说一句测试回复"
```

Lily can expose the current Hermes runtime selection to a future admin UI:

```bash
curl -s http://127.0.0.1:8765/admin/hermes/config \
  -u "$AURA_LILY_ADMIN_USER:$AURA_LILY_ADMIN_PASSWORD"
```

Update only the runtime selection:

```bash
curl -s http://127.0.0.1:8765/admin/hermes/config \
  -H 'content-type: application/json' \
  -u "$AURA_LILY_ADMIN_USER:$AURA_LILY_ADMIN_PASSWORD" \
  -d '{"provider":"deepseek","model":"deepseek-v4-pro"}'
```

In Docker this persists to
`/data/aura-persona/config/hermes_runtime.json` by default. This file stores
provider/model/toolset choices, not API keys.

Hermes currently exposes `hermes model` as an interactive picker rather than a
stable machine-readable model catalog. For a backend dropdown, set
`HERMES_MODEL_OPTIONS` in `.env`, for example:

```text
HERMES_MODEL_OPTIONS=deepseek:deepseek-v4-pro|DeepSeek V4,openrouter:anthropic/claude-sonnet-4.6|Claude Sonnet
```

You can override provider and model per request:

```bash
python -m integrations.hermes_lily_cli.cli \
  --provider deepseek \
  --model deepseek-v4-pro \
  "帮我列一个 ESP32 接入测试清单"
```

## Firmware

ESP32 firmware lives in:

```text
firmware/esp32/
```

Build artifacts, ESP-IDF managed components, and local caches are intentionally
not part of the release package.

The default WebSocket endpoint is configured by ESP-IDF Kconfig:

```text
CONFIG_AURA_WS_URI_DEFAULT="ws://192.168.1.100:8787/ws"
```

The device provisioning page can save a different `ws://` or `wss://` server URL
to NVS. Saved values take precedence over the compile-time default.

## Release Export

The current repo contains internal app work, historical docs, generated files,
and old overlay code. Do not publish this working tree directly.

Use the release allowlist:

```bash
python3 tools/prepare_mini_release.py --list
python3 tools/prepare_mini_release.py \
  --export \
  --dest /tmp/aura-lily-export \
  --scan
```

Expected result:

```text
scan: ok
```

The export script checks for:

- private key and token patterns
- private endpoint patterns
- closed desktop/life/persona/social imports
- accidental inclusion of the old desktop app path

## Development Gates

Before sharing an Aura Lily repository:

```bash
python3 -m pytest tests/test_hermes_lily_cli.py
python3 tools/prepare_mini_release.py --export --dest /tmp/aura-lily-export --scan
```

The first public or semi-public Aura Lily repository should be created from the
export directory, not from this mixed development workspace.

## Repository Map

```text
firmware/esp32/                 ESP32 firmware
integrations/hermes_lily_cli/    standalone no-patch Hermes CLI bridge
Dockerfile                       local deployment image
docker-compose.yml               local bridge and gateway services
tools/prepare_mini_release.py    Lily export and boundary scanner
```

## License

MIT. See [LICENSE](LICENSE).
