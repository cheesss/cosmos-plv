# Video validation — `submission/demo_final.mp4`

`media/hackathon/demo.mp4` is a byte-identical copy.

## Format (ffprobe)

```
stream|index=0|codec_name=h264|profile=High|codec_type=video|width=1920|height=1080|pix_fmt=yuv420p|r_frame_rate=30/1|tag:language=und
stream|index=1|codec_name=aac|profile=LC|codec_type=audio|sample_rate=48000|channels=1|r_frame_rate=0/0|tag:language=und
stream|index=2|codec_name=mov_text|profile=unknown|codec_type=subtitle|width=N/A|height=N/A|r_frame_rate=0/0|tag:language=eng
format|duration=165.222000|size=12949238|bit_rate=626998
[Parsed_volumedetect_0 @ 0x555c8f9fad80] mean_volume: -17.9 dB
[Parsed_volumedetect_0 @ 0x555c8f9fad80] max_volume: -1.4 dB
```

| check | result |
|---|---|
| Duration | **165.2 s (2:45.2)**. Under 180 s: PASS |
| Resolution / frame rate / codec | 1920x1080, 30 fps, H.264 High, yuv420p, faststart |
| Audio | AAC 48 kHz, mono. Synthetic English narration made locally with Kokoro-82M (Apache-2.0), loudness-normalized (EBU R128 target −16 LUFS). No music. |
| Subtitles | Burned into the caption strip below the picture, so they never cover the robot or the UI. Also embedded as an English `mov_text` track and as `submission/demo_final.srt`. 26 sentence cues are timed to the speech; every narration line ends before its segment ends (`submission/demo_final.narration.json`, all `fits: true`). |
| Real footage | **84.9 s** of real application and simulator footage. The longest continuous stretch of working modules is **0:27.9–1:31.2 (63.3 s)**: live recommendation, live PLV script, rollout viewer, live Cosmos3-Edge call, live Nemotron call on Nebius, results tab. More follows at 1:51.6–2:03.8 (audit panel). PASS (≥ 60 s). |
| First 10 s | A real robosuite rollout pair (failure vs success) with the problem statement. |
| Live inference evidence | The Nemotron verdict appears with its Nebius request id, latency and tokens. The recorder call was logged in `results/hackathon/nebius_calls.jsonl`. Cosmos3-Edge shows its model id and revision. |
| Speed-ups | Labelled in a top-left tag on every clip segment. |
| Readability | Frames downscaled to 640x360 (phone size) were checked: subtitles and figure titles stay readable. The fine dashboard tables are small at that size, which is expected for a 1080p screen recording. |
| Fabrication | None. Every clip comes from the recorded dashboard session `data/outputs/hackathon_demo/session.webm` or from recorded rollout videos. Figures are generated from result files (`scripts/hackathon/make_submission_figures.py`). The Nemotron answer shown at 2:03.8 is the logged answer for request `chatcmpl-a9fea12aad4fc59a`. |

## Segment list

| time | kind | real footage | on-screen tag / title | source |
|---|---|---|---|---|
| 0:00.0–0:09.3 | sidebyside | yes | REAL robosuite / MuJoCo rollouts recorded by the pipeline | `data/outputs/hackathon_rollouts_eval/can_s1_e0000/videos/seed200000.mp4, data/outputs/hackathon_rollouts_eval/can_s2_e0100/videos/seed200008.mp4` |
| 0:09.3–0:17.2 | card | no | Cosmos-PLV | `(title card)` |
| 0:17.2–0:27.9 | image | no | system architecture | `submission/figures/system_architecture.png` |
| 0:27.9–0:40.2 | clip | yes | LIVE screen recording - 1x | `data/outputs/hackathon_demo/session.webm [12.7–25.1 s of the session, 1.0x]` |
| 0:40.2–0:47.5 | clip | yes | LIVE - original PLV metric script - 2.4x | `data/outputs/hackathon_demo/session.webm [27.5–44.9 s of the session, 2.4x]` |
| 0:47.5–0:56.9 | clip | yes | LIVE screen recording - recorded rollout video - 1.9x | `data/outputs/hackathon_demo/session.webm [52.6–70.4 s of the session, 1.9x]` |
| 0:56.9–1:05.6 | clip | yes | LIVE - NVIDIA Cosmos3-Edge on the local RTX 3090 - 3x while waiting | `data/outputs/hackathon_demo/session.webm [70.4–96.3 s of the session, 3.0x]` |
| 1:05.6–1:09.8 | clip | yes | LIVE - Cosmos3-Edge answer - 1x | `data/outputs/hackathon_demo/session.webm [96.3–100.5 s of the session, 1.0x]` |
| 1:09.8–1:16.3 | clip | yes | LIVE - request to Nebius Token Factory - 2x while waiting | `data/outputs/hackathon_demo/session.webm [99.8–112.8 s of the session, 2.0x]` |
| 1:16.3–1:21.7 | clip | yes | LIVE - Nemotron-3.5-Super-VL verdict from Nebius (request id, latency, tokens) - 1x | `data/outputs/hackathon_demo/session.webm [112.8–118.2 s of the session, 1.0x]` |
| 1:21.7–1:31.2 | clip | yes | LIVE screen recording - 1.7x | `data/outputs/hackathon_demo/session.webm [121.2–137.4 s of the session, 1.7x]` |
| 1:31.2–1:40.1 | image | no | measured: 330 recorded rollouts | `submission/figures/judge_accuracy.png` |
| 1:40.1–1:51.6 | image | no | EXPLORATORY, post-hoc | `submission/figures/evaluation_efficiency.png` |
| 1:51.6–2:03.8 | clip | yes | LIVE screen recording - pre-registered audits - 1.15x | `data/outputs/hackathon_demo/session.webm [140.8–154.9 s of the session, 1.15x]` |
| 2:03.8–2:12.0 | image | no | real c1 episode and Nemotron's recorded answer | `submission/figures/real_case_wrong_compartment.png` |
| 2:12.0–2:27.8 | image | no | pre-registered s2 study | `submission/figures/accuracy_vs_blind_spot.png` |
| 2:27.8–2:33.6 | image | no | measured from the call logs | `media/hackathon/cost_latency.png` |
| 2:33.6–2:45.2 | card | no | Audit the judge before you train on it | `(title card)` |

## Known limitations

* The narration voice is synthetic, not a person.
* The recorder's "done" timestamps fire about 3.5 s before the live answer is drawn, because Streamlit replaces the spinner node. The live-answer segments were therefore cut at frame-checked offsets (`scripts/hackathon/demo/build_demo_plan.py`).
* At phone size the dashboard tables are too small to read. Subtitles and figures carry the message.

## Rebuild

```bash
source scripts/hackathon/envb.sh
python scripts/hackathon/demo/record_dashboard.py             # live session (makes 1 Nebius call, runs Cosmos locally)
python scripts/hackathon/make_submission_figures.py
~/.venvs/tts/bin/python scripts/hackathon/demo/make_narration.py synth
python scripts/hackathon/demo/build_demo_plan.py
python scripts/hackathon/demo/make_demo_video.py --plan configs/hackathon/demo_plan.json \
       --out submission/demo_final_silent.mp4 --size 1920x1080 --fps 30 --strip
~/.venvs/tts/bin/python scripts/hackathon/demo/make_narration.py mux --video submission/demo_final_silent.mp4 --out submission/demo_final.mp4
```
