# YouTube upload package

**File to upload:** `submission/demo_final.mp4`
- 1920x1080, 30 fps, H.264, 2:45.
- English AI narration (synthetic voice, Kokoro-82M) with AAC audio.
- English subtitles are burned in and also embedded. For YouTube, upload `submission/demo_final.srt` as English captions.
- Thumbnail: `submission/youtube_thumbnail.png` (1280x720).

**Recommended visibility: Public.** The rules ask for a publicly visible video. "Unlisted" may be accepted, but Public removes any doubt.

## Title (max 100 characters)

Cosmos-PLV: Robot Policy Evaluation with an Audited AI Judge (NVIDIA Cosmos + Nemotron on Nebius)

## Description

Cosmos-PLV helps robot-learning researchers decide which imitation-learning checkpoint to test next, and shows when the AI evaluator itself cannot be trusted.

What you see in this video:
- Real screen recordings of the running Streamlit dashboard.
- A live run of the original PLV screening script.
- NVIDIA Cosmos3-Edge running on a local RTX 3090.
- A live call to NVIDIA Nemotron-3.5-Super-VL on Nebius Token Factory.
- Real robosuite / MuJoCo rollouts.
- Speed-ups are labelled in the captions.

Measured results:
- Zero-shot success judging on 330 recorded simulator rollouts: Nemotron 96.4% accuracy (base rate 79.7%), Cosmos3-Edge 91.8%.
- Exploratory, post-hoc analyses:
  - 2-3x fewer rollouts with adaptive allocation;
  - 16% robot time saved by a time-limit abort;
  - 16% fewer Nebius calls with a local-first cascade.
- Pre-registered audit (c1): Nemotron called 32 of 33 "can in the wrong compartment" failures a success.
- Pre-registered self-training test (s2):
  - Training on Nemotron-filtered self-collected data lowered success by 10.4 points versus continued training (95% CI -14.8 to -6.0).
  - 36 extra human demos raised it by 8.2 points.
- Honest negative: no PLV metric beat the free "latest epoch" pick in these runs.

Code, data and pre-registered protocols: https://github.com/cheesss/cosmos-plv

Built for the Nebius x NVIDIA Global AI Hackathon 2026 (Physical AI track). The narration is a synthetic voice; every screen recording is real.

Chapters:
0:00 The problem: costly evaluation, overconfident judges
0:09 What Cosmos-PLV is
0:17 Architecture: local GPU + Nebius Token Factory
0:27 Live dashboard: recommendation and judge-trust warning
0:40 Live PLV screening
0:47 Recorded robosuite rollout
0:56 Live NVIDIA Cosmos3-Edge (local GPU)
1:09 Live NVIDIA Nemotron on Nebius Token Factory
1:21 Results on 330 rollouts
1:40 Exploratory efficiency results
1:51 Pre-registered judge audits
2:03 A real wrong-compartment failure judged 'success'
2:12 Consequence for self-training (s2)
2:27 Cost
2:33 Takeaway

## After uploading

1. Open the watch URL in a private or incognito window, logged out, and check that it plays.
2. Send the watch URL (https://www.youtube.com/watch?v=...). It goes into the README, `submission/devpost_final.md` and the Devpost video field.
