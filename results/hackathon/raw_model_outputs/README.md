# Raw model outputs (authentic, unedited)

One JSON object per (episode, judge) with the full raw generation (`raw_text`),
the parsed and validated fields (`cosmos_*`, `parse_status`, `parse_errors`),
the verified simulator labels (`truth_*`, refreshed with the settled-height
annotator), latency, token counts, model id and revision, Nebius request id
(`request_id`) and the backend parameters.

| file | judge | split |
|---|---|---|
| eval_local_v6.jsonl | nvidia/Cosmos3-Edge @344d602b, last frame, task-free perception | eval, 330 episodes |
| eval_local_v7.jsonl | nvidia/Cosmos3-Edge, 8 keyframes (field `track.frames[*].raw_text`) | eval, 330 episodes |
| eval_nebius_v2.jsonl | nvidia/Nemotron-3.5-Super-VL via Nebius Token Factory, whole video | eval, 330 episodes |
| eval_nebius_v2_pad201.jsonl | same, length-neutral clips | eval, 330 episodes |
| dev_prompt_selection.jsonl, dev_nebius_v2_pad201.jsonl | all prompt variants tried | dev, 36 episodes |
| dev_infra_failures.jsonl | 23 rows lost to CUDA out-of-memory (infrastructure failures, not model outputs) | dev |

Videos and keyframe images referenced by path live under the git-ignored
`data/outputs/` tree; a selection is in `media/hackathon/`.
