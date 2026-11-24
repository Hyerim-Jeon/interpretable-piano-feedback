# Interpretable Piano Performance Feedback — supplementary material

Data, outputs, and code for *Interpretable Piano Performance Feedback From Hierarchical and
Semantic Synergies in Features and Segments* (Jeon and Li, IEEE Access, under review).

For every analyzed performance, the repository shows which segments were selected, the level
of the evaluation ontology at which each remark is stated and why, the prompt sent to the
language model, and the remark it wrote.

## Performances

All 24 performances come from the MAESTRO dataset (v3.0.0,
<https://magenta.tensorflow.org/datasets/maestro>). Audio and MIDI are available there under the
file names below. Our earlier study's repository,
<https://github.com/Hyerim-Jeon/critical_segments>, describes the expert annotations of these
performances.

Beethoven: Thirty-Two Variations in C minor, WoO 80 (19 performances). Chopin: Étude Op. 10 No. 1 in C major (5 performances).

| Piece | Id | MAESTRO file |
|---|---|---|
| Beethoven | 61 | MIDI-Unprocessed_073_PIANO073_MID--AUDIO-split_07-08-17_Piano-e_2-02_wav--1 |
| Beethoven | 63 | MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_08_R2_2015_wav--1 |
| Beethoven | 64 | MIDI-Unprocessed_082_PIANO082_MID--AUDIO-split_07-09-17_Piano-e_2_-04_wav--1 |
| Beethoven | 66 | ORIG-MIDI_01_7_10_13_Group_MID--AUDIO_08_R3_2013_wav--1 |
| Beethoven | 68 | MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_19_R2_2015_wav--2 |
| Beethoven | 69 | ORIG-MIDI_02_7_10_13_Group_MID--AUDIO_12_R3_2013_wav--1 |
| Beethoven | 70 | MIDI-Unprocessed_04_R3_2008_01-07_ORIG_MID--AUDIO_04_R3_2008_wav--1 |
| Beethoven | 71 | MIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_12_R2_2015_wav--1 |
| Beethoven | 73 | MIDI-Unprocessed_22_R3_2011_MID--AUDIO_R3-D7_03_Track03_wav |
| Beethoven | 74 | ORIG-MIDI_01_7_10_13_Group_MID--AUDIO_07_R3_2013_wav--2 |
| Beethoven | 75 | MIDI-Unprocessed_071_PIANO071_MID--AUDIO-split_07-08-17_Piano-e_1-04_wav--1 |
| Beethoven | 76 | MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--4 |
| Beethoven | 78 | MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_11_R2_2015_wav--1 |
| Beethoven | 79 | MIDI-Unprocessed_02_R3_2008_01-03_ORIG_MID--AUDIO_02_R3_2008_wav--1 |
| Beethoven | 81 | MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_22_R2_2015_wav--2 |
| Beethoven | 82 | MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_21_R2_2015_wav--2 |
| Beethoven | 83 | MIDI-Unprocessed_12_R3_2008_01-04_ORIG_MID--AUDIO_12_R3_2008_wav--1 |
| Beethoven | 84 | MIDI-Unprocessed_17_R3_2011_MID--AUDIO_R3-D6_02_Track02_wav |
| Beethoven | 85 | MIDI-Unprocessed_075_PIANO075_MID--AUDIO-split_07-08-17_Piano-e_2-06_wav--2 |
| Chopin | 53 | MIDI-Unprocessed_043_PIANO043_MID--AUDIO-split_07-06-17_Piano-e_1-03_wav--3 |
| Chopin | 54 | MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_12_R1_2015_wav--4 |
| Chopin | 55 | ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_19_R1_2013_wav--3 |
| Chopin | 59 | MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_03_R1_2015_wav--2 |
| Chopin | 60 | MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_10_R1_2015_wav--2 |

## Contents

| Path | Content |
|---|---|
| `outputs/case_items.json` | For every performance: its critical segments, each with its share of the attributed weakness. It also gives the three selected feedback items (budget B = 3), each with its level and node, the reason for the level (diffuseness versus largest share), and the direction of every deviation (performance versus expert norm). |
| `outputs/prompt_results.jsonl` | For every selected item, under both prompt variants (`numeric`, `verbal`): the prompt, the generated remark, and the judge's reading of the remark's direction for each aspect. |
| `outputs/level_aware/` | Level-aware selection: coverage per budget, ablation, β sensitivity, per-performance results, and the example passages. |
| `outputs/simulation/` | Simulated feedback effectiveness, per performance (`table4.csv`) and aggregated (`table5.csv`). |
| `model/general_dag.json` | The estimated evaluation topology (edge weights and directions). |
| `data/` | Inputs to the scripts: performance features, SHAP values, guideline comparisons, expert annotations, and the feature-to-descriptor mapping. |
| `figures/` | Supplementary figure: per-performance effectiveness against the baselines |
| `scripts/` | Code that produces every number and figure (see below). |

### `data/`

| Path | Content |
|---|---|
| `shaptemp.csv` | Per-measure performance features with predicted descriptor scores (input to the topology and attribution) |
| `shap_v1/`, `overallscore_v1/` | Per-performance SHAP values for the 18 descriptors and for the overall score |
| `result_v1_b/`, `result_v1_c/` | Per-performance measure-level comparison with the expert guideline: Beethoven (`b`) and Chopin (`c`) |
| `beethoven1.csv`, `beethoven2.csv`, `chopin1.csv`, `chopin2.csv` | Expert annotations (passages and descriptor items) used for the expert comparison |
| `feature_mapping.csv` | Mapping between annotation items and the descriptors of the ontology |
| `pidname.csv` | Performance id to MAESTRO file name |
| `general_dag.json` | Copy of `model/general_dag.json`, so that the scripts can run from this folder |

## Reproducing the results

The scripts read their inputs from the working directory. Run them from `data/`:

```
cd data
python ../scripts/reproduce_table4.py                # simulated feedback effectiveness (table4.csv, table5.csv)
python ../scripts/level_aware_selection.py           # level-aware selection, ablation, expert comparison (la_*.csv)
python ../scripts/case_study.py 3                    # case_items.json
OPENAI_API_KEY=... python ../scripts/prompt_experiment.py   # prompt_results.jsonl and the prompt comparison
python ../scripts/prompt_experiment.py summary       # the comparison only, from an existing prompt_results.jsonl
python ../scripts/make_case_figure.py case_items.json fig_case_timeline
```

To fill `score_context` (the notated directives of a passage), `case_study.py` reads a MusicXML
score named `Chopin.musicxml` from the working directory. The score is not redistributed here.
Without it, `score_context` is left empty. The released `case_items.json` was produced with the
score present.

The language model `gpt-4o-mini` (temperature 0.2) writes the remarks, and `gpt-4o`
(temperature 0) reads the direction of each remark. Generated text may differ slightly from
run to run.

Requirements: Python ≥ 3.6, numpy, pandas, scipy, scikit-learn, xgboost, ruptures, requests,
matplotlib, and music21 (only for `score_context`).

## License and attribution

The code in `scripts/` is released under the MIT License (see `LICENSE`). The data and outputs in
`data/`, `outputs/`, `model/`, and `figures/` are released under CC BY-NC-SA 4.0, because they derive
from MAESTRO performances, which their authors distribute under CC BY-NC-SA 4.0, and from the
PercePiano annotations. This repository links to MAESTRO and does not redistribute audio or MIDI.
