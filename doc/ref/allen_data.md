# Allen Brain Observatory — Visual Coding Neuropixels (summary)

For a general overview see [Allen Brain page](https://portal.brain-map.org/circuits-behavior/visual-coding-neuropixels?).

**Setup.** Awake, **head-fixed** mice viewed visual stimuli while positioned on a running wheel (free to run or sit still). Recordings were made with **Neuropixels** high-density **extracellular electrophysiology (Ecephys)** probes across visual cortex and thalamus. ([PMC][1], [allensdk.readthedocs.io][2])

**What’s recorded.** NWB files include spike-sorted **units**, stimulus timing/IDs, and behavior/physiology streams such as **running speed**, **pupil diameter**, and **pupil position**, plus LFP. ([allensdk.readthedocs.io][2])

**Why “units” (not neurons).** With extracellular probes, spikes from nearby cells can mix and waveform drift can occur; after spike sorting (Kilosort2), activity is grouped into **units** that may correspond to single cells or mixtures. Quality metrics are provided so you can choose isolation thresholds. ([allensdk.readthedocs.io][2])

**Stimuli.** Animals viewed one of two standardized sets (“**Brain Observatory 1.1**” or “**Functional Connectivity**”). For **Natural Scenes**, the library contains **118 grayscale images**, typically shown for **250 ms** and repeated about **50** times in randomized order. ([allensdk.readthedocs.io][2], [observatory.brain-map.org][3])

### Key terms (concise)

* **High-density extracellular probes.** Electrodes sit **outside** cells and detect brief voltage deflections from nearby spikes. High site density lets one probe sample many nearby neurons; the raw signal is a mixture that requires spike sorting. (Background; see Allen page for processing overview.) ([allensdk.readthedocs.io][2])
* **Spike sorting → units.** Detect spikes → extract waveform features → cluster → curate; output clusters are called **units** (single-unit or multi-unit). ([allensdk.readthedocs.io][2])


## Dataset structure (for our project)

The following shapes reflect the data used by Nayebi (not the NWB storage), hence the ones that we use in our project:

* `stimuli` — 118 grayscale images: **(image\_idx=118, height=918, width=1174)**.
* `neural_data` — binned spike responses: **(trials=50, image\_idx=118, time\_bins=25, units=8301)**.

**Coordinates**

* `trials`: repeat presentations per image (≈50). ([observatory.brain-map.org][3])
* `frame_id`: natural-scene index (0–117). ([observatory.brain-map.org][3])
* `time_relative_to_stimulus_onset`: your 10 ms bins from 5–245 ms (project choice).
* `units`: spike-sorted clusters.

**Per-unit metadata**

* `unit_id` — unique unit identifier.
* `visual_area` — e.g., VISp, LM, AL, etc.
* `specimen_id` — Allen’s internal ID for the **animal** in that session.
* `image_selectivity_ns` — selectivity across natural scenes.
* `run_mod_ns`, **`p_run_mod_ns`** — running modulation and its p-value during natural scenes. ([allensdk.readthedocs.io][2])

**Reliability**

* `splithalf_r_mean`, `splithalf_r_std` — split-half correlation of responses across repeats (consistency over trials). *(Not an official Allen “precomputed metric” field; it’s a standard reliability measure you computed.)*



[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10399640/?utm_source=chatgpt.com "Survey of spiking in the mouse visual system reveals functional hierarchy - PMC"
[2]: https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html "Visual Coding – Neuropixels — Allen SDK dev documentation"
[3]: https://observatory.brain-map.org/visualcoding/stimulus/natural_scenes?utm_source=chatgpt.com "Natural Scenes"
