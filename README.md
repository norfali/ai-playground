# SpeechLab

**A closer listen.**

Turn audio into text, then explore what Whisper heard. SpeechLab is a small, local speech recognition playground for Apple Silicon: listen to a recording, refine a reference transcript and see where the model's output differs.

[GitHub](https://github.com/norfali/speech-lab) · [Fenka](https://www.fenka.co.uk)

## Why I built this

I've always enjoyed exploring technology and figuring out how things work. SpeechLab started simply out of curiosity: I wanted to experiment with speech recognition, see what modern models could do on my own Mac, and better understand how to evaluate their results.

There's no grand ambition behind it. It's a small, open-source playground for learning, experimenting and having a bit of fun along the way.

## Getting started

You'll need:

- An Apple Silicon Mac running macOS.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- [FFmpeg](https://ffmpeg.org/) for reading audio files.
- An internet connection for installation and the first download of each model.

Install FFmpeg with Homebrew, if you use it:

```bash
brew install ffmpeg
```

Clone the repository, install the dependencies and launch SpeechLab:

```bash
git clone https://github.com/norfali/speech-lab.git
cd speech-lab
uv sync
uv run speech-lab
```

Open the local address printed in your terminal, usually [http://localhost:7860](http://localhost:7860). Press `Ctrl+C` to stop the app.

The first transcription downloads the selected model from Hugging Face and may take longer. Later runs reuse the cached model.

## How it works

1. **Upload or record** an audio clip in your browser.
2. **Choose a model and transcribe.** Small is the default; Tiny is a faster option. Both support multiple languages and run locally on your Mac.
3. **Listen and edit the reference** to match what you hear. The original Whisper output stays unchanged beside it.
4. **Explore the results.** Word differences and scores update as you edit; performance measurements describe the last transcription run.

## Exploring evals

Evals compare Whisper's output with your editable reference and show how long the local run took.

- **Word Error Rate (WER)** counts different, missing and extra words relative to the number of words in the reference. Lower means closer agreement; 0% means a match after normalisation. Extra words can push it above 100%.
- **Character Error Rate (CER)** makes the same comparison at character level, helping you spot smaller spelling differences.
- **Word-level differences** show only the passages that differ, labelled “Different word”, “Missing word” or “Extra word”, with changed characters highlighted.

Comparisons ignore letter case and repeated whitespace, but keep punctuation and accents. The transcript fields retain their original text. An empty reference has no percentage scores.

The reference starts as a copy of Whisper's output, so the initial zero errors show agreement, **not independently verified transcription accuracy**. Editing the reference is a starting point; meaningful accuracy checks need a carefully verified transcript.

Performance results include:

- **Processing time:** total time for the transcription run, including audio decoding, model loading and any initial model download.
- **Audio duration:** the length of the decoded recording in seconds.
- **Processing speed:** how many seconds of audio were processed per second; 5× real time means a 10-second clip took 2 seconds to process.
- **Real-time factor (RTF):** processing time divided by audio duration. An RTF of 0.2 means processing took one fifth of the recording's length; 2 means it took twice as long.

These describe the whole local run, not a pure inference benchmark. First runs and runs with a cached model can differ substantially.

## Privacy

Speech recognition runs locally on your Mac. SpeechLab does not send audio to a cloud transcription service. The CLI binds to the local machine, disables public sharing and turns off Gradio analytics and Hugging Face telemetry.

Model weights are downloaded from Hugging Face when first needed and cached locally. Uploaded and recorded audio is stored in local temporary files managed by Gradio; those files may remain after use. Take care with sensitive recordings, especially on shared devices.

## Development

After `uv sync`, run the checks with:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The `speech_lab` Python package uses Gradio for the interface, [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for local transcription and [JiWER](https://github.com/jitsi/jiwer) for evaluation. Inference and scoring are separate from the UI. Tests mock inference and do not download model weights.

## Disclaimer

SpeechLab is an educational experiment, provided “as is” without warranty. Transcripts may be inaccurate, incomplete or misleading; independently verify them before using them for consequential decisions. You are responsible for the rights and permissions needed to record and process audio, and for complying with applicable privacy laws. To the extent permitted by law, the author accepts no liability for loss or damage arising from use. See the [MIT License](LICENSE) for the full warranty disclaimer and limitation of liability.

## License

Released under the [MIT License](LICENSE).
