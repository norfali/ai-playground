# SpeechLab

**A closer listen.**

Turn audio into text, then explore what Whisper heard. SpeechLab is a small, local speech recognition playground for Apple Silicon. Listen to a recording, refine a reference transcript and see where the model's output differs.

[AI Playground](https://github.com/norfali/ai-playground) · [Fenka](https://www.fenka.co.uk)

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
git clone https://github.com/norfali/ai-playground.git
cd ai-playground
make sync
make speech
```

Open the local address printed in your terminal, usually [http://localhost:7860](http://localhost:7860). Press `Ctrl+C` to stop the app.

The first transcription downloads the selected model from Hugging Face and may take longer. Later runs reuse the cached model.

## Tech stack

- **Python 3.12** — application logic and evaluation.
- **[Gradio](https://www.gradio.app/)** — browser-based interface for recording, transcription and comparing results.
- **[MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)** — local speech recognition accelerated by Apple Silicon.
- **[JiWER](https://github.com/jitsi/jiwer)** — word and character error rate calculations.
- **[uv](https://docs.astral.sh/uv/)** — dependency and workspace management.
- **Ruff and pytest** — linting, formatting and automated tests.

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

### Performance

SpeechLab also reports:

- **Processing time:** total time for the transcription run, including audio decoding, model loading and any initial model download.
- **Audio duration:** the length of the decoded recording in seconds.
- **Processing speed:** how many seconds of audio were processed per second; 5× real time means a 10-second clip took 2 seconds to process.
- **Real-time factor (RTF):** processing time divided by audio duration. An RTF of 0.2 means processing took one fifth of the recording's length; 2 means it took twice as long.

These measurements describe the whole local run, not a pure inference benchmark. First runs and runs with a cached model can differ substantially.

## Development

From the repository root:

```bash
make sync           # Install workspace dependencies
make speech         # Launch SpeechLab
make speech-test    # Run tests
```

Inference and scoring are kept separate from the UI. Tests mock inference and do not download model weights.

## Privacy

Speech recognition runs locally on your Mac. SpeechLab does not send audio to a cloud transcription service. The CLI binds to the local machine, disables public sharing and turns off Gradio analytics and Hugging Face telemetry.

Model weights are downloaded from Hugging Face when first needed and cached locally. Uploaded and recorded audio is stored in local temporary files managed by Gradio; those files may remain after use. Take care with sensitive recordings, especially on shared devices.

## Disclaimer

SpeechLab is an educational exploration, provided “as is” without warranty. Transcripts may be inaccurate, incomplete or misleading; independently verify them before using them for consequential decisions.

You are responsible for the rights and permissions needed to record and process audio, and for complying with applicable privacy laws. To the extent permitted by law, the author accepts no liability for loss or damage arising from use.

## License

Released under the [MIT License](../LICENSE).
