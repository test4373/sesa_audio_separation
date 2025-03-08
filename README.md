# SESA Audio Separation Toolkit

**SESA Audio Separation Toolkit** is a Python library designed for professional audio source separation tasks. It allows you to separate music and audio files into vocals, instrumental, and other components. The library provides over 50 pre-trained models and advanced ensemble methods.

## Features

- **Multiple Model Support**: 50+ pre-trained models (Vocals, Instrumental, 4-Stem, Denoise, Dereverb, etc.).
- **Gradio-Based GUI**: User-friendly interface for easy usage.
- **YouTube and Google Drive Support**: Directly download audio files from URLs.
- **Ensemble Methods**: Combine outputs from multiple models (avg_wave, median_wave, max_fft, etc.).
- **Multiple Output Formats**: Support for WAV, FLAC (PCM_16, PCM_24).
- **GPU Support**: Accelerated processing with CUDA-compatible GPUs.
- **Phase Processing**: Advanced phase correction algorithms.

## Installation

To install the library, use the following command:


pip install sesa-audio-separation


### Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended)
- PyTorch (will be installed automatically)

## Usage

### Command Line Interface (CLI)

To run the library from the command line:

```bash
sesa-audio-separation --method gradio
```

This command launches a Gradio-based web interface. You can access it by navigating to `http://localhost:7860` in your browser.

### Python API

To use the library in your Python code:

```python
from sesa_audio_separation import separate_audio

# Separate audio file
result = separate_audio(
    input_path="song.mp3",
    model_name="VOCALS-MelBand-Roformer BigBeta5e",
    output_format="flac PCM_24",
    device="cuda"
)

# Save separated files
result.save("output_folder")
```

### Gradio Interface

The Gradio interface offers the following features:
- Upload audio files or provide file paths.
- Select models and configure settings.
- Listen to and download separated audio files.
- Perform ensemble operations with multiple models.

## Models

The library provides models in the following categories:
- **Vocal Models**: VOCALS-MelBand-Roformer, VOCALS-BS-Roformer, etc.
- **Instrumental Models**: INST-Mel-Roformer, INST-VOC-Mel-Roformer, etc.
- **4-Stem Models**: 4STEMS-SCNet, 4STEMS-BS-Roformer, etc.
- **Denoise Models**: DENOISE-MelBand-Roformer, denoisedebleed, etc.
- **Dereverb Models**: DE-REVERB-MDX23C, DE-REVERB-MelBand-Roformer, etc.

## Example Use Cases

1. **Vocal Separation**:
   ```python
   from sesa_audio_separation import separate_audio

   result = separate_audio(
       input_path="song.mp3",
       model_name="VOCALS-MelBand-Roformer BigBeta5e",
       output_format="wav FLOAT",
       device="cuda"
   )
   result.save("vocal_output")
   ```

2. **Ensemble Processing**:
   ```python
   from sesa_audio_separation import ensemble_audio

   result = ensemble_audio(
       input_path="song.mp3",
       models=["VOCALS-MelBand-Roformer BigBeta5e", "VOCALS-BS-Roformer_1297"],
       ensemble_type="avg_wave",
       output_format="flac PCM_24",
       device="cuda"
   )
   result.save("ensemble_output")
   ```

3. **Separate Audio from YouTube**:
   ```python
   from sesa_audio_separation import separate_audio

   result = separate_audio(
       input_path="https://www.youtube.com/watch?v=example",
       model_name="VOCALS-MelBand-Roformer BigBeta5e",
       output_format="wav FLOAT",
       device="cuda"
   )
   result.save("youtube_output")
   ```

## Contributing

If you'd like to contribute, please submit a **Pull Request**. For feature requests and bug reports, use the **Issues** section.

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.
```
