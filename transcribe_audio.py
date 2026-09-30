"""
Transcribe all audio files using OpenAI Whisper.
Saves results as JSON with full segment metadata for feature extraction.
Run with: conda run -n shl env KMP_DUPLICATE_LIB_OK=TRUE python transcribe_audio.py
"""
import whisper
import json
import os
import time
import torch

# Configuration
WHISPER_MODEL = "small"  # Better accuracy than base
TRAIN_DIR = "Dataset_Final/train"
TEST_DIR = "Dataset_Final/test"
OUTPUT_DIR = "transcripts"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def transcribe_all(audio_dir, output_file):
    """Transcribe all audio files in a directory and save results."""
    # Check if output already exists
    if os.path.exists(output_file):
        print(f"Loading existing transcripts from {output_file}")
        with open(output_file, "r") as f:
            return json.load(f)

    # Determine device
    if torch.backends.mps.is_available():
        device = "cpu"  # Whisper has issues with MPS, use CPU for stability
        print(f"Using CPU (MPS available but Whisper is more stable on CPU)")
    else:
        device = "cpu"
        print(f"Using CPU")

    print(f"Loading Whisper model: {WHISPER_MODEL}")
    model = whisper.load_model(WHISPER_MODEL, device=device)
    print(f"Model loaded successfully")

    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    total = len(audio_files)
    transcripts = {}

    print(f"\nTranscribing {total} files from {audio_dir}...")
    start_time = time.time()

    for i, filename in enumerate(audio_files):
        filepath = os.path.join(audio_dir, filename)
        try:
            result = model.transcribe(filepath, language="en", fp16=False)
            transcripts[filename] = {
                "text": result["text"],
                "language": result.get("language", "en"),
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"],
                        "avg_logprob": seg.get("avg_logprob", 0),
                        "no_speech_prob": seg.get("no_speech_prob", 0),
                        "compression_ratio": seg.get("compression_ratio", 0),
                        "temperature": seg.get("temperature", 0),
                    }
                    for seg in result["segments"]
                ]
            }
        except Exception as e:
            print(f"  ERROR on {filename}: {e}")
            transcripts[filename] = {"text": "", "language": "en", "segments": []}

        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (total - i - 1)
        print(f"  [{i+1}/{total}] {filename} - {len(transcripts[filename]['text'])} chars "
              f"(~{remaining/60:.0f} min remaining)")

        # Save intermediate results every 50 files
        if (i + 1) % 50 == 0:
            with open(output_file, "w") as f:
                json.dump(transcripts, f, indent=2)
            print(f"  >> Saved intermediate results ({i+1}/{total})")

    # Save final results
    with open(output_file, "w") as f:
        json.dump(transcripts, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nDone! Transcribed {total} files in {total_time/60:.1f} minutes")
    print(f"Saved to {output_file}")
    return transcripts


if __name__ == "__main__":
    print("=" * 60)
    print("SHL Grammar Scoring - Audio Transcription")
    print("=" * 60)
    
    # Transcribe training data
    print("\n--- Training Data ---")
    train_transcripts = transcribe_all(TRAIN_DIR, os.path.join(OUTPUT_DIR, "train_transcripts.json"))
    
    # Transcribe test data
    print("\n--- Test Data ---")
    test_transcripts = transcribe_all(TEST_DIR, os.path.join(OUTPUT_DIR, "test_transcripts.json"))
    
    print(f"\n{'=' * 60}")
    print(f"Transcription complete!")
    print(f"Train: {len(train_transcripts)} files")
    print(f"Test:  {len(test_transcripts)} files")
    print(f"{'=' * 60}")
