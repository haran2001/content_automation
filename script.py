# transcribe_and_extract_marketing.py

import os
import openai
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import json
import time
from datetime import timedelta
import math
import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def extract_audio(video_path, audio_path="extracted_audio.wav"):
    """
    Extracts audio from the given video file and saves it as a WAV file.
    """
    try:
        print(f"Extracting audio from {video_path}...")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec="pcm_s16le")
        print(f"Audio extracted and saved to {audio_path}.")
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None


def split_audio(audio_path, chunk_length_ms=240000):
    """
    Splits the audio file into chunks of specified length in milliseconds.
    Default is 4 minutes (240,000 ms) per chunk to stay within size limits.
    Returns a list of AudioSegment objects.
    """
    try:
        print(
            f"Splitting audio from {audio_path} into chunks of {chunk_length_ms / 60000} minutes each..."
        )
        audio = AudioSegment.from_wav(audio_path)
        chunks = []
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i : i + chunk_length_ms]
            chunks.append(chunk)
        print(f"Audio split into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"Error splitting audio: {e}")
        return []


def save_audio_chunks(chunks, base_path="audio_chunk"):
    """
    Saves each AudioSegment chunk to a separate WAV file.
    Returns a list of file paths.
    """
    try:
        chunk_paths = []
        for idx, chunk in enumerate(chunks):
            chunk_path = f"{base_path}_{idx + 1}.wav"
            chunk.export(chunk_path, format="wav")
            chunk_paths.append(chunk_path)
            print(f"Saved {chunk_path}.")
        return chunk_paths
    except Exception as e:
        print(f"Error saving audio chunks: {e}")
        return []


def transcribe_audio_chunks(chunk_paths, openai_api_key):
    """
    Transcribes each audio chunk using OpenAI's Whisper API.
    Returns a list of all transcript segments with accurate timestamps.
    """
    try:
        print("Starting transcription of audio chunks...")
        openai.api_key = openai_api_key
        all_segments = []
        for idx, chunk_path in enumerate(chunk_paths):
            print(f"Transcribing chunk {idx + 1}/{len(chunk_paths)}: {chunk_path}")
            with open(chunk_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe(
                    "whisper-1", audio_file, response_format="verbose_json"
                )
            segments = transcript.get("segments", [])
            # Adjust timestamps based on chunk index and chunk length
            chunk_start_time = idx * 240  # in seconds (4 minutes per chunk)
            for segment in segments:
                adjusted_segment = {
                    "start": segment["start"] + chunk_start_time,
                    "end": segment["end"] + chunk_start_time,
                    "text": segment["text"].strip(),
                }
                all_segments.append(adjusted_segment)
            # Optional: Delete chunk file after transcription to save space
            os.remove(chunk_path)
            print(f"Chunk {idx + 1} transcribed and removed.")
            time.sleep(1)  # To respect rate limits
        print("All audio chunks transcribed.")
        return all_segments
    except Exception as e:
        print(f"Error transcribing audio chunks: {e}")
        return []


def format_transcript(segments, output_path="transcript_with_timestamps.txt"):
    """
    Formats the transcript segments with timestamps and saves to a text file.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for segment in segments:
                start = timedelta(seconds=segment["start"])
                end = timedelta(seconds=segment["end"])
                text = segment["text"]
                f.write(f"[{start} - {end}] {text}\n")
        print(f"Formatted transcript saved to {output_path}.")
    except Exception as e:
        print(f"Error formatting transcript: {e}")


def count_tokens(text, model="gpt-4"):
    """
    Counts the number of tokens in the given text using tiktoken.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)


def split_segments_into_batches(segments, max_tokens=3500, model="gpt-4"):
    """
    Splits the transcript segments into batches that do not exceed max_tokens.
    Returns a list of batches, each containing a list of segments.
    """
    batches = []
    current_batch = []
    current_tokens = 0

    for segment in segments:
        # Estimate tokens for this segment
        segment_text = segment["text"]
        segment_tokens = (
            count_tokens(segment_text, model=model) + 10
        )  # Adding buffer for metadata

        if current_tokens + segment_tokens > max_tokens:
            if current_batch:
                batches.append(current_batch)
            current_batch = [segment]
            current_tokens = segment_tokens
        else:
            current_batch.append(segment)
            current_tokens += segment_tokens

    if current_batch:
        batches.append(current_batch)

    print(f"Transcript split into {len(batches)} batches.")
    return batches


def extract_marketing_content_from_batch(batch, openai_api_key, model="gpt-4"):
    """
    Extracts marketing content from a batch of transcript segments using OpenAI's GPT-4.
    Returns a list of dictionaries with timestamp and content.
    """
    try:
        print(f"Processing a batch of {len(batch)} segments for marketing content...")

        # Combine the batch segments into a single string with timestamps
        batch_text = "\n".join(
            [
                f"[{timedelta(seconds=seg['start'])} - {timedelta(seconds=seg['end'])}] {seg['text']}"
                for seg in batch
            ]
        )

        prompt = (
            "You are a marketing expert. Given the following transcript with timestamps, "
            "extract the most interesting and engaging content suitable for creating marketing materials. "
            "For each extracted piece of content, provide the corresponding timestamp.\n\n"
            "Transcript:\n"
            f"{batch_text}\n\n"
            "Extracted Marketing Content (in JSON format with 'timestamp' and 'content' fields):"
        )

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are ChatGPT, a large language model trained by OpenAI.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.7,
        )

        marketing_content = response.choices[0].message["content"].strip()

        # Attempt to parse the response as JSON
        try:
            marketing_data = json.loads(marketing_content)
            # Validate structure
            if isinstance(marketing_data, list) and all(
                "timestamp" in item and "content" in item for item in marketing_data
            ):
                print("Marketing content extracted successfully from the batch.")
                return marketing_data
            else:
                raise ValueError("JSON structure is invalid.")
        except json.JSONDecodeError:
            # If GPT-4 did not return valid JSON, attempt manual parsing
            print(
                "Warning: GPT-4 response is not in valid JSON format. Attempting to parse manually."
            )
            marketing_data = []
            lines = marketing_content.split("\n")
            for line in lines:
                if line.strip():
                    # Example line format: {"timestamp": "00:01:23 - 00:01:30", "content": "Engaging content here."}
                    try:
                        data = json.loads(line)
                        if "timestamp" in data and "content" in data:
                            marketing_data.append(data)
                    except json.JSONDecodeError:
                        continue
            return marketing_data

    except Exception as e:
        print(f"Error extracting marketing content from batch: {e}")
        return []


def extract_marketing_content(segments, openai_api_key, model="gpt-4"):
    """
    Processes the transcript segments in batches and extracts marketing content.
    Returns a list of dictionaries with timestamp and content.
    """
    try:
        # Define maximum tokens per batch
        # OpenAI's GPT-4 has a context window of 8192 tokens; reserve some for the prompt and response
        max_tokens = 3500  # Adjust based on needs

        # Split segments into batches
        batches = split_segments_into_batches(
            segments, max_tokens=max_tokens, model=model
        )

        all_marketing_content = []

        for batch in batches:
            marketing_content = extract_marketing_content_from_batch(
                batch, openai_api_key, model=model
            )
            if marketing_content:
                all_marketing_content.extend(marketing_content)
            time.sleep(1)  # To respect rate limits

        print("All marketing content extracted.")
        return all_marketing_content

    except Exception as e:
        print(f"Error extracting marketing content: {e}")
        return []


def save_marketing_content(
    marketing_data, output_path="marketing_content_with_timestamps.json"
):
    """
    Saves the extracted marketing content with timestamps to a JSON file.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(marketing_data, f, indent=4)
        print(f"Marketing content with timestamps saved to {output_path}.")
    except Exception as e:
        print(f"Error saving marketing content: {e}")


def main():
    # Configuration
    video_path = "video.mp4"  # Replace with your video file path
    audio_path = "extracted_audio.wav"
    transcript_output = "transcript_with_timestamps.txt"
    marketing_output = "marketing_content_with_timestamps.json"
    openai_api_key = os.getenv(
        "OPENAI_API_KEY"
    )  # Ensure you set this environment variable

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    # Step 1: Extract audio from video
    audio_file = extract_audio(video_path, audio_path)
    if not audio_file:
        return

    # Step 2: Split audio into chunks
    # chunks = split_audio(audio_file, chunk_length_ms=240000)  # 4 minutes per chunk
    chunks = split_audio(audio_file, chunk_length_ms=60000)  # 4 minutes per chunk
    if not chunks:
        return

    # Step 3: Save audio chunks to files
    chunk_paths = save_audio_chunks(chunks, base_path="audio_chunk")
    if not chunk_paths:
        return

    # Optional: Delete the full extracted audio to save space
    try:
        os.remove(audio_file)
        print(f"Deleted extracted audio file {audio_file}.")
    except Exception as e:
        print(f"Error deleting extracted audio file: {e}")

    # Step 4: Transcribe each audio chunk
    segments = transcribe_audio_chunks(chunk_paths, openai_api_key)
    if not segments:
        return

    # Step 5: Format and save the transcript with timestamps
    format_transcript(segments, transcript_output)

    # Step 6: Extract marketing content with timestamps
    marketing_data = extract_marketing_content(segments, openai_api_key, model="gpt-4")
    if not marketing_data:
        print("No marketing content extracted.")
        return

    # Step 7: Save marketing content with timestamps
    save_marketing_content(marketing_data, marketing_output)


if __name__ == "__main__":
    main()
