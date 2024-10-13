# Smart Video Transcription and Marketing Content Extraction Tool

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![OpenAI API](https://img.shields.io/badge/OpenAI-API-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Outputs](#outputs)
- [Error Handling](#error-handling)
- [Sample Output](#sample-output)
- [Additional Considerations](#additional-considerations)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

The **Smart Video Transcription and Marketing Content Extraction Tool** is a Python-based utility designed to transform MP4 video content into valuable marketing materials. By leveraging OpenAI's Whisper and GPT-4 APIs, this tool automates the process of:

1. **Extracting Audio** from an MP4 video.
2. **Transcribing** the audio into text with accurate timestamps.
3. **Extracting Engaging Marketing Content** from the transcript, preserving the corresponding timestamps.
4. **Handling Large Transcripts** by splitting audio and transcript into manageable chunks to comply with API size and token limits.

This automation streamlines the creation of marketing content from video materials, ensuring efficiency and scalability.

---

## Features

- **Audio Extraction:** Converts MP4 video files to WAV audio format with reduced size parameters (mono channel, lower sample rate).
- **Audio Splitting:** Divides audio into smaller chunks (e.g., 3 minutes, further into 1.5 minutes) to comply with OpenAI's API size limits.
- **Transcription with Timestamps:** Utilizes OpenAI's Whisper API to transcribe audio segments, capturing start and end times.
- **Marketing Content Extraction:** Employs OpenAI's GPT-4 to analyze transcripts and extract the most engaging content for marketing purposes, along with corresponding timestamps.
- **Batch Processing:** Splits large transcripts into batches based on token counts to adhere to GPT-4's token limitations.
- **Output Generation:** Produces formatted transcript files and structured JSON files containing marketing content with timestamps.
- **Automated Cleanup:** Removes temporary audio chunks after successful transcription to conserve disk space.
- **Robust Error Handling:** Implements checks and fallback mechanisms to handle API errors and ensure smooth execution.

---

## Prerequisites

Before using the tool, ensure you have the following installed and configured:

1. **Python:**

   - Version **3.7** or higher.
   - [Download Python](https://www.python.org/downloads/)

2. **FFmpeg:**

   - Required for audio processing with `moviepy` and `pydub`.
   - **Installation:**
     - **Windows:**
       - Download from [FFmpeg Official Website](https://ffmpeg.org/download.html).
       - Extract the downloaded archive.
       - Add the `bin` folder of FFmpeg to your system's PATH environment variable.
     - **macOS (using Homebrew):**
       ```bash
       brew install ffmpeg
       ```
     - **Linux (Ubuntu/Debian):**
       ```bash
       sudo apt update
       sudo apt install ffmpeg
       ```

3. **OpenAI API Key:**

   - Sign up at [OpenAI](https://platform.openai.com/) to obtain your API key.
   - **Security:** Keep your API key confidential. Do not expose it in public repositories or share it openly.

4. **Python Libraries:**
   - Install the required Python packages using `pip`.

---

## Installation

1. **Clone the Repository or Create a Project Directory:**

   ```bash
   mkdir smart_video_transcription
   cd smart_video_transcription
   ```
