import os
import json
from pdf2markdown4llm import PDF2Markdown4LLM
from PyPDF2 import PdfReader
from faster_whisper import WhisperModel
from pydub import AudioSegment

# from moviepy.editor import VideoFileClip

class SpeechToText:
    def __init__(self, model_size="tiny", device="cpu"):
        """
        Initialize the SpeechToText class by loading the Whisper model.

        Parameters:
        - model_size: The size of the Whisper model to use ("tiny", "base", "medium", "large").
        - device: The device to use for computation ("cpu" or "cuda").
        """
        try:
            self.model = WhisperModel(model_size, device=device)
            print(f"################################################Model {model_size} loaded on {device}.")
        except Exception:
            print("Error initializing Whisper model.")
            self.model = None

    def transcribe(self, audio_file, dest_path):
        """
        Transcribe the given audio file to text.

        Parameters:
        - audio_file: Path to the MP3 file to transcribe.

        Returns:
        - Transcribed text as a string.
        """
        try:
            segments, _ = self.model.transcribe(audio_file)
            transcript = {"text": []}

            for segment in segments:
                transcript["text"].append(segment.text)

            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)
            return " ".join([segment.text for segment in segments])
        except Exception:
            print(f"Error during transcription of {audio_file}.")
            return ""



def is_valid_inputs(resume, jd, video):
    return resume is not None and jd is not None and video is not None

def pdf_to_markdown(file) -> str:
    try:
        # Validate file extension
        if not file.name.lower().endswith(".pdf"):
            raise ValueError("Invalid file format. Please upload a PDF.")

        # Save temporarily
        temp_path = "temp_uploaded_file.pdf"
        with open(temp_path, "wb") as f:
            f.write(file.read())

        # Check if PDF has extractable text (not just images)
        reader = PdfReader(temp_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""
        if not full_text.strip():
            raise ValueError("PDF appears to contain only images or no readable text.")

        # Convert to Markdown
        converter = PDF2Markdown4LLM()
        markdown_text = converter.convert(temp_path)

        if not markdown_text.strip():
            raise ValueError("PDF conversion resulted in empty content.")

        os.remove(temp_path)
        return markdown_text

    except Exception:
        if os.path.exists("temp_uploaded_file.pdf"):
            os.remove("temp_uploaded_file.pdf")
        return ""


def save_individual_json(content: str, filename: str, output_dir="markdowns/"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        data = {"content": content}
        filepath = os.path.join(output_dir, f"{filename}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return filepath
    except Exception:
        print(f"Error saving {filename}.json.")
        return ""


def convert_mp4_to_mp3(mp4_path, mp3_path):
    try:
        # Use ffmpeg to extract audio from the MP4 file
        audio = AudioSegment.from_file(mp4_path, format="mp4")
        audio.export(mp3_path, format="mp3")
    except Exception:
        print(f"Error converting {mp4_path} to MP3.")
