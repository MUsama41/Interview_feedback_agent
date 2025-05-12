import os
import json
from pdf2markdown4llm import PDF2Markdown4LLM
from PyPDF2 import PdfReader
import re

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# from moviepy.editor import VideoFileClip

import json
from pyannote.audio import Pipeline
from pydub import AudioSegment
from faster_whisper import WhisperModel

# from configurations.config import diarization_model, llm_name,LABEL, KEY, TYPES, wav_filename, default_stt_modelname,default_stt_device, output_dir, temp_video_name,temp_pdf_filename


from configurations.config import FilePaths, ModelConfig, UIConfig,SchemaKeys




def read_file(filepath):
    """
    Generic file reader for different formats.

    Supports:
    - .json → returns parsed dict
    - .txt/.md → returns string
    - .pdf (optional) → future extension
    - binary files → returns bytes (optional)

    Returns:
    - File content or None on failure
    """
    try:
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        if ext == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)

        elif ext in [".txt", ".md"]:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None


def save_temp_file(
    uploaded_file=None,
    filename="temp_file",
    output_dir=".",
    as_json=False,
):

    """
    Generic file-saving utility.

    Parameters:
    - uploaded_file: file-like object (if saving uploaded file)
    - content: string to be saved (if saving generated text)
    - filename: name without extension
    - output_dir: where to save
    - as_json: if True, save content as JSON

    Returns:
    - Full path to saved file or empty string on failure
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        if as_json:
            path = os.path.join(output_dir, f"{filename}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"content": uploaded_file}, f, indent=4)

        elif uploaded_file:
            path = os.path.join(output_dir, filename)

            with open(path, "wb") as f:
                f.write(uploaded_file.read())
        else:
            raise ValueError("Nothing to save: Provide either `uploaded_file` or `content`.")
        return path
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return ""


class SpeechToText:
    def __init__(self,hf_token=None,  model_size=ModelConfig.default_stt_modelname, device=ModelConfig.default_stt_device):
        """
        Initialize the SpeechToText class by loading the Whisper model and PyAnnote pipeline.

        Parameters:
        - model_size: The size of the Whisper model to use ("tiny", "base", "medium", "large").
        - device: The device to use for computation ("cpu" or "cuda").
        - hf_token: Hugging Face authentication token for accessing PyAnnote models.
        """
        try:
            self.model = WhisperModel(model_size, device=device)
            print(f"Model {model_size} loaded on {device}.")
        except Exception as e:
            print(f"Error initializing Whisper model: {e}")
            self.model = None

        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                ModelConfig.diarization_model,
                use_auth_token=hf_token
            )
            print("PyAnnote diarization pipeline loaded.")
        except Exception as e:
            print(f"Error initializing PyAnnote pipeline: {e}")
            self.diarization_pipeline = None

    def transcribe_with_diarization(self, audio_file, transcribed_filename, output_dir):
        """
        Transcribe the given audio file to text with speaker diarization.

        Parameters:
        - audio_file: Path to the MP3 file to transcribe.
        - dest_path: Path to save the transcription JSON.

        Returns:
        - Transcribed text with speaker labels as a string.
        """
        if not self.model or not self.diarization_pipeline:
            print("Model or diarization pipeline not initialized.")
            return ""

        try:
            print(f"[INFO] Starting diarization for: {audio_file}")
            # Perform speaker diarization
            diarization = self.diarization_pipeline(audio_file)
            print("[INFO] Diarization completed.")

            # Load the original audio
            print("[INFO] Loading original audio...")
            audio = AudioSegment.from_file(audio_file)

            transcript = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_ms = int(turn.start * 1000)
                end_ms = int(turn.end * 1000)
                print(f"[INFO] Processing segment: Speaker={speaker}, Start={start_ms}, End={end_ms}")

                speaker_audio = audio[start_ms:end_ms]
                segment_filename = f"temp_{speaker}_{start_ms}_{end_ms}.wav"
                speaker_audio.export(segment_filename, format="wav")
                print(f"[INFO] Exported segment to: {segment_filename}")

                # Transcribe the segment
                segments, _ = self.model.transcribe(segment_filename)
                text = " ".join([segment.text for segment in segments])
                print(f"[INFO] Transcribed text: {text}")

                transcript.append({
                    "speaker": speaker,
                    "text": text
                })

            print(f"[INFO] Saving transcript to: {output_dir}")
            # with open(dest_path, "w", encoding="utf-8") as f:
            #     json.dump(transcript, f, ensure_ascii=False, indent=2)
            save_temp_file(transcript, transcribed_filename,output_dir)
            readable_transcript = ""
            for entry in transcript:
                readable_transcript += f"{entry['speaker']}: {entry['text']}\n"

            print("[INFO] Transcription with diarization completed successfully.")
            return readable_transcript

        except Exception as e:
            print(f"[ERROR] Error during transcription with diarization: {e}")
            return ""

def filter_multiple_speakers_text(data: list, speaker_ids: list) -> list:
    """
    Filters and extracts individual sentences spoken by specified speaker IDs.

    Args:
        data (list): List of dicts with 'speaker' and 'text' keys.
        speaker_ids (list): List of speaker IDs to filter.

    Returns:
        list: Flat list of individual sentences (strings).
    """
    sentences = []
    for item in data:
        speaker = item.get("speaker")
        if speaker in speaker_ids:
            text = item.get("text", "")
            # Split text into sentences using punctuation
            parts = re.split(r'(?<=[.!?])\s+', text.strip())
            sentences.extend([s for s in parts if s])
    return sentences


def is_valid_inputs(uploaded_files: dict):
    return all(file is not None for file in uploaded_files.values())


def pdf_to_markdown(file) -> str:
    try:
        # Validate file extension
        if not file.name.lower().endswith(".pdf"):
            raise ValueError("Invalid file format. Please upload a PDF.")
        print("in pdf_to_markdown : ",file, FilePaths.temp_pdf_filename,FilePaths.output_dir )
        temp_pdf_path = save_temp_file(file,FilePaths.temp_pdf_filename,FilePaths.output_dir)

        # Check if PDF has extractable text (not just images)
        reader = PdfReader(temp_pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""
        if not full_text.strip():
            raise ValueError("PDF appears to contain only images or no readable text.")

        # Convert to Markdown
        converter = PDF2Markdown4LLM()
        markdown_text = converter.convert(temp_pdf_path)

        if not markdown_text.strip():
            raise ValueError("PDF conversion resulted in empty content.")

        os.remove(temp_pdf_path)
        return markdown_text

    except Exception:
        if os.path.exists(FilePaths.temp_pdf_filename):
            os.remove(FilePaths.temp_pdf_filename)
        return ""




def save_individual_json(content: str, filename: str, output_dir=FilePaths.output_dir):
    return save_temp_file(content, filename, output_dir, as_json=True)

    


    
def convert_mp4_to_wav(uploaded_file, output_dir=FilePaths.output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)

        temp_mp4_path = save_temp_file(uploaded_file,FilePaths.temp_video_name, output_dir )

        # Define output WAV path
        wav_output_path = os.path.join(output_dir, FilePaths.wav_filename)

        # Convert MP4 to WAV using pydub
        audio = AudioSegment.from_file(temp_mp4_path, format="mp4")
        audio.export(wav_output_path, format="wav")

        # Clean up temp MP4 if needed
        os.remove(temp_mp4_path)

        return wav_output_path

    except Exception as e:
        print(f"Error converting to WAV: {e}")
        return ""


def process_uploaded_files(uploaded_files,file_inputs):
    """Process all uploaded PDF files (convert + save) and return paths."""
    processed_paths = {}
    
    for item in file_inputs:
        key = item[SchemaKeys.KEY]
        uploaded_file = uploaded_files.get(key)
        print("inside process_upload_files functions with key : ", key)


        if uploaded_file is not None:
            if uploaded_file.name.lower().endswith(".pdf"):
                markdown = pdf_to_markdown(uploaded_file)
                # key = f"{key}.json"
                processed_paths[key] = save_individual_json(markdown, key)

            elif uploaded_file.name.lower().endswith(".mp4"):
                mp3_path = convert_mp4_to_wav(uploaded_file)
                processed_paths[key] = mp3_path
                

    return processed_paths
    





# Ensure GROQ API key is set via environment variable
# export GROQ_API_KEY="your_key_here"

def analyze_candidate(resume: str, jd: str, interview_responses: str) -> str:
    # Load Groq API key from environment
    groq_api_key = ModelConfig.groq_api_key
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    # Initialize the Groq LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=ModelConfig.llm_name)

    # Define the evaluation prompt
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an expert HR Analyst in an Applicant Tracking System (ATS).
        Given the following information:

        1. Resume:
        {resume}

        2. Job Description:
        {jd}

        3. Interview Responses (Conversation format):
        {interview_responses}

        Generate a detailed candidate evaluation report with the following structure:

        - Strong Points
        - Weak Points
        - Suggestions for Improvement
        - Final Decision Summary about the Candidate (Suitable/Not Suitable + Short Justification)
        """
    )

    # Format the prompt
    prompt = prompt_template.format(
        resume=resume,
        jd=jd,
        interview_responses=interview_responses
    )

    # Get response from the LLM
    response = llm.invoke(prompt)
    return response.content

# helper functions for chunk