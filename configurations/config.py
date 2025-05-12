import os
from dotenv import load_dotenv

load_dotenv()


class SchemaKeys:
    LABEL = "label"
    KEY = "key"
    TYPES = "types"
    grouping_threshold = 0.2


class UIConfig:
    submit_button_label = "Submit files"
    required_speakers_text = ["SPEAKER_01"]
    file_inputs = [
        {"label": "### 1. Upload Resume", "key": "resume", "types": ["pdf"]},
        {"label": "### 2. Upload Job Description", "key": "jd", "types": ["pdf"]},
        {"label": "### 3. Upload Introduction Video", "key": "video", "types": ["mp4"]},
    ]


class FilePaths:
    wav_filename = "converted_audio.wav"
    output_dir = "markdowns/"
    temp_video_name = "temp_video.mp4"
    temp_pdf_filename = "temp_uploaded_file.pdf"
    interview_wavfile = "data/output.wav"
    transcribed_filename = "transcription.json"
    resume_filename = "resume.json"
    jd_filename = "jd.json"
    report_file = "report"
    interview_wavfile = "output.wav"
    


class ModelConfig:
    diarization_model = "pyannote/speaker-diarization-3.1"
    llm_name = "llama-3.3-70b-versatile"

    stt_modelname = os.getenv("STT_MODELNAME", "tiny")
    stt_device = os.getenv("STT_DEVICE", "cpu")

    encoder_model = "all-MiniLM-L6-v2"


    hf_token = os.environ["HF_TOKEN"]
    groq_api_key = os.environ["GROQ_API_KEY"]
    
    default_stt_modelname = "tiny"
    default_stt_device = "cpu"
