# -------------------- System Info (Optional Debugging) --------------------
import sys
print(sys.executable)  # Shows current Python interpreter (useful for debugging)

# -------------------- Standard Library --------------------
import os
import warnings
import json

# -------------------- Environment Setup --------------------
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- Third-party Libraries --------------------
import streamlit as st
from pyannote.audio import Pipeline
from pydub import AudioSegment
from faster_whisper import WhisperModel

# -------------------- App-specific Modules --------------------
from configurations.config import FilePaths, ModelConfig, UIConfig,SchemaKeys
from utils.routers import summarize_resume_jd_interview,process_interview_json

from utils.helpers import (
    is_valid_inputs,
    SpeechToText,
    analyze_candidate,
    process_uploaded_files,
    read_file,
    filter_multiple_speakers_text,
    save_temp_file
)

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(page_title="Job Match App", layout="centered")

# -------------------- Resource Caching --------------------
@st.cache_resource
def load_stt_model():
    return SpeechToText(
        ModelConfig.hf_token,
        model_size=ModelConfig.stt_modelname,
        device=ModelConfig.stt_device
    )

stt = load_stt_model()

# -------------------- Custom Styles --------------------
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error("⚠️ Stylesheet not found. Ensure the 'assets/styles.css' file is present.")
except Exception as e:
    st.error(f"🚨 Unexpected error loading stylesheet: {str(e)}")

# -------------------- UI Layout --------------------
st.title("Job Match Application")
st.subheader("Please upload the following files for processing:")

uploaded_files = {}
for item in UIConfig.file_inputs:
    uploaded_files[item[SchemaKeys.KEY]] = st.file_uploader(item[SchemaKeys.LABEL], type=item[SchemaKeys.TYPES])

valid_inputs = is_valid_inputs(uploaded_files)


if valid_inputs:
    if st.button(UIConfig.submit_button_label):
        try:
            st.info("Processing inputs...")

            processed_paths = process_uploaded_files(uploaded_files, UIConfig.file_inputs)


            # text = stt.transcribe_with_diarization(FilePaths.interview_wavfile, FilePaths.transcribed_filename, FilePaths.output_dir)


            transcribed_text = os.path.join(FilePaths.output_dir,FilePaths.transcribed_filename)
            interview_text = read_file(transcribed_text)
            filtered_interview_text = filter_multiple_speakers_text(interview_text, UIConfig.required_speakers_text)


            resume_path = os.path.join(FilePaths.output_dir,FilePaths.resume_filename)
            jd_path = os.path.join(FilePaths.output_dir,FilePaths.jd_filename)

            grouped_text = process_interview_json(filtered_interview_text)
            resume_jd_dict = summarize_resume_jd_interview(resume_path, jd_path,grouped_text)
   
            report = analyze_candidate(resume_jd_dict["parsed_resume"], resume_jd_dict["parsed_jd"], resume_jd_dict["interview_summary"])

            # Define the file path where the report will be saved
            report_dict = {
                "evaluation_report": report
            }
            st.session_state.report_dict = report_dict  # Save to session state
            if "report_dict" in st.session_state:
                st.markdown("### 📝 Evaluation Report")
                st.json(st.session_state.report_dict)  # Display the saved report dict as JSON

            
            save_temp_file(report_dict, FilePaths.report_file, FilePaths.output_dir,as_json = True)

            st.success("✅ All files processed and saved successfully.")
            st.markdown(f"📄 Resume saved to: ")
            st.markdown(f"📄 JD saved to")
            st.markdown(f"🎥 Video saved to: ")
            st.markdown(f"🔊 Audio extracted to: ")

        except RuntimeError as e:
            st.error(f"⚠️ {str(e)}")
        except IOError as e:
            st.error(f"📁 File saving error: {str(e)}")
        except Exception as e:
            st.error(f"🚨 Unexpected error: {str(e)}")

else:
    st.warning("Please upload all three files to enable the submit button.")
