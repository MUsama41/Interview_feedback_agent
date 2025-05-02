import streamlit as st
import os
from utils.helpers import is_valid_inputs, pdf_to_markdown, save_individual_json, convert_mp4_to_mp3, SpeechToText

st.set_page_config(page_title="Job Match App", layout="centered")


@st.cache_resource
def load_stt_model():
    return SpeechToText(model_size="base", device="cpu")


stt = load_stt_model()


try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error("⚠️ Stylesheet not found. Ensure the 'assets/styles.css' file is present.")
except Exception as e:
    st.error(f"🚨 Unexpected error loading stylesheet: {str(e)}")

st.title("Job Match Application")
st.subheader("Please upload the following files for processing:")

resume = st.file_uploader("### 1. Upload Resume (PDF)", type=["pdf"])
jd = st.file_uploader("### 2. Upload Job Description (PDF)", type=["pdf", "docx"])
video = st.file_uploader("### 3. Upload Introduction Video (MP4)", type=["mp4"])

valid_inputs = is_valid_inputs(resume, jd, video)

if valid_inputs:
    if st.button("Submit"):
        try:
            st.info("Processing inputs...")

            resume_md = pdf_to_markdown(resume)
            jd_md = pdf_to_markdown(jd)

            resume_path = save_individual_json(resume_md, "resume")
            jd_path = save_individual_json(jd_md, "job_description")

            # Save uploaded video
            video_folder = "uploads"
            os.makedirs(video_folder, exist_ok=True)
            video_path = os.path.join(video_folder, "intro_video.mp4")
            with open(video_path, "wb") as f:
                f.write(video.read())

            # Convert video to MP3
            audio_path = os.path.join(video_folder, "intro_audio.mp3")
            convert_mp4_to_mp3(video_path, audio_path)

            text = stt.transcribe("uploads/intro_audio.mp3", "markdowns/transcription.json")
            st.success("✅ All files processed and saved successfully.")
            st.markdown(f"📄 Resume saved to: `{resume_path}`")
            st.markdown(f"📄 JD saved to: `{jd_path}`")
            st.markdown(f"🎥 Video saved to: `{video_path}`")
            st.markdown(f"🔊 Audio extracted to: `{audio_path}`")

        except RuntimeError as e:
            st.error(f"⚠️ {str(e)}")
        except IOError as e:
            st.error(f"📁 File saving error: {str(e)}")
        except Exception as e:
            st.error(f"🚨 Unexpected error: {str(e)}")

else:
    st.warning("Please upload all three files to enable the submit button.")
