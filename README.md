### *Automated Interview Intelligence System | AI/ML Solution for Hiring Optimization*

I developed a comprehensive AI/ML-powered solution to streamline and automate a key submodule of the hiring process for a large-scale company with daily recruitment operations. The goal was to enhance the speed and quality of interview feedback, reducing manual effort and improving decision-making efficiency.

* **Problem**: With high-volume recruitment, providing fast feedback on interviews was a challenge, leading to delays in decision-making.

* **Solution**: I built an end-to-end pipeline that processes resumes, job descriptions, and interview videos to generate structured candidate insights. The pipeline involves:

  1. **Parsing the Interview Video**: The interview is initially in MP4 format, which is converted to WAV, followed by **speaker diarization** (speaker separation) to differentiate between the candidate and interviewer.
  2. **Speech-to-Text (STT)**: Using the *Faster-Whisper* model, the audio is transcribed to text, preserving context and speaker differentiation.
  3. **Semantic Chunking and Grouping**: The interview text is chunked based on sentence structure and semantically grouped using embeddings with a threshold to ensure coherent context.
  4. **Summarization**: Interview responses are summarized alongside the resume and job description to provide a concise, yet comprehensive evaluation. I carefully managed the context window to preserve the most relevant information while keeping the text concise.

* **Built with**: *Pyannote*, *Faster-Whisper*, *Sentence Transformers*, and *LLaMA 3.1*

* **Hosted on**: An *AWS EC2 GPU-enabled instance* for high-performance processing, ensuring scalability and fast inference times.

* **Techniques**: Applied *multi-threading* and *multiprocessing* for efficient resource usage and load balancing during peak usage.

This solution drastically reduces the time spent on manual evaluation and provides structured, actionable insights to streamline the hiring process.

