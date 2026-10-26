# Multilingual Text-to-Speech Generation Using Indic Parler-TTS

## Project Description

This project is a deep learning-based text-to-speech application that converts written text into spoken audio using the pretrained Indic Parler-TTS model. It focuses on multilingual speech generation, particularly Malayalam and English, subject to model support.

The application provides a simple interface for entering text, generating speech, listening to the output, and downloading the generated audio.

## Objectives

- Convert text into speech using a pretrained deep learning model.
- Explore multilingual speech synthesis for Indian languages.
- Provide audio playback and download functionality.
- Evaluate speech quality and generation time.

## Technologies Used

- Python
- PyTorch
- Indic Parler-TTS
- Hugging Face Transformers
- Streamlit
- SoundFile
- Google Colab

## Project Structure

```text
Indic_TTS_Project/
├── app.py
├── requirements.txt
├── README.md
├── screenshots/
└── outputs/
```

## Installation

1. Install Python and Git on your computer.
2. Open a terminal in the project folder.
3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Obtain access to the model on Hugging Face and authenticate using a read-access token.
5. Run the application:

```bash
streamlit run app.py
```

Note: Model access, compatible library versions, and suitable hardware are required. Google Colab can be used for model inference if local resources are insufficient.

## How to Use

1. Open the application.
2. Enter text in a supported language.
3. Select the appropriate language or voice option if available.
4. Generate speech.
5. Play and download the generated audio.

## Evaluation

The application can be evaluated using speech intelligibility, pronunciation quality, and audio generation time for test sentences in supported languages.

## Future Enhancements

- Support for additional Indian languages.
- Improved pronunciation and speech quality.
- Additional voice and speech settings.

## Author

BHADRA R B
