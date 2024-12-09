# We used OpenAI's Whisper model locally for speech recognition. Thus, it needs to be installed with pip install openai-whisper
import speech_recognition as speech
# This needs to have ffmpeg installed on the system
from pydub import AudioSegment
import pytesseract
# We decided to use markdown2 instead of just markdown because the normal markdown package was incapable of handling tables,
# which is necessary for our use case as will be shown below
import markdown2
import cv2
import os

# Program parameters
# see comment below in `audio_to_text` about setting this
video_path_param = './Meditation For Programmers [5gZdTZa8bOw].mp4'
audio_path_param = video_path_param + '.wav'
markdown_path_param = './output.md'
html_path_param = './output.html'

def audio_to_text(video_path, audio_path):
    """
    Extracts audio from video and returns text from it using OpenAI's Whisper model (locally) alongside some other
    details.

    :param video_path: the path to the video file. If not absolute path (i.e. if it's a local path like just the name
      of the file or if it starts with './'), please make sure the current working directory is setup correctly, since
      the file would be located relative to the current working directory, and not the location of the script.
    :param audio_path: similar to `video_path`, this is the absolute path of the file to which the audio will be saved,
      so it can be processed.
    :return: Returns the Whisper dictionary of the transcription of the audio. This object has three attributes: text,
      language, segments. The first is a string which is the transcription of the audio. The second is the 2 character
      code of the language of the audio. The third is a list of dictionaries containing the transcription of each
      segment of the audio, including its starting and ending time.
    """
    audio = AudioSegment.from_file(video_path)
    # write the audio file so we can import it later with speech_recognizer
    audio.export(audio_path, format='wav')

    recognizer = speech.Recognizer()
    with speech.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    whisper_obj = recognizer.recognize_whisper(audio, show_dict=True)
    return whisper_obj

def video_to_text(video_path, frame_mod = 10):
    """
    Uses Tesseract OCR to scan frames of the video and returns a string with the results.

    :param video_path: the path to the video file. If not absolute path (i.e. if it's a local path like just the name
      of the file or if it starts with './'), please make sure the current working directory is setup correctly, since
      the file would be located relative to the current working directory, and not the location of the script.
    :param frame_mod: for the sake of efficiency, this function does not OCR every single frame; instead, only one frame
      every `frame_mod` frames is scanned by the OCR model. Increasing this parameter will increase the speed of OCR at
      the cost of accuracy, while decreasing it will increase the accuracy of OCR at the cost of efficiency.
    :return: A string containing the content scanned from the video frames using our OCR model.
    """

    video = cv2.VideoCapture(video_path)
    texts_set = set()
    count = 0
    success = 1
    ocr_text = ""

    while success:
        # get frame
        success, frame = video.read()

        if success and count % frame_mod == 0:
            # by default, OpenCV stores images in BGR format and since pytesseract assumes RGB format,
            # we need to convert from BGR to RGB format/mode:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_text: str = pytesseract.image_to_string(img_rgb).strip()
            # do this to avoid printing the same text multiple times
            if frame_text not in texts_set:
                # print(frame_text)
                ocr_text = ocr_text + frame_text + '\n'
                texts_set.add(frame_text)

        count += 1

    return ocr_text

def create_document(whisper_obj, ocr_text, markdown_path, html_path):
    """
    Takes the whisper object from `audio_to_text`, the ocr text from `video_to_text`, combines them together into
    Markdown text, writes the Markdown text to `markdown_path`, generates HTML from the Markdown text, and writes
    the HTML to `html_path`.
    """

    # The syntax used is a bit weird because python's multiline strings are horrible, especially with indentation
    markdown_string = (
        "# Audio Transcription\n"
        "\n"
        "This section contains audio transcription from the Whisper model in a table, where each phrase has its start"
        "and end time in seconds included in the table.\n"
        "\n"
        "| Start | End | Phrase |\n"
        "| ----- | --- | ------ |\n")
    for phrase in whisper_obj['segments']:
        markdown_string += f"| {phrase['start']:.2f} | {phrase['end']:.2f} | {phrase['text']} |\n"

    markdown_string += (
        "\n# Video Frames OCR Content\n"
        "\n"
        "This section contains the text extracted from the video frames using the Tesseract OCR model.\n"
        "\n"
        "```\n"
        f"{ocr_text}\n"
        "```\n")

    f = open(markdown_path, 'w')
    f.write(markdown_string)
    f.close()

    html = markdown2.markdown(markdown_string, extras=['tables', 'fenced-code-blocks'])
    f = open(html_path, 'w')
    f.write(html)
    f.close()

    return markdown_string

if __name__ == '__main__':
    print(
        create_document(
            audio_to_text(video_path_param, audio_path_param),
            video_to_text(video_path_param),
            markdown_path_param,
            html_path_param
        )
    )