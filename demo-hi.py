import gradio as gr
import nemo
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy
import librosa
import tempfile
import soundfile
import uuid
import os
 
model = nemo_asr.models.EncDecCTCModelBPE.restore_from("stt/models/hi/c_m_hi.nemo")
SAMPLE_RATE= 16000

def reformat_audio(file):
    data, sr = librosa.load(file)
    if sr!=16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    data = librosa.to_mono(data)
    return data
    
    
def transcribe(microphone, audio):
    '''
    Speech to text fxn
    '''

    audio_data = None
    
    if (microphone is not None) and (audio is not None):
        print(
            "WARNING: You have uploaded an audio file and used the microphone. "
            "The recorded file from the microphone will be used and the uploaded audio will be discarded. \n"
        )

        audio_data = microphone
    elif (microphone is None) and (audio is None):
        print("ERROR: You need to either use the microphone or upload an audio file.")
    elif microphone is not None:
        audio_data = microphone
    else:
        audio_data = audio
    

    audio_data = reformat_audio(audio_data)

    
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, f'audio_{uuid.uuid4()}.wav')
        soundfile.write(audio_path, audio_data, SAMPLE_RATE)
        transcriptions = model.transcribe([audio_path])
        transcriptions = transcriptions[0]


    print(transcriptions)

    return transcriptions


with gr.Blocks() as demo:
    gr.Markdown("Speech Recognition")
    with gr.Tab("Transcribe Audio"):
        with gr.Row() as row:
            file_upload = gr.components.Audio(source="upload", type = "filepath", label="Upload File")
            microphone = gr.components.Audio(source="microphone", type="filepath", label="Microphone")
        run = gr.components.Button('Transcribe')

        transcript = gr.components.Label(label='Transcript')
        

        run.click(
            transcribe, inputs=[microphone, file_upload], outputs=[transcript]
        )


demo.launch()




