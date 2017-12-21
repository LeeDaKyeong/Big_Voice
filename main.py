import util
import speaker_recog

if __name__ == "__main__":
    audio_path = "./녹음/11.30/LS_11.30/LS_sentence1/sentence1_2.wav"
    speaker = speaker_recog.speaker_recog(audio_path)
    print(speaker)