# import speech_recognition as sr
# import sounddevice as sd
# import numpy as np
#
#
# def get_audio_input():
#     recognizer = sr.Recognizer()
#
#     # Ustawienia nagrywania
#     samplerate = 16000  # Sample rate dla nagrywania
#     duration = 5  # Czas nagrywania w sekundach
#
#     # Nagrywanie dźwięku
#     print("Say something...")
#     audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
#     sd.wait()  # Czekaj na zakończenie nagrywania
#
#     # Przetwarzanie nagrania na dane audio do rozpoznawania mowy
#     audio = sr.AudioData(audio_data.tobytes(), samplerate, 2)
#
#     try:
#         text = recognizer.recognize_google(audio, language="en-US")
#         print(f"Recognized: {text}")
#         return text
#     except sr.UnknownValueError:
#         print("I couldn't understand, please try again.")
#         return None
#     except sr.RequestError:
#         print("There was an issue with the connection.")
#         return None
#
#
# if __name__ == "__main__":
#     while True:
#         user_input = get_audio_input()
#         if user_input:
#             print(f"User said: {user_input}")
#             if user_input.lower() == "exit":
#                 print("Program terminated.")
#                 break
