from tkinter import *
from PIL import Image, ImageTk
from chat import get_response, bot_name

import speech_recognition as sr
import pyttsx3


BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

chatbot_function = "Restaurant Waiter Chatbot"
img ='Images/chatbot.ico'
title = "Waiter"


class ChatApplication:

    def __init__(self):
        self.window = Tk()
        self.window.iconbitmap(img)
        self._setup_main_window()



    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title(title)
        self.window.resizable(width=True, height=True)
        self.window.configure(width=640, height=480, bg=BG_GRAY)

        # head lebel
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR, text=chatbot_function, font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider     # separates chatbot_function from chatting
        #line = Label(self.window, width=450, bg=BG_GRAY)
        #line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # message entry box
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.11)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)


        # send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY, command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.85, rely=0.008, relheight=0.03, relwidth=0.15)


        microphoneImg = self.getImageButton("Images/microphone.jpg")   # image = microphoneImg, relief=GROOVE,    text="Voice"  # ImageTk.PhotoImage(file='Images/microphone_.png')#
        microphoneImg_button = Button(bottom_label, image = microphoneImg, font=FONT_BOLD, width=20, bg=BG_GRAY, command=self.speechRecognition)
        microphoneImg_button.image = microphoneImg
        microphoneImg_button.place(relx=0.9, rely=0.038, relheight=0.03, relwidth=0.05)




    def speechRecognition(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            #print("listening.....")
            r.pause_threshold = 1
            audio = r.listen(source)

        try:
            print("Recognizing.....")
            query = r.recognize_google(audio, language='en-ar')
            print(f"user said : {query}\n")


            self._insert_message(query, "You")

        except Exception as e:
            print("say that again please.....")
            return "None"




    def sayResponce(self, msg):
        text_speech = pyttsx3.init()
        text_speech.say(msg)
        text_speech.runAndWait()


    def _insert_voice_message(self, msg):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")



    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")

    def _insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)    # to delete the entered message from the enterring tap
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        #self.text_widget.configure(state=DISABLED)

        bot_resbonse = get_response(msg)
        msg2 = f"{bot_name}: {bot_resbonse}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        #self.text_widget.configure(state=DISABLED)
        #self.sayResponce(self.msg)
        self.sayResponce(bot_resbonse)


        self.text_widget.see(END)     # to make program scroll down automatically for the last message



    def getImageButton(self, imgPath):
        global img
        img = Image.open(imgPath)
        # Resizing image to fit on button
        resized_img = img.resize((30, 30))   # , Image.ANTIALIAS
        photoImg =  ImageTk.PhotoImage(resized_img)
        return photoImg











if __name__ == "__main__":
    app = ChatApplication()
    app.run()
