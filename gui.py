import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageTk
import io
import tkinter as tk
import glob
import tkinter.ttk as ttk
import tensorflow as tf
import tkinter.messagebox as msg
import cv2
from resizer import fit
from tensorflow.keras.preprocessing import image
from test_structures import test_structure_basic, test_structure_small, test_structure_large


class GUI:
    def __init__(self, master):
        self.master = master
        self.master.configure(background='gray')
        self.master.minsize(1280, 800)
        self.master.resizable(0, 0)
        self.filename = "Empty"
        self.master.title("Style classifier")
        self.button = tk.Button(master=self.master, text="Browse", command=self.openFile)
        self.button.place(y=550, x=50)

        self.origName = tk.Label(self.master, text=self.filename)
        self.origName.place(y=550, x=100, width=450)

        img = "instagram-placeholder.png"
        photo = tk.PhotoImage(file=img)
        self.preview = tk.Label(self.master, image=photo)
        self.preview.configure(background='black')
        self.preview.image = photo
        self.preview.place(y=25, x=50, height=500, width=800)

        self.model = None
        self.model_name = None
        self.model_definitions = None

        self.scrollbar = tk.Scrollbar(self.master, orient=tk.VERTICAL)
        self.listBox = tk.Listbox(self.master, selectmode=tk.SINGLE, yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.listBox.yview)
        self.scrollbar.place(y=25, x=1150, height=500)
        self.listBox.place(y=25, x=900, width=250, height=500)
        self.listDesc = tk.Label(self.master, text="Models")
        self.listDesc.place(y=5, x=1000)
        self.displayButton = tk.Button(master=self.master, text="Analyze image", command=self.analyze_model)
        self.displayButton.place(y=525, x=900)

        i = 0
        for structure in [test_structure_basic, test_structure_small, test_structure_large]:
            for model_name, _ in structure.items():
                self.listBox.insert(i, model_name)

        self.ResDisplay = tk.Text(self.master)
        self.ResDisplay.config(state=tk.DISABLED)
        self.ResDisplay.place(y=600, x=50, height=150, width=1200)

        tf.config.set_visible_devices([], 'GPU')

    def analyze_model(self):
        i = self.listBox.curselection()
        if len(i) > 0:
            classes = ['Cubism', 'Impressionism', 'Photorealism', 'Pop art']
            index = i[0]
            selected_name = self.listBox.get(index, index+1)[0]
            p1 = "E:\\tf\\checkpoints\\"
            p2 = "E:\\tf\\Big_model_checkpoints\\"
            folder = ''
            for p in [p1, p2]:
                tmp = glob.glob(p+"*"+selected_name+"*")
                if len(tmp) > 0:
                    folder = tmp[0]
                    break
            if folder == '':
                msg.showerror("Error", "Model folder not found!")
            else:
                if self.origName['text'] != "Empty":
                    self.model = tf.keras.models.load_model(folder + "\\1\\best_model.hdf5")
                    self.model_name = selected_name
                    self.model_definitions = {**test_structure_small,
                                              **test_structure_basic,
                                              **test_structure_large}[selected_name]

                    img = image.load_img(self.origName['text'], target_size=self.model_definitions['pref-size'])
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = self.model_definitions['Preprocessing_function'](x)

                    prediction = self.model.predict(x)
                    confidence = np.max(prediction)
                    cls = list(prediction[0]).index(confidence)

                    self.ResDisplay.config(state=tk.NORMAL)
                    self.ResDisplay.delete('1.0', tk.END)
                    self.ResDisplay.insert(tk.END, "Model "+selected_name+" classifies the image as "+classes[cls]
                                           + " with "+str(round(confidence*100, 2))+"% confidence.")
                    self.ResDisplay.config(state=tk.DISABLED)
                else:
                    msg.showerror("Error!", "No image selected!")

        else:
            msg.showerror("Error!", "No model selected!")

    def openFile(self):  # open file with original image and create a thumbnail
        filename = tk.filedialog.askopenfilename(initialdir="/", title="Select file",
                                                 filetypes=(("jpeg files", "*.jpg"),
                                                            ("png files", "*.png"),
                                                            ("bmp files", "*.bmp"),
                                                            ("tif files", "*.tif"),
                                                            ("all files", "*.*")))
        if filename != "Empty" and len(filename) > 0:
            self.filename = filename
            maxwidth = self.preview.winfo_width()
            maxheight = self.preview.winfo_height()
            img = cv2.imread(self.filename)
            if img.shape[2] > 1:
                b, g, r = cv2.split(img)
                img = cv2.merge((r, g, b))
            imgtk = fit(img, maxwidth, maxheight)
            self.preview.configure(image=imgtk)
            self.preview.image = imgtk
            self.origName.configure(text=self.filename)


if __name__ == "__main__":
    root = tk.Tk()
    menu = GUI(root)
    root.mainloop()
