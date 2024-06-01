import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import pytesseract
from pytesseract import Output
from PIL import Image,ImageTk
img_label=None
def load_img():
    global file_path
    global img_label
    global img_tk
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg")])
    img = cv2.imread(file_path)
    RGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
    img_r=cv2.resize(RGB,(500,300))
    img_pil = Image.fromarray(img_r)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    if img_label:
        img_label.destroy()
    img_label = tk.Label(root, image=img_tk)
    
    img_label.image = img_tk  
    img_label.pack(side='left') 
def preprocess_and_classify():
    img = cv2.imread(file_path)   
    if file_path:
        my_config1=r"--psm 6 osm 3"
        my_config2=r"--psm 3 osm 3"
        img_r = cv2.resize(img, (1200, 1000))
        def bradly(img, window_size, k):
             gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
             a=cv2.morphologyEx(gray_img,cv2.MORPH_OPEN,kernel)
             b=cv2.morphologyEx(a,cv2.MORPH_CLOSE,kernel)
             v=cv2.GaussianBlur(b,(5,5), 0.5) 
             local_mean = cv2.boxFilter(v, ddepth=-1, ksize=(window_size, window_size))
             threshold = local_mean * (1-k)
             binary_img = np.zeros_like(b)
             binary_img[b > threshold] = 255
             return binary_img
        
        def sauvola_threshold(img, window_size, k, r):
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
            a=cv2.morphologyEx(gray_img,cv2.MORPH_OPEN,kernel)
            b=cv2.morphologyEx(a,cv2.MORPH_CLOSE,kernel)
            c=cv2.GaussianBlur(b,(7,7),0.5)
            local_mean = cv2.boxFilter(c, ddepth=-1, ksize=(window_size, window_size))
            local_sqr_mean = cv2.boxFilter(c * c, ddepth=-1, ksize=(window_size, window_size))
            local_std = np.sqrt(local_sqr_mean - (local_mean * local_mean))
            threshold_img = local_mean * (1.0 + k * ((local_std / r) - 1.0))
            binary_img = np.zeros_like(b)
            binary_img[b > threshold_img] = 255
            return binary_img

        def de_skew(img,angle):             
             h,w=img.shape[:2]
             center=(w//2,h//2)
             rot=cv2.getRotationMatrix2D(center,angle,1)
             rot_img=cv2.warpAffine(img, rot, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
             return rot_img
        window_size = 35
        k1=0.07
        k2=0.2
        r=128
        binary_image1 = bradly(img_r, window_size, k1)
        binary_image2=sauvola_threshold(img_r,window_size,k2,r)
        rotate1=de_skew(binary_image1,-2)
        rotate2=de_skew(binary_image2,-2)

        text = pytesseract.image_to_string(rotate1, config=my_config1, lang="eng")
        text2 = pytesseract.image_to_string(rotate1, config=my_config2, lang="eng")
        text3 = pytesseract.image_to_string(rotate2, config=my_config1, lang="eng")
        text4 = pytesseract.image_to_string(rotate2, config=my_config2, lang="eng")

        text=text.split()
        text2=text2.split()
        text3=text3.split()
        text4=text4.split()
        output="Image not clear, please provide clear image"
        lowercase_text = [word.lower() for word in text]
        lowercase_text2 = [word.lower() for word in text2]
        lowercase_text3 = [word.lower() for word in text3]
        lowercase_text4 = [word.lower() for word in text4]


        for word in lowercase_text2:
            lowercase_text.append(word)
        for word in lowercase_text4:
            lowercase_text3.append(word)


        for word in lowercase_text:
            if word in ["driving", "licence", "union", "vechicles"]:
                output = "Driving Licence"
                break
        for word in lowercase_text3:
            if word in ["driving", "licence", "union", "vechicles"]:
                output = "Driving Licence" 
                break    

        
        for word in lowercase_text:
            if word in ["govt.", "govt", "income", "tax", "department", "permanent", "account", "number"]:
                output = "Pan Card"
                break
        for word in lowercase_text3:
            if word in ["govt.", "govt", "income", "tax", "department", "permanent", "account", "number"]:
                output = "Pan Card"
                break   

        for word in lowercase_text:
            if word in ["government","govemment","goverment"]:
                output = "Aadhar card"
                break
        for word in lowercase_text3:
            if word in ["government","goverment","govemment"]:
                output = "Aadhar card"
                break

        for word in lowercase_text:
            if word in ["republic","passport","country","nationality"]:
                output = "Passport"
                break
        for word in lowercase_text3:
            if word in ["republic","passport","country","nationality"]:
                output = "Passport"
                break
        
        for word in lowercase_text:
            if word in ["election","commision","elector","identity"]:
                output = "Voter ID"
                break
        for word in lowercase_text3:
            if word in ["election","commision","elector","identity"]:
                output = "Voter ID"
                break


        result_text = "Document Type: " + output
        tk.messagebox.showinfo("Document Type", result_text)


root = tk.Tk()
root.title("Document Classification")

button_frame = tk.Frame(root)
button_frame.pack(side="right", padx=10, pady=10)
label=tk.Label(root,text="Document classifier",)
label.pack()
load_button=tk.Button(button_frame,text='Load Image',width=10,activeforeground="#000000",command=load_img,state="active")
load_button.pack(pady=10)

classify_button = tk.Button(button_frame, text="Classify",width=10,activeforeground='#000000',command=preprocess_and_classify)
classify_button.pack(pady=10)


root.mainloop()