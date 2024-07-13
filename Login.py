import tkinter as tk
from tkinter import ttk, LEFT, END
from tkinter import messagebox as ms
import sqlite3
from PIL import Image, ImageTk
import re
import cv2

##############################################+=============================================================
root = tk.Tk()
root.configure(background="black")
root.geometry("700x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("700x700")
root.title("Login Form")




username = tk.StringVar()
password = tk.StringVar()
        

# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 = Image.open('1.jpg')
image2 = image2.resize((700,700), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)



def registration():
    from subprocess import call
    call(["python","registration.py"])
    root.destroy()

def login():
        # Establish Connection

    with sqlite3.connect('evaluation.db') as db:
         c = db.cursor()

        # Find user If there is any take proper action
         db = sqlite3.connect('evaluation.db')
         cursor = db.cursor()
         cursor.execute("CREATE TABLE IF NOT EXISTS admin_registration"
                           "(Fullname TEXT, address TEXT, username TEXT, Email TEXT, Phoneno TEXT,Gender TEXT,age TEXT , password TEXT)")
         db.commit()
         find_entry = ('SELECT * FROM admin_registration WHERE username = ? and password = ?')
         c.execute(find_entry, [(username.get()), (password.get())])
         result = c.fetchall()

         if result:
            msg = ""
            # self.logf.pack_forget()
            # self.head['text'] = self.username.get() + '\n Loged In'
            # msg = self.head['text']
            #            self.head['pady'] = 150
            print(msg)
            ms.showinfo("messege", "LogIn sucessfully")
            # ===========================================
            root.destroy()

            
            # ================================================
         else:
           ms.showerror('Oops!', 'Username Or Password Did Not Found/Match.')


l1 = tk.Label(root, text="LOGIN HERE", font=("Times new roman", 30, "bold"), bg="#192841", fg="white")
l1.place(x=250, y=150)

# that is for label1 registration

l2 = tk.Label(root, text="Username", width=12, font=("Times new roman", 20, "bold"), bg="snow")
l2.place(x=140, y=250)
t1 = tk.Entry(root, textvar=username, width=15, font=('', 20))
t1.place(x=370, y=250)
# that is for label 2 (full name)


l3 = tk.Label(root, text="Password", width=12, font=("Times new roman", 20, "bold"), bg="snow")
l3.place(x=140, y=300)
t2 = tk.Entry(root, textvar=password, width=15, font=('', 20),show="*")
t2.place(x=370, y=300)
# that is for label 3(address)
        

    
        



btn_log=tk.Button(root,text="Login",command=login,width=12,font=("Times new roman", 18, "bold"),bg="green",fg="white")
btn_log.grid(row=3,column=1,pady=10)
btn_log.place(x=400 , y=400 )
btn_reg=tk.Button(root,text="Register",command=registration,width=12,font=("Times new roman", 18, "bold"),bg="red",fg="white")
btn_reg.grid(row=3,column=0,pady=10)
btn_reg.place(x=170 , y=400 )   
        
    

root.mainloop()