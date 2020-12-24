import gsp_code
import noise
from matrix import wave
from sklearn import preprocessing
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import *
import numpy as np
import pandas as pd

global filename


def Raman_Photolum_split(f, gsp_function,  wavelength):
    def rem(x):
        res = []
        for i in x:
            if i not in res:
                res.append(i)
        return res

    def rem0(x):
        res = []
        for i in x:
            if i == 0.0:
                res.append(0)
            else:
                res.append(i)
        return res

    file = open(f, "r")  # read file
    list1 = list(file.readlines())  # list of lines
    row1 = list1[0]  # access each row
    # x1 y1
    x2 = []
    y2 = []
    z = [[]]
    for lol in range(1, len(list1)):
        row = list1[lol]
        x2.append(float(row.split("\t")[0]))
        y2.append(float(row.split("\t")[1]))
    # removed dubli
    x1 = rem(x2)
    x2 = rem0(x2)
    y2 = rem0(y2)
    y1 = rem(y2)
    # z
    z = [[float(list1[j].split("\t")[k]) for k in range(2, len(list1[j].split("\t")))] for j in range(1, len(list1))]
    matrix = []  # ed array
    matrix2 = []  # ed array
    k = 0
    for i in range(len(x1)):
        matrix.append([])
        matrix2.append([])
        for j in range(len(y1)):
            matrix[i].append(z[k])
            matrix2[i].append(z[k])
            k = k + 1
    file.close()
    # after import noise and gsp

    for i in range(len(matrix)):
        aa = gsp_function.split_Raman_photoluminescence(noise.smoothing(matrix[i]), wavelength)
        for j in range(len(matrix[i])):
            matrix[i][j] = aa[0][j]
            matrix2[i][j] = aa[1][j]

    for i in range(len(matrix)):

            matrix[i]=preprocessing.normalize(matrix[i])

    for i in range(len(matrix2)):
          matrix2[i] = preprocessing.normalize(matrix2[i])



    f_splited_array=f.split("/")
    file=f_splited_array[len(f_splited_array)-1].split(".")[0]
    # texts
    file2 = open( file+"_RS.txt", "a")
    file3 = open(file+"_PL.txt", "a")
    file2.write(row1)
    file3.write(row1)

    lol = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            file2.write(str(x2[lol]) + "\t" + str(y2[lol]) + "\t")
            for k in range(len(matrix[i][j])):
                if k != len(matrix[i][j]) - 1:
                    file2.write(str(format(matrix[i][j][k], '.6f')) + "\t")
                else:
                    file2.write(str(format(matrix[i][j][k], '.6f')))
            file2.write("\n")
            lol += 1

    lol = 0
    for i in range(len(matrix2)):
        for j in range(len(matrix2[i])):
            file3.write(str(x2[lol]) + "\t" + str(y2[lol]) + "\t")
            for k in range(len(matrix2[i][j])):
                if k != len(matrix2[i][j]) - 1:
                    file3.write(str(format(matrix2[i][j][k], '.6f')) + "\t")
                else:
                    file3.write(str(format(matrix2[i][j][k], '.6f')))
            file3.write("\n")
            lol += 1

    label3 = ttk.Label()
    label3.configure(text="Finished with file " + f)
    label3.pack()


def Sample_split(f, gsp_function):
    file = open(f, "r")  # read file
    list1 = list(file.readlines())  # list of lines
    # x1 y1
    wavelength = []
    y2 = []

    for lol in range(0, len(list1)):
        row = list1[lol]
        wavelength.append(float(row.split("\t")[0]))
        y2.append(float(row.split("\t")[1]))

    file.close()
    matrix = [[y2[k] for k in range(len(wavelength))]]
    a = gsp_function.split_Raman_photoluminescence(noise.smoothing(matrix), wavelength)
    matrix = a[0]
    matrix2 = a[1]

    matrix = preprocessing.normalize(matrix)
    matrix2 = preprocessing.normalize(matrix2)
    f_splited_array = f.split("/")
    f = f_splited_array[len(f_splited_array) - 1].split(".")[0]
    file2 = open(f+"_RS" + '.txt', "a")
    file3 = open(f+"_PL"+'.txt', "a")
    lol = 0
    for i in range(len(matrix[0])):
        file2.write(str(wavelength[lol]) + "\t" + str(matrix[0][lol]) + "\n")
        lol += 1

    lol = 0
    for i in range(len(matrix2[0])):
        file3.write(str(wavelength[lol]) + "\t" + str(format(matrix[0][lol], '.6f')) + "\n")
        lol += 1

    print("Finished with file " + f)


def Run():
  filename=Import_File()
  for f in filename:
    wavelength = wave(f)
    Raman_Photolum_split(f, gsp_code, wavelength)  # Returns a raman_file and a photolum


def Run2():
  filename=Import_File()
  for f in filename:
    Sample_split(f, gsp_code)  # Returns a raman_file and a photolum


def Run3():
    """ This works for map (like the Alina data)"""
    filename = Import_File()
    for f in filename:
        # Read the file and extact wavelength and img
        df = pd.read_csv(f, delimiter='\t', skipinitialspace=True, header=None, skiprows=[])
        wavelength = df.iloc[0].to_numpy()[2:]
        data = df.iloc[1:].to_numpy()
        pos = df.iloc[1:,:2].to_numpy()

        # This can be used to transform the txt to an image (but is not needed for this application)
        img = False
        if img:
            X = list(np.unique(data[:,0]))
            Y = list(np.unique(data[:,1]))

            img = np.empty((len(X), len(Y), len(wavelength)), dtype=np.float64)

            for d in data:
                i = X.index(d[0])
                j = Y.index(d[1])
                img[i,j,:] = d[2:]
            X = img.reshape(-1, len(wavelength))
        else:
            X = data[:,2:]

        # extract both Signals
        X_smooth = noise.smoothing(X)
        print(f"Done smoothing {f}")
        X_raman, X_PL = gsp_code.split_Raman_photoluminescence(X_smooth, wavelength)
        print(f"Done splitting {f}")

        matrix = preprocessing.normalize(X_raman)
        matrix2 = preprocessing.normalize(X_PL)

        f_splited_array = f.split("/")
        f = f_splited_array[len(f_splited_array) - 1].split(".")[0]
        file2 = f+"_RS" + '.txt'
        file3 = f+"_PL"+'.txt'

        np.savetxt(file2, np.vstack((np.concatenate(([np.nan,np.nan], wavelength)), np.hstack((pos, matrix)))), delimiter='\t')
        np.savetxt(file3, np.vstack((np.concatenate(([np.nan,np.nan], wavelength)), np.hstack((pos, matrix2)))), delimiter='\t')

        print("Finished with file " + f)


def Import_File(event=None):
    file_list=[]
    filename = filedialog.askopenfilenames()

    file_list.append(filename)

    label2=ttk.Label()
    label2.configure(text=filename)
    label2.pack()
    print('Selected:', filename)
    return filename


#GUI
root = tk.Tk()
root.title("Raman and Photoluminescence Signal Split - @ 2020 FORTH IMBB - G.Manios, V.Psilakis")
photo = PhotoImage(file = r"TIPP_LOGO.png")
photoimage = photo.subsample(1, 1)
photoimage2 = photo.subsample(1, 1)
button=tk.Button(root,height=100, width=200,text='Split Signals (SAMPLE) ',command=Run2,image=photoimage2,compound=LEFT).pack(side = TOP)
button2 = tk.Button(root,height=100, width=200, text='Split Signals (MAP)', command=Run3,image=photoimage,compound=LEFT).pack(side = TOP)
label = tk.Label(text="Press Split Signals - Choose your file(s) - Wait for the results ! ")
label.pack()
root.mainloop()
