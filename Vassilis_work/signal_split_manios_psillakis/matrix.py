def wave(filename):
    file = open(filename, "r")  # read file
    list1 = list(file.readlines())  # list of lines
    row22 = list1[0]
    # wavelength
    wavelength1 = row22.split("\t")
    wavelength = []
    for i in range(2, len(row22.split("\t"))):
        wavelength.append(float(wavelength1[i]))

    return wavelength
