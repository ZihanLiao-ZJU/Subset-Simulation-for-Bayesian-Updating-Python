def DspRst(ite, Nsam, Nsze, Ncal):
    """Print the progress of one level (same layout as the MATLAB code)."""
    if ite == 1:
        print("---------------------------------------------")
        print("     i      Nsam      Nc      Ns      Ncal   ")
        print("---------------------------------------------")
    print("%6d    %6d  %6d  %6d    %6d " % (ite, Nsam, Nsze[0], Nsze[1], Ncal))
