
# coding: utf-8

import os
import numpy as np
import scipy
from scipy.signal import hamming, blackmanharris, spectrogram, medfilt,convolve2d,argrelmax
from scipy.io import loadmat
import librosa
import matplotlib

##### Parameters required to specify for the transcription #####
initNote = 60 # starting midi note from the trained note
templateFile = os.path.abspath('./result/templates.mat')
inputFile = os.path.abspath('./data/arpeggio-example.wav')
resultFile = os.path.abspath('./result/arpeggio-example-transcription.npy')
pianoRollFile = os.path.abspath('./result/arpeggio-example-pianoroll.npy')
y, fs = librosa.load(inputFile,sr=44100)
endTimeInSecond = int(np.floor(librosa.get_duration(y=y, sr=fs))) # audio length in second

parameters_R = len(np.arange(60,84+1)) # how many notes are used as candidate (should equal to the number of traning notes)
parameters_update = np.array([0,0,0,1,0]) # update flags for [W,TS,a,H,pattern]
parameters_sparsity = np.array([1,1.04]) # annealing sparsity
parameters_threshold = -20
parameters = {'R':parameters_R,'update':parameters_update,'sparsity':parameters_sparsity,'threshold':parameters_threshold}
##############################

##### Define Fuctions #####

# equivalent to function smooth in matlab, i.e.moving average
def smooth(a,WSZ):
    WSZ = WSZ-1+WSZ%2
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

# slice the array so as to be used for median filtering
def strided_app(aorin, L, S ):  # Window len = L, Stride len/stepsize = S
    # zero padding
    if(L%2==1):
        zeroPadS = (L-1)/2
        zeroPadE = (L-1)/2
    else:
        zeroPadS = L/2
        zeroPadE = L/2-1
    aPad = np.concatenate((np.zeros(zeroPadS),aorin))
    a = np.concatenate((aPad,np.zeros(zeroPadE)))
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def D(X,Y):
    XY = X*np.log(X/(Y+np.finfo(float).eps))-X+Y
    return XY

def DP(C, w):
    S,T = C.shape
    D = np.zeros((S,T))
    E = np.zeros((S,T))
    E[:,0] = np.arange(S)
    path = np.zeros(T)

    D[:,0] = C[:,0]
    for t in np.arange(1,T):
        for s in range(S):
            D[s,t] = np.min(D[:,t-1]+C[s,t]*w[:,s])
            E[s,t] = np.argmin(D[:,t-1]+C[s,t]*w[:,s],axis=0)

    path[T-1] = np.argmin(D[:,T-1])
    for t in np.arange(T-1)[::-1]:
        path[t] = E[int(path[t+1]),t]

    return D,path

def setGlobalDM(endTimeInSecond):
    # spectrogram factors 
    global window 
    global noverlap 
    global frame 
    global nfft 
    global fs
    window = hamming(4096)
    frame = 882
    noverlap = len(window)-frame
    nfft = 8192
    fs = 44100

    global Wt 
    global Tt 
    global T 
    global Tmax
    Wt = 0 # the number of frames of the harmonic part behind the onset
    Tt = int(np.floor(len(window)/float(frame))) # the spectral bluring length
    T = endTimeInSecond # the cutoff length in second
    Tmax = T/(frame/float(fs)) # the maximum duration of a note

def computeTFR(inputFile,endTimeInSecond): 
    # In matplotlib, the window size is specified using the NFFT argument. 
    # The window argument, on the other hand, is only for specifying the window itself, rather than the size.
    # the MATLAB window argument is split into the window and NFFT arguments in matplotlib,
    # while the MATLAB NFFT argument is equivalent to the matplotlib pad_to argument.
    from matplotlib.mlab import specgram
    
    # read spectrogram factors
    setGlobalDM(endTimeInSecond)

    # read audio
    fs = 44100
    x, sr = librosa.load(inputFile, sr=fs)

    # cut audios and add zeros at the end
    x = np.concatenate((x[:min(T*fs,len(x))],np.zeros(noverlap))) 

    S = specgram(x,window=window,NFFT=len(window),Fs=fs,pad_to=nfft,noverlap=noverlap,mode='magnitude')
    X_orin = S[0]

    # smoothing
    X_med = np.zeros((len(X_orin),len(X_orin[0])))
    for cind in range(len(X_orin[0])):
        X_med[:,cind]=medfilt(X_orin[:,cind],5)

    return X_med
    
def setInitialisation(templates,X,H,parameters):  
    F,T = np.shape(X)
    R = parameters['R']

    if templates != []:
#         initialisation_W = templates['W'][0,0]
        initialisation_W = templates['W']
        initialisation_TS = templates['TS']
        initialisation_a = templates['a'] 
        initialisation_P = templates['pattern']
    else:
        initialisation_W = np.random.rand(F,R)
        initialisation_TS = np.ones((F,R))
        initialisation_a = np.ones((R,1)) 
        initialisation_P = np.ones((1,int(2*Tt+1)))

    if bool(H):
        initialisation_H = H
    else:
        prng = np.random.RandomState(42)
        initialisation_H = prng.rand(R,T)
#         initialisation_H = loadmat('../relatedPaper/Tian_Supplement/code/H.mat')['H']

    initialisation_R = parameters['R']
    initialisation_update = parameters['update']
    initialisation_sparsity = parameters['sparsity']
    initialisation_beta = 1
    initialisation_iter = 50

    initialisation = {'W': initialisation_W,'TS': initialisation_TS,'a':initialisation_a,'P':initialisation_P,'H':initialisation_H,'R':initialisation_R,
                      'update':initialisation_update,'sparsity':initialisation_sparsity,'beta':initialisation_beta,'iter':initialisation_iter}
    
    return initialisation


def convNMFT(X, initialisation, endTimeInSecond):
    W = initialisation['W'] # harmonic templates
    TS = initialisation['TS'] # percussive templates
    a = initialisation['a'] # decay rates
    H = initialisation['H'] # activations
    pattern = initialisation['P'] # transient pattern
    R = initialisation['R'] # number of pitches
    update = initialisation['update'] # update flags for [W,TS,a,H,pattern]
    sparsity = initialisation['sparsity'] # control the sparseness of H
    beta = initialisation['beta'] # KL-divergence
    iteration = initialisation['iter'] # iteration number
    setGlobalDM(endTimeInSecond)

    T = np.shape(X)[1]
    ea = np.zeros((R,int(Tmax)))
    eat = np.zeros((R,int(Tmax)))
    Hea = np.zeros((R,T))
    Heat = np.zeros((R,T))
    WVXup = np.zeros((R,T))
    WVdown = np.zeros((R,T))
    TVXup = np.zeros((R,T))
    TVdown = np.zeros((R,T))
    Pup = np.zeros(int(2*Tt+1))
    Pdown = np.zeros(int(2*Tt+1))

    # reconstruction of V
    t = np.arange(Tmax)
    for r in range(R):
        ea[r] = np.exp(-a[r]*t)
    Ea = np.concatenate((np.zeros((R,Wt)),ea[:,:ea.shape[1]-Wt],np.zeros((R,int(T-Tmax)))),1)+np.finfo(float).eps

    for cind in range(T):
        Hea[:,cind] = np.sum(H[:,:(cind+1)]*Ea[:,:(cind+1)][...,::-1],axis=1)
    Hs = convolve2d(H, pattern, mode='same')
    V = W.dot(Hea) + TS.dot(Hs) + np.finfo(float).eps

    # update flag
    updateW = update[0]
    updateTS = update[1]
    updatea = update[2]
    updateH = update[3]
    updateP = update[4]

    spar = np.ones((iteration,1))
    if(len(sparsity)==1):
        spar[:] = sparsity
    elif(len(sparsity)==2):
        spar = sparsity[0]+(sparsity[1]-sparsity[0])*(np.arange(1,51))/float(iteration)


    for it in range(iteration):   

        if updateW:
            W = W * ((V**(beta-2) * X).dot(Hea.conj().T)) / ((V**(beta-1)).dot(Hea.conj().T)) + np.finfo(float).eps
            V = W.dot(Hea) + TS.dot(Hs) + np.finfo(float).eps

        if updateTS:
            TS = TS * ((V**(beta-2) * X).dot(Hs.conj().T)) / ((V**(beta-1)).dot(Hs.conj().T)) + np.finfo(float).eps
            V = W.dot(Hea) + TS.dot(Hs) + np.finfo(float).eps 

        if updatea:
            t = np.arange(Tmax)
            for r in range(R):
                eat[r] = np.exp(-a[r]*t)*t    
            Eat = np.concatenate((np.zeros((R,Wt)),eat[:,:eat.shape[1]-Wt],np.zeros((R,int(T-Tmax)))),1)+np.finfo(float).eps
            for cind in range(T):
                Heat[:,cind] = np.sum(H[:,:(cind+1)]*Eat[:,:(cind+1)][...,::-1],axis=1)
            for r in range(R):
                a[r] = a[r].dot((W[:,r].conj().T).dot(V**(beta-1)).dot(Heat[r,:].conj().T))/((W[:,r].conj().T).dot(V**(beta-2)*X).dot(Heat[r,:].conj().T))

            t = np.arange(Tmax)
            for r in range(R):
                ea[r] = np.exp(-a[r]*t)
            Ea = np.concatenate((np.zeros((R,Wt)),ea[:,:ea.shape[1]-Wt],np.zeros((R,int(T-Tmax)))),1)+np.finfo(float).eps

            for cind in range(T):
                Hea[:,cind] = np.sum(H[:,:(cind+1)]*Ea[:,:(cind+1)][...,::-1],axis=1)

            V = W.dot(Hea) + TS.dot(Hs) + np.finfo(float).eps   

        if updateH:
            WVX = np.concatenate((W.conj().T.dot(V**(beta-2)*X),np.zeros((R,T))),1)
            WV = np.concatenate((W.conj().T.dot(V**(beta-1)),np.zeros((R,T))),1)   

            for t in range(T):
                WVXup[:,t] = np.sum(WVX[:,int(t+Wt):int(t+Tmax)]*ea[:,:int(Tmax-Wt+1)],axis=1)
                WVdown[:,t] = np.sum(WV[:,int(t+Wt):int(t+Tmax)]*ea[:,:int(Tmax-Wt+1)],axis=1)

            TVX = np.concatenate((np.zeros((R,Tt)),TS.conj().T.dot(V**(beta-2)*X),np.zeros((R,Tt))),1)
            TV = np.concatenate((np.zeros((R,Tt)),TS.conj().T.dot(V**(beta-1)),np.zeros((R,Tt))),1)

            for t in range(T):
                TVXup[:,t] = TVX[:,t:int(t+2*Tt+1)].dot(pattern.conj().T)[:,0]
                TVdown[:,t] = TV[:,t:int(t+2*Tt+1)].dot(pattern.conj().T)[:,0]

            H = H * (WVXup+TVXup)/(WVdown+TVdown)
            H = H**spar[it]

            # normalise
            if R == 1:
                H = H/(a.max())

            # update V
            for cind in range(T):
                Hea[:,cind] = np.sum(H[:,:(cind+1)]*Ea[:,:(cind+1)][...,::-1],axis=1)
            Hs = convolve2d(H, pattern, mode='same')
            V = W.dot(Hea) + TS.dot(Hs) + np.finfo(float).eps

        if updateP:
            TVX = np.concatenate((np.zeros((R,Tt)),TS.conj().T.dot(V**(beta-2)*X),np.zeros((R,Tt))),1)
            TV = np.concatenate((np.zeros((R,Tt)),TS.conj().T.dot(V**(beta-1)),np.zeros((R,Tt))),1)

            for t in range(int(2*Tt+1)):
                Pup[t] = np.sum(np.sum(H*TVX[:,t:int(t+T)]))
                Pdown[t] = np.sum(np.sum(H*TV[:,t:int(t+T)]))

            pattern = pattern * Pup/Pdown
            pattern = pattern/pattern.max()

            Hs = convolve2d(H, pattern, mode='same')
            V = W.dot(Hea) + TS.dot(Hs) + np.finfo(float).eps   

    result = {'W':W,'TS':TS,'a':a,'H':H,'pattern':pattern}
    return result


# In[3]:


def detectingOnsets(H,Threshold):
    R,T = H.shape
    HP = np.zeros((R,T))
    P = np.zeros((R,T))

    Thre = 10**(Threshold/20.0)*H.max()

    for r in range(R):
        tH = H[r,:]
        tH[tH-smooth(tH,20)<Thre] = 0

        ind = argrelmax(tH)[0]
        c = tH[ind]

        if bool(ind.any()):
            dind = np.diff(ind)
            for i in range(len(dind)):
                if dind[i]<5:
                    ind[i] = round((c[i]*ind[i]+c[i+1]*ind[i+1])/(c[i]+c[i+1]))
                    ind[i+1] = 0
                    c[i] = c[i] + c[i+1]
                    c[i+1] = 0

            ind = [s for s in ind if s!=0]
            c = [s for s in c if s!=0]
            HP[r,ind] = c
            P[r,ind] = 1
    return HP

def detectingOffsets(onsets, offsets):
    offsets = np.asarray([s for s in offsets if s>=onsets[0]])

    Lon = len(onsets)
    Loff = len(offsets)

    P = np.zeros(((Lon+Loff),2))
    P[:,0] = np.concatenate((onsets,offsets))
    P[:,1] = np.concatenate((np.ones((Lon,1)),-1*np.ones((Loff,1))))[:,0]

    P = P[np.argsort(P[:,0]),:]

    Po = P

    for i in range(Lon+Loff-1):
        if(P[i,1]+P[i+1,1] == 2):
            Po = np.concatenate((Po,np.array([P[i+1,0]-1,-1]).reshape(-1,np.array([P[i+1,0]-1,-1]).shape[0])))
        elif(P[i,1]+P[i+1,1] == -2):
            Po[i+1,:] = 0

    PoNoZero = []
    for (row,col) in enumerate(Po):
        if(col[0] != 0):
            PoNoZero.append(row)
    Po = Po[PoNoZero]

    P = Po[np.argsort(Po[:,0]),:]

    offsets = P[1::2,0].astype(int)
    
    return onsets,offsets


def noteTracking(X, result, threshold, endTimeInSecond, initNote):
    W = result['W']
    TS = result['TS']
    a = result['a']
    H = result['H']
    pattern = result['pattern']

    # attack activations
    Ha = convolve2d(H, pattern, mode='same')
    HP = detectingOnsets(Ha,threshold) 
    HP = detectingOnsets(HP,threshold)

    setGlobalDM(endTimeInSecond)
    interval = frame/float(fs)

    # reconstruction of V
    R,T = H.shape
    ea = np.zeros((R,int(Tmax)))
    Hea = np.zeros((R,T))
    t = np.arange(Tmax)
    for r in range(R):
        ea[r] = np.exp(-a[r]*t)
    Ea = np.concatenate((np.zeros((R,Wt)),ea[:,:ea.shape[1]-Wt],np.zeros((R,int(T-Tmax)))),1)+np.finfo(float).eps
    for cind in range(T):
        Hea[:,cind] = np.sum(H[:,:(cind+1)]*Ea[:,:(cind+1)][...,::-1],axis=1)
    Hs = convolve2d(H, pattern, mode='same')
    V = W.dot(Hea) + TS.dot(Hs) + np.finfo(float).eps

    HPnoZero = 0
    for (row,col) in enumerate(HP):
        for (key,val) in enumerate(col):
            if val>0:
                HPnoZero += 1

    Note = np.zeros((HPnoZero,3))
    pianoRoll = np.zeros(H.shape)
    num = 0

    for r in range(R):

        onsets = argrelmax(HP[r,:])[0]
        if bool(onsets.any()):

            Vp = W[:,r].reshape(W[:,r].shape[0],-1).dot(Hea[r,:].reshape(-1,Hea[r,:].shape[0]))+TS[:,r].reshape(TS[:,r].shape[0],-1).dot(Hs[r,:].reshape(-1,Hs[r,:].shape[0])) + np.finfo(float).eps
            Vep = V - Vp + np.finfo(float).eps

            Cp = np.zeros((2,T))
            Cp[0,:] = np.sum(D(X,Vep),axis=0)
            Cp[1,:] = np.sum(D(X,V),axis=0)
            rCp = Cp/np.tile(np.sum(Cp,axis=0),(2,1))

            w = np.array([[0.5,0.55],[0.55,0.5]])

            offsets = np.array([T-1])
            onsets,offsets = detectingOffsets(onsets.conj().T, offsets.conj().T)

            for i in range(len(onsets)):
                index = np.where(abs(np.median(strided_app(np.diff(rCp[:,onsets[i]:offsets[i]+1], axis=0)[0],10,1),axis=1))<0.005)[0]
                index = np.asarray([ind for ind in index if ind>0])
                if bool(index.any()):
                    offsets[i] = onsets[i]+index[0]+1-1

                    Dis,path = DP(rCp[:,onsets[i]:offsets[i]+1],w);
                    duration = np.where(np.diff(np.append(path,0))==-1)[0]
                    if bool(duration.any()):
                        offsets[i] = onsets[i]+duration[0]+1-1

            for i in range(len(onsets)):
                pianoRoll[r,onsets[i]:offsets[i]+1] = 1
                Note[num,0] = (onsets[i]+1)*interval
                Note[num,1] = (offsets[i]+1)*interval
                
                Note[num,2] = r+initNote
                num = num+1

    Note = Note[np.argsort(Note[:,0]),:]
    return Note, pianoRoll

##############################


templates = loadmat(templateFile)
X = computeTFR(inputFile, endTimeInSecond)
initialisation = setInitialisation(templates,X,np.array([]),parameters)
result = convNMFT(X,initialisation, endTimeInSecond);
Note,pianoRoll = noteTracking(X, result, parameters['threshold'], endTimeInSecond, initNote)
# np.savetxt(resultFile, Note, delimiter=',')
np.save(resultFile, Note)
np.save(pianoRollFile, pianoRoll)

print("Transcription result of " + inputFile)
print("for each row of the result, it shows: onset time, offset time, note midi no.")
print(Note) # show the result on screen
