
# coding: utf-8

import numpy as np
import scipy
from scipy.signal import hamming, blackmanharris, spectrogram, medfilt, convolve2d, argrelmax
from scipy.io import loadmat, savemat
import librosa
import matplotlib


##### Customised Parameters #####
midiNotes = np.arange(60,84+1)
endTimeInSecond = 4
inputFileIndicator = './data/note-%s.wav'
outputTemplate = './result/templates.mat'
initialFileH = './data/initialH.mat'

parameters_R = 1 # 1 pitch to train the template
parameters_update = np.array([1,1,1,0,1]) # update flags for [W,TS,a,H,pattern]
parameters_sparsity = np.array([1,1.04]) # annealing sparsity
parameters_threshold = -30
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
        initialisation_W = templates['W']
        initialisation_TS = templates['TS']
        initialisation_a = templates['a']
        initialisation_P = templates['pattern']
    else:
        initialisation_W = np.random.rand(F,R)
        initialisation_TS = np.ones((F,R))
        initialisation_a = np.ones((R,1)) 
        initialisation_P = np.ones((1,int(2*Tt+1)))

    if bool(H.any()):
        initialisation_H = H
    else:
        prng = np.random.RandomState(42)
        initialisation_H = prng.rand(R,T)
        initialisation_H = loadmat(initialFileH)['H']

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
##############################


X = computeTFR(inputFileIndicator%midiNotes[0], endTimeInSecond)
freqBinNum,frmNum = np.shape(X)
frmInd = int(frmNum*0.5/T - 1)
Hini = np.zeros((parameters_R,frmNum))
Hini[0,:][frmInd] = 0.99
templates = np.array([])
initialisation = setInitialisation(templates,X,Hini,parameters)
result = convNMFT(X,initialisation,endTimeInSecond)

templates_W = np.zeros((freqBinNum,len(midiNotes)))
templates_TS = np.zeros((freqBinNum,len(midiNotes)))
templates_a = np.zeros((len(midiNotes),1))
templates_pattern = np.zeros((len(midiNotes),result['pattern'].shape[1]))

for (ind,midi) in enumerate(midiNotes):
    inputFile = inputFileIndicator%midi
    X = computeTFR(inputFile,endTimeInSecond)
    initialisation = setInitialisation(templates,X,Hini,parameters)
    result = convNMFT(X,initialisation,endTimeInSecond)

    templates_W[:,ind] = result['W'][:,0]
    templates_TS[:,ind] = result['TS'][:,0]                  
    templates_a[ind,:] = result['a']
    templates_pattern[ind,:] = result['pattern']

    print("Isolated note midi no.%s template obtained." %midi)


pattern_average = np.average(templates_pattern,axis=0)
templates_pattern_average = pattern_average.reshape(-1,pattern_average.shape[0])


templates = {'W':templates_W,'TS':templates_TS,'a':templates_a,'pattern':templates_pattern_average,'midiNotes':midiNotes}
savemat(outputTemplate, templates)
print("Finished templates training from %s isolated notes!" %len(midiNotes))
