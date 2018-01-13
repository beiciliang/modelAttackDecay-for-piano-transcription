# modelAttackDecay-for-piano-transcription
Implementation of an attack/decay model for piano transcription

The code implements the estimation approach in

```
T. Cheng, M. Mauch, E. Benetos, and S. Dixon, 
“An attack/decay model for piano transcription,” 
in ISMIR, 2016.
```

The paper proposes a method based on non-negative matrix factorisation, with the following three refinements: (1) introduction of attack and harmonic decay components; (2) use of a spike-shaped note activation that is shared by these components; (3) modelling the harmonic decay with an exponential function. 

Transcription is performed in a supervised way, with the training and test datasets produced by the same piano. First we train parameters for the attack and decay components on isolated notes, then update only the note activations for transcription. 

### How to run
1. Clone the directory
```
 $ git clone https://github.com/beiciliang/modelAttackDecay-for-piano-transcription.git
```
2. Install the requirements using pip
```
$ cd modelAttackDecay-for-piano-transcription
$ pip install --user -r requirements.txt
```
3. Run the python file

To train parameters on isolated notes (20 example piano audio file `./data/note-*.wav`):
```
$ python train-template.py
```

This will save the trained parameters in file `./result/templates.mat`.

Transcription of example piano audio file `./data/arpeggio-example.wav` uses the above parameters to update the note activations by:
```
$ python nmf-transcription.py
```

It will return:
```
Transcription result of ./data/arpeggio-example.wav
for each row of the result, it shows: onset time, offset time, note midi no.
[[  0.48   1.2   60.  ]
 [  0.74   4.12  64.  ]
 [  0.98   4.12  67.  ]
 [  1.22   1.96  72.  ]
 [  1.48   2.08  76.  ]
 [  1.74   2.28  79.  ]
 [  1.98   2.42  84.  ]]
```

Transcription result is also saved in `./result/arpeggio-example-transcription.csv`.

#### If you wish to train and test on your own audio data, please change the initialisation at the beginning of each python script. 

P.S. A matlab implementation is provided at [Tian Cheng's soundsoftware repository](https://code.soundsoftware.ac.uk/projects/decay-model-for-piano-transcription).
