# hmm_speech_recognition_demo

### Setup Environment
This demo project is running on python3.x, please install the following required packages as well:
- python_speech_features: Calculation of MFCC features on audio 
- hmmlearn: Hidden Markov Models in Python, with scikit-learn like API	
- scipy: Fundamental library for scientific computing

#### Demo running results:
In python3.x, run the script `demo.py`, get the result below:
```
Finish prepare the training data
Finish training of the GMM_HMM models for digits 0-9
Test on true label  9 : predict result label is  4
Test on true label  7 : predict result label is  4
Test on true label  8 : predict result label is  1
Test on true label  5 : predict result label is  1
Test on true label  3 : predict result label is  4
Test on true label  6 : predict result label is  4
Test on true label  1 : predict result label is  1
Test on true label  4 : predict result label is  4
Test on true label  10 : predict result label is  10
Test on true label  2 : predict result label is  2
Final recognition rate is 40.00 %

```