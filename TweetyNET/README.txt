Pipeline:
* Data loading
    Where the datasets are and how to load them
    WAV Files
* Data preprocessing
    Mel Spectrograms and segmentation of clips
* Data augmentation (TODO)
    Add noise and other techniques to get more data from our data
* Training
    Outline the steps for training tweetynet and how that works
* Evaluating local Score arrays
    Currently, look for at least 1 positive bird call in the local score array.
    We do this by taking the max score in a clip.
* Hyperparameter tuning (Work In Progress)
    Optuna will try to tune these parameters to the model:
        learning rate
        batch size
        epochs
        Stopping conditions
* Results
    Performance Metrics:
        AUROC Curves
        Precision Recall curves
        Accuracy
        precision
        recall
* Conclusion
    Improve the current metrics for binary bird classification for whole clips
    Will need more data and datasets for binary bird classification at the strongly labeled level.

File Structure:
Pipeline:
    (NEED) Data loading script
    train_model.py
    NIPS_Training.py
    learn model.ipynb
    ICML_ROC_Training.ipynb
    
TweetyNet:
    network.py
    model.py
    TweetyNetModel.py
    TweetyNetAudio.py
    CustomAudioDataset.py
    EvaluationFunctions.py

Hyperparameter Tuning:
    optuna.py
Evaluation Files:
    Handle_predictions.ipynb
    Experiment1*.ipynb
