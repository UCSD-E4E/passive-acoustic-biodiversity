    #!/usr/bin/env python
    # coding: utf-8

    # In[3]:
if __name__ == '__main__':

    #!pip uninstall opensoundscape
    #!pip install opensoundscape==0.6
    print("START")
    from opensoundscape.preprocess.preprocessors import BasePreprocessor, CnnPreprocessor
    print("TEST")
    import torch
    import pandas as pd
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from opensoundscape.torch.models import cnn
    from opensoundscape.torch.architectures import cnn_architectures
    import sys
    sys.setrecursionlimit(3000)


    # In[4]:


    df = pd.read_csv('screaming_pyha_test.csv')
    ef = pd.read_csv('flycatcher_test.csv')
    ff = pd.read_csv('antbird_test.csv')
    gf = pd.read_csv('attila_test.csv')
    hf = pd.read_csv('peppershrike_test.csv')
    print(df.shape)
    print(ef.shape)
    print(ff.shape)
    print(gf.shape)
    print(hf.shape)


    # In[5]:


    # df = df.append(ef, ignore_index=True)
    # df = df.append(ff, ignore_index=True)
    # df = df.append(gf, ignore_index=True)
    # df = df.append(hf, ignore_index=True)
    # df.shape
    df = pd.read_csv('xeno_canto_train_data.csv')
    ef = pd.read_csv('xeno_canto_test_data.csv')
    df


    # In[6]:


    ef


    # In[ ]:


    #def fix_folder_path(filepath):
    #    folder = "./XenoCanto_Data/Training_Xeno_Canto_2022/"
    #    filename = filepath.split("/")[-1]
    #    return folder + filename
    #df.iloc[0]["file"]
    #df["file"] = df["file"].apply(fix_folder_path)
    #fix_folder_path('./temp_clips/../XenoCanto_Data/Training_Xeno_Canto_2022/Hemitriccus-griseipectus-130084_0.0s_3.0s.wav')
    #df


    # In[ ]:


    #def fix_folder_path(filepath):
    #    folder = "./XenoCanto_Data/Testing_Xeno_Canto_2022/"
    #    filename = filepath.split("/")[-1]
    #    return folder + filename
    #ef["file"] = ef["file"].apply(fix_folder_path)
    #ef


    # In[8]:


    # ff.drop('Unnamed: 0', inplace=True)
    #df = df.drop('Unnamed: 0', inplace=True)
    #ef = ef.drop('Unnamed: 0', inplace=True)
    df.columns


    # In[9]:


    df['MANUAL ID'].value_counts()


    # In[10]:


    ef['MANUAL ID'].value_counts()


    # In[15]:


    df.loc[df['MANUAL ID'].str.contains('Toucan'), 'MANUAL ID'] = ef.loc[ef['MANUAL ID'].str.contains('Toucan'), 'MANUAL ID'] = 'Toucan'
    df.loc[df['MANUAL ID'].str.contains('Flycatcher'), 'MANUAL ID'] = 'Flycatcher'
    df.loc[df['MANUAL ID'].str.contains('Thrush'), 'MANUAL ID'] = 'Thrush'
    df.loc[df['MANUAL ID'].str.contains('Vireo'), 'MANUAL ID'] = ef.loc[ef['MANUAL ID'].str.contains('Vireo'), 'MANUAL ID'] = 'Vireo'
    df.loc[df['MANUAL ID'].str.contains('Antwren'), 'MANUAL ID'] = ef.loc[ef['MANUAL ID'].str.contains('Antwren'), 'MANUAL ID'] = 'Antwren'
    df.loc[df['MANUAL ID'].str.contains('Tanager'), 'MANUAL ID'] = ef.loc[ef['MANUAL ID'].str.contains('Tanager'), 'MANUAL ID'] = 'Tanager'
    df.loc[df['MANUAL ID'].str.contains('Kingbird'), 'MANUAL ID'] = ef.loc[ef['MANUAL ID'].str.contains('Kingbird'), 'MANUAL ID'] = 'Kingbird'
    df.loc[df['MANUAL ID'].str.contains('Tody-Tyrant'), 'MANUAL ID'] = ef.loc[ef['MANUAL ID'].str.contains('Tody-Tyrant'), 'MANUAL ID'] = 'Tody-Tyrant'
    df.loc[df['MANUAL ID'].str.contains('Hummingbird'), 'MANUAL ID'] = ef.loc[ef['MANUAL ID'].str.contains('Hummingbird'), 'MANUAL ID'] = 'Hummingbird'
    df.loc[df['MANUAL ID'].str.contains('Antshrike'), 'MANUAL ID'] = ef.loc[ef['MANUAL ID'].str.contains('Antshrike'), 'MANUAL ID'] = 'Antshrike'

    classes = ['Antwren', 'Antshrike', 'Toucan', 'Vireo', 'Kingbird', 'Tody-Tyrant']
    df = df[df["MANUAL ID"].isin(classes)]
    ef = ef[ef["MANUAL ID"].isin(classes)]
    #for i, row in df.iterrows():
    #    if(row['MANUAL ID'] not in classes):
    #        df.drop(i, inplace=True)
    #for i, row in ef.iterrows():
    #    if(row['MANUAL ID'] not in classes):
    #        ef.drop(i, inplace=True)
    #print(df['MANUAL ID'].value_counts())
    #print(ef['MANUAL ID'].value_counts())


    # In[16]:


    from opensoundscape.annotations import categorical_to_one_hot
    one_hot_labels, trainClasses = categorical_to_one_hot(df[['MANUAL ID']].values)
    trainValDF = pd.DataFrame(index=df['file'],data=one_hot_labels,columns=trainClasses)


    # In[17]:


    one_hot_labels, testClasses = categorical_to_one_hot(ef[['MANUAL ID']].values)
    testDF = pd.DataFrame(index=ef['file'],data=one_hot_labels,columns=testClasses)


    # In[18]:


    '''
        'Antwren', 'Antshrike', 'Toucan', 'Vireo', 'Kingbird', 'Tody-Tyrant'
    '''


    # In[19]:


    trainClasses


    # In[20]:


    testClasses


    # In[21]:


    # reference - http://opensoundscape.org/en/latest/api/modules.html?highlight=multiclass#opensoundscape.torch.models.cnn.Resnet18Multiclass

    # http://opensoundscape.org/en/latest/api/modules.html?highlight=multiclass#opensoundscape.metrics.multiclass_metrics


    # In[22]:


    trainValDF


    # In[23]:


    from sklearn.model_selection import train_test_split
    train_df, valid_df = train_test_split(trainValDF, test_size=0.2, random_state=42)
    # train_df, test_df = train_test_split(train_df, test_size=0.1, random_state=42)


    # In[24]:


    print(train_df.shape)
    print(valid_df.shape)
    print(testDF.shape)


    # In[25]:


    train_df


    # In[26]:


    train_dataset = CnnPreprocessor(df=train_df)
    train_dataset.augmentation_on()
    train_dataset.actions.load_audio.set(sample_rate=44100)
    valid_dataset = CnnPreprocessor(df=valid_df)
    valid_dataset.augmentation_on()
    valid_dataset.actions.load_audio.set(sample_rate=44100)


    # In[27]:


    model = cnn.Resnet18Multiclass(trainClasses)


    # In[28]:


    #train_dataset.classes


    # In[29]:


    dir(train_dataset)


    # In[30]:


    model.classes


    # In[32]:


    model.train(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        save_path='./multilabel_train_six_classes_xeno_canto/',
        epochs=5, #10
        batch_size=32,
        save_interval=100,
        num_workers=0
    )


    # In[ ]:


    import matplotlib.pyplot as plt
    plt.scatter(model.loss_hist.keys(),model.loss_hist.values())
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


    # In[ ]:


    from opensoundscape.torch.models.cnn import load_model
    model = load_model('./multilabel_train_six_classes_xeno_canto/best.model')


    # In[ ]:


    prediction_dataset = model.train_dataset.sample(n=0)
    prediction_dataset.augmentation_off()
    prediction_dataset.df = testDF


    # In[ ]:


    valid_scores_df, valid_preds_df, valid_labels_df = model.predict(prediction_dataset,
                                                                    binary_preds='single_target',
                                                                    batch_size=16,
                                                                    num_workers=2,
                                                                    activation_layer='softmax')


    # In[ ]:


    from sklearn.metrics import classification_report, roc_auc_score, roc_curve
    print(classification_report(valid_labels_df, valid_preds_df))


    # In[ ]:


    testDF.columns


    # In[ ]:


    '''
        Toucan                       3813
    Vireo                        2217
    Eastern Kingbird             1403
    White-bellied Tody-Tyrant    1018
    '''
    fpr, tpr, thresh = roc_curve(valid_labels_df['Antwren'],  valid_scores_df['Antwren'])
    auc = roc_auc_score(valid_labels_df['Antwren'],  valid_preds_df['Antwren'])

    fpr2, tpr2, thresh2 = roc_curve(valid_labels_df['Kingbird'],  valid_scores_df['Kingbird'])
    auc2 = roc_auc_score(valid_labels_df['Kingbird'],  valid_preds_df['Kingbird'])

    fpr3, tpr3, thresh3 = roc_curve(valid_labels_df['Vireo'],  valid_scores_df['Vireo'])
    auc3 = roc_auc_score(valid_labels_df['Vireo'],  valid_preds_df['Vireo'])

    fpr4, tpr4, thresh4 = roc_curve(valid_labels_df['Toucan'],  valid_scores_df['Toucan'])
    auc4 = roc_auc_score(valid_labels_df['Toucan'],  valid_preds_df['Toucan'])

    fpr5, tpr5, thresh5 = roc_curve(valid_labels_df['Tody-Tyrant'],  valid_scores_df['Tody-Tyrant'])
    auc5 = roc_auc_score(valid_labels_df['Tody-Tyrant'],  valid_preds_df['Tody-Tyrant'])

    fpr6, tpr6, thresh6 = roc_curve(valid_labels_df['Antshrike'],  valid_scores_df['Antshrike'])
    auc6 = roc_auc_score(valid_labels_df['Antshrike'],  valid_preds_df['Antshrike'])

    plt.plot(fpr,tpr,label="AUC Antwren"+str(auc))
    plt.plot(fpr2,tpr2,label="AUC Kingbird"+str(auc2))
    plt.plot(fpr3,tpr3,label="AUC Vireo"+str(auc3))
    plt.plot(fpr4,tpr4,label="AUC Toucan"+str(auc4))
    plt.plot(fpr5,tpr5,label="AUC Tody-Tyrant"+str(auc5))
    plt.plot(fpr6,tpr6,label="AUC Antshrike"+str(auc6))
    # plt.title('Prediction on peru dataset (Train: microfaune)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.show()


    # In[ ]:




