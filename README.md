# A Unified Model for Zero-shot Music Source Separation, Transcription and Synthesis
This is the code repository for the paper: A Unified Model for Zero-shot Music Source Separation, Transcription and Synthesis. The paper is available [here](https://arxiv.org/abs/2108.03456).

## Introduction
We propose a unified model for three inter-related tasks:
- to *separate* individual sound sources from a mixed music audio;
- to *transcribe* each sound source to MIDI notes;
- to *synthesize* new pieces based on the timber of separated sources.

The model is inspired by the fact that when humans listen to music, our minds can not only separate the sounds of different instruments, but also at the same time perceive high-level representations such as score and timbre.

## Model architecture
### - Components of the proposed model
The proposed model comprises four components:
- a query-by-example (QBE) network
- a pitch-timber disentanglement module
- a transcriptor
- an audio encoder-decoder network

![](https://github.com/Kikyo-16/A-unified-model-for-zero-shot-musical-source-separation-transcription-and-synthesis/blob/main/imgs/model-fig-1-ab.png)
>The baseline models and the proposed model. In the left figure, the large orange and gray box indicate a QBE
transcription-only and QBE separation-only model respectively. The whole figure indicates a QBE multi-task model.


### - Training losses
The model is trained with separatiopn loss, transcription loss and contrastive loss. See details in [our paper](https://arxiv.org/abs/2108.03456).

### - Pitch-translation Invariance Loss
To further improve the timbre disentanglement performance, we propose a *pitch-translation invariance loss*. We term the model without pitch-transformation invariance loss `multi-task informed (MSI) model`. And we term MSI model with further disentanglement via pitch-transformation invariance loss `MSI-DIS model`.

### - Detailed hyper-parameters of the proposed model
![](https://github.com/Kikyo-16/A-unified-model-for-zero-shot-musical-source-separation-transcription-and-synthesis/blob/main/imgs/model-fig-3.png)

## Experimental results

|            Model|MSS-only|        Multi-task|       MSI (ours)| MSI-DIS (ours)|
|                  ----|        ----|        ----|        ----|        ----|
|  Seen|        4.69 ± 0.31| 3.32 ± 0.1|   **6.33 ± 0.17**|     5.04 ± 0.16|
|   Unseen|    **6.20 ± 0.26**|   4.63 ± 0.34|   5.53 ± 0.11|      3.99 ± 0.22| 
|   **Overall**|     5.07 ± 0.22|   3.65 ± 0.22|   **6.13 ± 0.15**|     4.77 ± 0.14|  


## Demos
The initial version of the demo page is available [here](https://kikyo-16.github.io/demo-page-of-a-unified-model-for-separation-transcriptiion-synthesis/). New demo page with more demos will be updated soon.

## Quick start

### Requirements
You will need at least Python 3.6 and Pytorch . See requirements.txt for requirements. Install dependencies with pip:
```
pip install -r requirements.txt
```

### Data preparation
1. Download URMP Dataset from [URMP homepage](http://www2.ece.rochester.edu/projects/air/projects/URMP.html).
2. Run the following command to generate your feature and annotations.
```
 python src/dataset/urmp/urmp_feature.py --dataset_dir=ur_unzipped_dataset_folder --feature_dir=dataset/hdf5s/urmp --process_num=1
```
**NOTE** that `ur_unzipped_dataset_folder` is your unzipped data folder and it should contain directories of songs:
> .
├── `ur_unzipped_dataset_folder`  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── `0_song0`  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── `1_song1`  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── ...  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  

### Training
Run the following command to train the proposed MSI-DIS Model:
```
python train.py --model_name=MSI_DIS --model_folder=folder_to_store_model_weights --epoch=200
```

### Evaluation
Download models weights [here](https://drive.google.com/drive/folders/1fT3Fva5JywhpYnOhsORbDkLQ9Vnhv_Lj?usp=sharing).
```
Run the following command to evaluate the proposed MSI-DIS Model on the test set:
```
python evaluate.py --model_name=MSI_DIS --model_path=path_of_model_weights --evaluation_folder=folder_to_store_evaluation_results --epoch=199
```
**NOTE:** Since we do not divide a validation set to chose the bestperformance model among all the training epochs, we report average results with a 95% confidence interval (CI) of models at the last 10 epochs.
Therefore, if you want to reproduce the results of our paper, please
1. Evaluate all last-10-epoch models.
2. Run the following command to print experimental result tables:
```
python src/analyze/draw_table.py --evaluation_folder=`folder_to_store_evaluation_results
```

### Synthesis
Run the following command to synthesize audios using the given midi, the test set, and the proposed MSI-DIS Model:
```
python synthesis.py --model_name=MSI-DIS --model_path=path_of_model_weights --evaluation_folder=folder_to_store_synthesis_results
```

## Citation
Please cite our work as:

>@inproceedings{lin2021unified,  
>title={A Unified Model for Zero-shot Music Source Separation, Transcription and Synthesis},   
>author={Liwei Lin and Qiuqiang Kong and Junyan Jiang and Gus Xia},  
>booktitle = {Proceedings of 21st International Conference on Music Information Retrieval, {ISMIR}},  
>year = {2021}  
>}

## License
This code is released under the MIT license as found in the LICENSE file.
