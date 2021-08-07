# pack urmp dataset to hdf5
#sh scripts/generate_feature.sh

cuda_id=0

# train transcription-only baseline
#sh scripts/train-model.sh $cuda_id AMT save_model/AMT

# train separation-only baseline
#sh scripts/train-model.sh $cuda_id MSS save_model/MSS

# train multi-task baseline
#sh scripts/train-model.sh $cuda_id MSS-AMT save_model/MSS-AMT

# train the proposed multi-task score-informed (MSI) model
#sh scripts/train-model.sh $cuda_id MSI save_model/MSI

# train the proposed multi-task score-informed with further disentanglement (MSI-DIS) model
sh scripts/train-model.sh $cuda_id MSI-DIS save_model/MSI-DIS

