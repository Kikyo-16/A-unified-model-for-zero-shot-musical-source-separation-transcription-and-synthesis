cuda_id=1
evaluate_epoch=193

# evaluate MSI-DIS model
sh scripts/evaluate-model.sh $cuda_id MSS $evaluate_epoch save_model/wei_MSS-206
