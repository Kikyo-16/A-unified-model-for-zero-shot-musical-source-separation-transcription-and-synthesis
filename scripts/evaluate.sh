cuda_id=0
evaluate_epoch=199

# evaluate MSI-DIS model
sh scripts/evaluate-model.sh $cuda_id MSI $evaluate_epoch save_model/MSI 8
