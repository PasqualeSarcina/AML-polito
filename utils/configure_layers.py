
def configure_model(model, unfreeze_last_n_layers):
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    
    for param in model.image_encoder.neck.parameters():
        param.requires_grad = True

    blocks_to_train = model.image_encoder.blocks[-unfreeze_last_n_layers :]

    print(f"Scongelamento degli ultimi {len(blocks_to_train)} blocchi.")
    for block in blocks_to_train:
        for param in block.parameters():
            param.requires_grad = True
    