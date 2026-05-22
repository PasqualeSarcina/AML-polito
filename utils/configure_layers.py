
def configure_model(model, unfreeze_last_n_layers):
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    
    for param in model.image_encoder.neck.parameters():
        param.requires_grad = True

    # Scongela solo gli ultimi N blocchi
    blocks_to_train = model.image_encoder.blocks[-unfreeze_last_n_layers :]

    print(f"Scongelamento degli ultimi {len(blocks_to_train)} blocchi.")
    for block in blocks_to_train:
        for param in block.parameters():
            param.requires_grad = True
    
    # Conta parametri
    trainable_params = sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.image_encoder.parameters())
    print(f"--- Configurazione Fine-Tuning ({unfreeze_last_n_layers} blocchi) ---")
    print(f"Parametri totali: {total_params:,}")
    print(f"Parametri allenabili: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")
