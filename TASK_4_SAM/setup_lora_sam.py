from peft import LoraConfig, get_peft_model

def setup_lora_sam(model, r=16):
    # congeliamo TUTTI i parametri originali
    for param in model.parameters():
        param.requires_grad = False

    # configurazione LoRA
    config = LoraConfig(
        r=r,
        lora_alpha=r,
        target_modules=["qkv"],
        lora_dropout=0.1,
        bias="none"
    )

    model.image_encoder = get_peft_model(model.image_encoder, config)

    # parametri addestrabili
    model.image_encoder.print_trainable_parameters()
    return model