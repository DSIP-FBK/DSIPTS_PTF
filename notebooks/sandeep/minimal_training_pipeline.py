import pytorch_lightning as pl
from dsipts.data_structure.d1_layers import MultiSourceTSDataSet
from dsipts.data_structure.d2_layers import EncoderDecoder
from dsipts.models.LinearTS import LinearTS

# STEP 1 D1 Layer:  Data loading
d1_dataset = MultiSourceTSDataSet(
    file_paths=["data/_grouped.csv"],
    time_col="time",
    target_cols=["OT"],
    group_cols=["group"],
    enrich_cat=["hour", "dow"],  # Temporal enrichment
    global_forecasting=True,
)

# STEP 2: D2 Layer: Encoder Decoder
d2_datamodule = EncoderDecoder(
    d1_dataset=d1_dataset,
    past_len=96,
    future_len=24,
    batch_size=32,  # Reduced batch size
    scaling_method="standard",
    split_ratio=(0.7, 0.15, 0.15),
)

# step 3: Model init
# The model expects: past_steps × (past_channels + cat_emb_dim) when sum_emb=True
# Since sum_emb=True, all categorical features are summed into one embedding of size cat_emb_dim
model = LinearTS(
    past_steps=96,
    future_steps=24,
    past_channels=1,  # Only numeric features (OT)
    future_channels=0,
    embs=d1_dataset.metadata.get("cat_past_cardinalities", []),
    cat_emb_dim=8,
    kernel_size=25,
    sum_emb=True,
    out_channels=len(d1_dataset.target_cols),
    hidden_size=128,
    kind="linear",
    verbose=True,
)

# STEP 4: train with PyTorch Lightning
trainer = pl.Trainer(
    max_epochs=1,  # Reduced for quick demo
    accelerator="auto",  # should autoselect gpu if available
    devices=1,
)

print("Starting training...")
trainer.fit(model, d2_datamodule)
print("Training complete!")

# STEP 5: Test the model (using validation step instead)
print("\nEvaluating model...")
val_results = trainer.validate(model, d2_datamodule)
print(f"Validation results: {val_results}")

# STEP 6: Save the model (optional)
import torch
torch.save(model.state_dict(), "minimal_model.pt")
print("Model saved to minimal_model.pt")
