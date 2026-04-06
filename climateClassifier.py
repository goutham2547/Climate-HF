import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as evaluation_tools
import copy
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from huggingface_hub import PyTorchModelHubMixin, login

# --- Global Settings ---
processing_unit = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FULL_COLLECTION = False  # Toggle to use entire dataset for final model weights
BATCH_VOLUME = 4
TOTAL_ITERATIONS = 12
AUTH_TOKEN = "PASTE_YOUR_HF_TOKEN_HERE"

# --- Architecture Definition ---
class ClimateDiscourseNet(nn.Module, PyTorchModelHubMixin):    
    """
    Multi-layer Perceptron for classifying climate change skepticism 
    using sentence embeddings.
    """
    def __init__(self, output_dimension=8):
        super().__init__()
        self.input_layer = nn.Linear(768, 128)
        self.hidden_a = nn.Linear(128, 128)
        self.hidden_b = nn.Linear(128, 64)
        self.hidden_c = nn.Linear(64, 32)
        self.classifier_head = nn.Linear(32, output_dimension)
        
        self.dropout_layer = nn.Dropout(0.15)
        self.non_linearity = nn.ReLU()

    def forward(self, feature_vectors):
        x = self.dropout_layer(self.non_linearity(self.input_layer(feature_vectors)))
        x = self.dropout_layer(self.non_linearity(self.hidden_a(x)))
        x = self.dropout_layer(self.non_linearity(self.hidden_b(x)))
        x = self.dropout_layer(self.non_linearity(self.hidden_c(x)))
        return self.classifier_head(x)

# --- Data Orchestration ---
def fetch_and_encode_assets():
    raw_corpus = load_dataset("quotaclimat/frugalaichallenge-text-train")
    encoder_engine = SentenceTransformer("sentence-transformers/sentence-t5-large")

    training_subset = raw_corpus['train']
    testing_subset = raw_corpus['test']

    text_samples_train = [item['quote'] for item in training_subset]
    text_samples_test = [item['quote'] for item in testing_subset]
    
    # Extracting numeric category from label string
    targets_train = [int(item['label'][0]) for item in training_subset]
    targets_test = [int(item['label'][0]) for item in testing_subset]

    if USE_FULL_COLLECTION:
        text_samples_train += text_samples_test
        targets_train += targets_test

    # Calculate class balancing weights
    occurrence_counts = [targets_train.count(idx) for idx in range(8)]
    penalty_weights = torch.FloatTensor([len(targets_train)/(count + 1) for count in occurrence_counts]).to(processing_unit)

    print("Transforming text to embeddings...")
    train_embeddings = torch.Tensor(encoder_engine.encode(text_samples_train))
    train_resource = TensorDataset(train_embeddings, torch.tensor(targets_train))
    train_generator = DataLoader(train_resource, sampler=RandomSampler(train_resource), batch_size=BATCH_VOLUME)

    test_embeddings = torch.Tensor(encoder_engine.encode(text_samples_test))
    test_resource = TensorDataset(test_embeddings, torch.tensor(targets_test))
    test_generator = DataLoader(test_resource, sampler=SequentialSampler(test_resource), batch_size=BATCH_VOLUME)

    return train_generator, test_generator, penalty_weights

# --- Execution Engine ---
def execute_training_cycle(neural_net, train_loader, val_loader, loss_weights):
    optimizer_engine = AdamW(neural_net.parameters(), lr=3e-4, weight_decay=0.02)
    dynamic_lr = ReduceLROnPlateau(optimizer_engine, mode='min', patience=3, factor=0.5)
    objective_function = nn.CrossEntropyLoss(weight=loss_weights)

    top_accuracy_score = 0
    optimized_params = None

    for epoch_idx in trange(TOTAL_ITERATIONS, desc="Training Progress"):
        neural_net.train()
        running_loss = 0.0
        
        for batch_data in train_loader:
            inputs, goal_labels = tuple(item.to(processing_unit) for item in batch_data)
            
            optimizer_engine.zero_grad()
            predictions = neural_net(inputs)
            loss_value = objective_function(predictions, goal_labels.long())
            
            loss_value.backward()
            optimizer_engine.step()
            running_loss += loss_value.item()

        # Validation Phase
        neural_net.eval()
        accumulated_preds, accumulated_labels = [], []
        sum_val_loss = 0.0
        
        with torch.no_grad():
            for val_batch in val_loader:
                v_inputs, v_labels = tuple(item.to(processing_unit) for item in val_batch)
                v_outputs = neural_net(v_inputs)
                
                v_loss = objective_function(v_outputs, v_labels.long())
                sum_val_loss += v_loss.item()
                
                accumulated_preds.extend(v_outputs.argmax(1).cpu().numpy())
                accumulated_labels.extend(v_labels.cpu().numpy())

        avg_loss = sum_val_loss / len(val_loader)
        dynamic_lr.step(avg_loss)
        
        current_acc = evaluation_tools.accuracy_score(accumulated_labels, accumulated_preds)
        
        if current_acc > top_accuracy_score:
            top_accuracy_score = current_acc
            optimized_params = copy.deepcopy(neural_net.state_dict())
            print(f"\n[Epoch {epoch_idx}] Performance Boost: Accuracy = {current_acc:.2%}")

    neural_net.load_state_dict(optimized_params)
    return neural_net

# --- Main Entry Point ---
if __name__ == "__main__":
    # 1. Setup
    model_instance = ClimateDiscourseNet(output_dimension=8).to(processing_unit)
    train_stream, val_stream, balancing_weights = fetch_and_encode_assets()

    # 2. Train
    final_model = execute_training_cycle(model_instance, train_stream, val_stream, balancing_weights)

    # 3. Export
    try:
        login(AUTH_TOKEN)
        final_model.save_pretrained("./climate_analysis_model")
        print("Success: Model weights exported to local directory.")
    except Exception as error:
        print(f"Export skipped or failed: {error}")