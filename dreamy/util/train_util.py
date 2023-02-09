
import torch


####################################################################

# loss and metrics

####################################################################

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

####################################################################

# train/val 

####################################################################

def train(epoch, model, training_loader, optimizer, return_losses=False, device="cuda"):
    Losses = []
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            Losses.append(loss.item())
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if return_losses:
        return Losses
    
def validation(model, testing_loader, device="cuda", return_inputs=False):
    model.eval()
    fin_targets = []
    fin_outputs = []
    fin_ids     = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            fin_ids.extend(ids.cpu().detach().numpy().tolist())
    if return_inputs:
        return fin_outputs, fin_targets, fin_ids
    else:
        return fin_outputs, fin_targets


####################################################################

# k-fold ssettings 

####################################################################

def get_Fold(final_df_dataset, tokenizer, k_seed, train_batch_size, valid_batch_size, max_length=512, train_size=0.8):
    
    train_dataset = final_df_dataset.sample(frac=train_size, random_state=k_seed)
    test_dataset  = final_df_dataset.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    training_set = CustomDataset(train_dataset, tokenizer, max_length=max_length)
    testing_set  = CustomDataset(test_dataset, tokenizer, max_length=max_length)

    train_params = {
        'batch_size': train_batch_size,
        'shuffle': True,
        'num_workers': 0
    }

    test_params = {
        'batch_size': valid_batch_size,
        'shuffle': True,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader  = DataLoader(testing_set, **test_params)
    
    return training_loader, testing_loader

def get_collection_Fold(final_df_dataset, collection, tokenizer, train_batch_size, valid_batch_size, max_length=512):
    
    train_dataset = final_df_dataset[~final_df_dataset["collection"].isin([collection])]
    test_dataset  = final_df_dataset[final_df_dataset["collection"].isin([collection])]

    train_dataset = train_dataset.reset_index(drop=True)
    test_dataset  = test_dataset.reset_index(drop=True)

    training_set = CustomDataset(train_dataset, tokenizer, max_length)
    testing_set  = CustomDataset(test_dataset, tokenizer, max_length)

    train_params = {
        'batch_size': train_batch_size,
        'shuffle': True,
        'num_workers': 0
    }

    test_params = {
        'batch_size': valid_batch_size,
        'shuffle': True,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader  = DataLoader(testing_set, **test_params)
    
    return training_loader, testing_loader