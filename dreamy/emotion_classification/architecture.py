
import numpy as np
import torch
import transformers 

class BERTClass(torch.nn.Module):
    def __init__(self, model_name, n_classes, freeze_BERT=False, layer=-1, idx=0):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(model_name)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(1024, n_classes)
        self.layer = layer
        self.idx   = idx  
        # Froze the weight of model aside of the classifier
        if freeze_BERT:
            print("Freezing the layer of BERT model")
            for name, param in self.l1.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
                    
    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, output_hidden_states=True)
        last_hidden_states = output_1.hidden_states[self.layer]
        CLS = last_hidden_states[:,self.idx,:]        
        output_2 = self.l2(CLS)
        output   = self.l3(output_2)
        return output

class BERT_PTM(transformers.PreTrainedModel):
    def __init__(self, config, model_name, n_classes, freeze_BERT=False, layer=-1, idx=0):
        super(BERT_PTM, self).__init__(config)
        self.l1 = transformers.BertModel.from_pretrained(model_name)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(1024, n_classes)
        self.layer = layer
        self.idx   = idx  
        # Froze the weight of model aside of the classifier
        if freeze_BERT:
            print("Freezing the layer of BERT model")
            for name, param in self.l1.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
                    
    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, output_hidden_states=True)
        last_hidden_states = output_1.hidden_states[self.layer]
        CLS = last_hidden_states[:,self.idx,:]         
        output_2 = self.l2(CLS)
        output   = self.l3(output_2)
        return output
