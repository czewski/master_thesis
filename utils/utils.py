import torch

def collate_fn(data, max_len):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lens = [len(sess) for sess, label in data]
    labels = []
    padded_sesss = torch.zeros(len(data), max_len).long() #max(lens)
    for i, (sess, label) in enumerate(data):
        padded_sesss[i,:lens[i]] = torch.LongTensor(sess)
        labels.append(label)
    
    padded_sesss = padded_sesss.transpose(0,1)
    return padded_sesss, torch.tensor(labels).long(), lens

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False