import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    """
    A simple CNN Encoder built from scratch to extract features from images.
    """

    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(256, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.conv_blocks(images)
        features = self.avgpool(features).view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    """
    RNN Decoder (LSTM) to generate a sequence of attribute tokens.
    """

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        return self.linear(hiddens)


class ImageToTextModel(nn.Module):
    """
    A wrapper model that combines the Encoder and Decoder.
    """

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageToTextModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)

    @torch.no_grad()
    def calculate_log_likelihood(self, images, captions, pad_idx):
        """
        Calculates the total log-likelihood of a given sequence for an image.
        """
        self.eval()
        logits = self.forward(images, captions)
        log_probs = F.log_softmax(logits, dim=-1)
        targets = captions[:, 1:]
        target_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)
        mask = targets != pad_idx
        return (target_log_probs * mask).sum(dim=1)

    @torch.no_grad()
    def predict(self, images, dataset, max_length=10):
        """
        Generates attribute descriptions for a batch of images.
        """
        self.eval()
        batch_size = images.size(0)
        features = self.encoder(images)
        inputs = torch.full(
            (batch_size, 1),
            dataset.token_to_idx["[SOS]"],
            dtype=torch.long,
            device=images.device,
        )

        predicted_sequences = []
        _, states = self.decoder.lstm(features.unsqueeze(1))

        for _ in range(max_length):
            embedded = self.decoder.embed(inputs)
            hiddens, states = self.decoder.lstm(embedded, states)
            outputs = self.decoder.linear(hiddens.squeeze(1))
            predicted_ids = outputs.argmax(dim=1).unsqueeze(1)
            predicted_sequences.append(predicted_ids)
            inputs = predicted_ids

        sequences = torch.cat(predicted_sequences, dim=1)

        output_strings = []
        for i in range(batch_size):
            raw_tokens = sequences[i].cpu().numpy()
            try:
                eos_idx = list(raw_tokens).index(dataset.token_to_idx["[EOS]"])
                tokens = raw_tokens[:eos_idx]
            except ValueError:
                tokens = raw_tokens
            output_strings.append(dataset.detokenize(tokens))

        return output_strings
