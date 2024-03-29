
"""Пример кода с обучением CRNN, чтобы убедиться, что все работает. Для простоты учим на одной картинке."""
from src.model import CRNN
from src.generator.dataset_generator import BarcodeDataset
import torch

criterion = torch.nn.CTCLoss(zero_infinity=True)

img_size = (512, 280)
vocab = ' 0123456789'
max_length = 15
batch_size = 8
epochs = 25

crnn = CRNN(
    cnn_backbone_name='resnet18d',
    cnn_backbone_pretrained=True,
    cnn_output_size=8960,
    rnn_features_num=128,
    rnn_dropout=0.1,
    rnn_bidirectional=True,
    rnn_num_layers=2,
    num_classes=len(vocab) + 1,
)

barcode_loader = torch.utils.data.DataLoader(
    BarcodeDataset(epoch_size=100, vocab=vocab, max_length=max_length, img_size=img_size),
    batch_size=batch_size,
)

optimizer = torch.optim.Adam(crnn.parameters(), lr=10e-3)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    crnn.to(device)
    crnn = crnn.train()

    for epoch in range(epochs):
        for input_images, targets, target_lengths in barcode_loader:
            input_images = input_images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            output = crnn(input_images)
            input_lengths = [output.size(0) for _ in input_images]
            input_lengths = torch.LongTensor(input_lengths)
            target_lengths = torch.flatten(target_lengths)

            loss = criterion(output, targets, input_lengths, target_lengths)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f'epoch {epoch}\tloss {loss.item()}')  # noqa: WPS421

            torch.save(crnn.state_dict(), f'./weights/generator_state_dict_{epochs}.pth')
            torch.save(crnn, f'./weights/generator_model_{epochs}_fix.pth')
