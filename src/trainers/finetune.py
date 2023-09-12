import torch


def train(model, optimizer, device, train_loader, criterion):
    model.train()
    train_loss, train_acc, train_total = 0, 0, 0
    for x, y in train_loader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(pred, 1)
        correct = predicted.eq(y).sum().item()
        train_loss += loss.item() * y.size(0)
        train_acc += correct
        train_total += y.size(0)

    return train_loss / train_total, train_acc / train_total


def eval(model, device, data_loader, criterion):
    model.eval()
    eval_loss, eval_acc, eval_total = 0, 0, 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        _, predicted = torch.max(pred, 1)
        correct = predicted.eq(y).sum().item()
        eval_loss += loss.item() * y.size(0)
        eval_acc += correct
        eval_total += y.size(0)

    return eval_loss / eval_total, eval_acc / eval_total


def ft_train(model, options, device, train_loader, test_loader):
    # Prepare model: just put it to device
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=options['ft_lr'], weight_decay=options['ft_wd'])

    results = torch.zeros((options['ft_epochs'], 4))  # train_loss, train_acc, test_loss, test_acc

    for epoch in range(options['ft_epochs']):
        train(model, optimizer, device, train_loader, criterion)
        # Get train stats
        results[epoch, 0], results[epoch, 1] = eval(model, device, train_loader, criterion)
        # Get test stats
        results[epoch, 2], results[epoch, 3] = eval(model, device, test_loader, criterion)

        if not options['noprint']:
            print(f"Epoch: {epoch + 1:03d}, Train_loss: {results[epoch, 0].item():.4f}, "
                  f"Train_acc: {results[epoch, 1].item():.4f}, "
                  f"Test_loss: {results[epoch, 2].item():.4f}, "
                  f"Test_acc: {results[epoch, 3].item():.4f}")

    print(">>> Fine-tuning done!")
    return results