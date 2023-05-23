
import torch


def ft_train(model,optimizer,device,train_loader, criterion):
    # model.train()
    for x,y in train_loader:
        optimizer.zero_grad()
        x,y = x.to(device), y.to(device)
        pred = model(x)
        zipped = zip(pred,y)
        loss = 0
        for i in zipped:
            loss += criterion(i[0], i[1])
        
        loss.backward()
        optimizer.step()
        del pred

def ft_eval(model,device,data_loader, criterion):
    model.eval()
    acc, loss, total = 0, 0, 0
    for x,y in data_loader:
        x,y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        _, predicted = torch.max(pred, 1)
        correct = predicted.eq(y).sum().item()
        target_size = y.size(0)
        loss += loss.item() * y.size(0)
        acc += correct
        total += target_size

        del pred
        
    total_loss = loss/total
    total_acc = acc/total
    
    return total_loss, total_acc



def ft_trn_main(model, options, device,train_loader, test_loader, criterion):
    # Prepare model: just put it to device
    model = model.to(device)

    # Prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=options['finetune_lr'], weight_decay=options['finetune_wd']) 



    # training loops
    results = torch.zeros((options['finetune_epochs'],4)) # train_loss, train_acc, test_loss, test_acc

    for epoch in range(options['finetune_epochs']):
        # Train 1 epoch
        ft_train(model,optimizer,device,train_loader, criterion)
        
        # Get train stats
        results[epoch,0], results[epoch,1] = ft_eval(model,device,train_loader,criterion)
        
        # Get test stats
        results[epoch,2], results[epoch,3] = ft_eval(model,device,test_loader, criterion)
        
        print(f"Epoch:{epoch+1:03d}, Trn_loss:{results[epoch,0].item():.4f}, Trn_acc:{results[epoch,1].item():.4f}, Tst_loss:{results[epoch,2].item():.4f}, Tst_acc:{results[epoch,3].item():.4f}")

    print("Finetune done")