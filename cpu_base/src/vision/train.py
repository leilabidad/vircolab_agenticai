def train_swin(model, loader, optimizer, criterion, device, epochs):
    model.train()

    for ep in range(epochs):
        total = 0
        for img_fr, img_la, y, _, _ in loader:
            img_fr = img_fr.to(device)
            img_la = img_la.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred, _ = model(img_fr, img_la)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"[Swin] Epoch {ep+1}: loss={total/len(loader):.4f}")
