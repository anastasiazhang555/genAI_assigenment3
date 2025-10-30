import torch
# Assignment 2
def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=5):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(data_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

    print("✅ Training completed!")
    return model

# ===  GAN TRAINING FUNCTION (Assignment3) ===
def train_gan(models, data_loader, device='cpu', epochs=10, lr=0.0002):
    import torch
    import torch.nn as nn

    G, D = models["generator"].to(device), models["discriminator"].to(device)
    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for imgs, _ in data_loader:
            imgs = imgs.to(device)
            batch_size = imgs.size(0)

            # --- Train Discriminator ---
            z = torch.randn(batch_size, 100, device=device)
            fake_imgs = G(z)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            D_real = D(imgs)
            D_fake = D(fake_imgs.detach())
            loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --- Train Generator ---
            D_fake = D(fake_imgs)
            loss_G = criterion(D_fake, real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch [{epoch+1}/{epochs}] | D_loss: {loss_D.item():.4f} | G_loss: {loss_G.item():.4f}")

    return G, D