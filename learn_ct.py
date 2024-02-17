import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from torch import nn
T=1
# Define the SDE





# Load data
import numpy as np
latent=np.load("latent.npy")
latent=latent.reshape(latent.shape[0],-1)
latent_size=latent.shape[1]
latent=torch.tensor(latent).float()
N=100

    

class ConsistencyModel(nn.Module):
    def __init__(self, latent_size):
        super(ConsistencyModel, self).__init__()
        self.latent_size = latent_size
        self.model = nn.Sequential(nn.Linear(latent_size+1, 100),nn.Tanh(),nn.Linear(100, 100),nn.Tanh(),nn.Linear(100, 100),nn.Tanh(),nn.Linear(100, 100),nn.Tanh(),nn.Linear(100, 100),nn.Tanh(),nn.Linear(100, 100),nn.Tanh(),nn.Linear(100, self.latent_size))
        self.T=T
        self.t=torch.linspace(0,1,N)
    
    def forward(self, x, t):
        if t.ndim<2:
            print(x.shape)
            t=torch.tensor(t)
            t=t.unsqueeze(0)
            t=t.unsqueeze(0)
            t=t.repeat(x.shape[0],1)
        tmp=torch.cat((x,t),1)
        return x+t*self.model(tmp)
    
    def sample(self,n_samples):
        z = torch.randn(n_samples, self.latent_size)
        t=self.t[-1].reshape(1,1).repeat(n_samples,1)
        return self.forward(z,t)


dataset=torch.utils.data.TensorDataset(latent)
loader=torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)



model=ConsistencyModel(latent_size)
model_minus = ConsistencyModel(latent_size)
model_minus.load_state_dict(model.state_dict())

# Train model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 100
loss_list = []
for epoch in range(num_epochs):
    for i, x in enumerate(loader):
        optimizer.zero_grad()
        x0=x[0]
        index=torch.randint(0,N-1,(x0.shape[0],))
        t=model.t[index].unsqueeze(1)
        t_plus=model.t[index+1].unsqueeze(1)
        z=torch.randn(x0.shape[0],latent_size)
        x_t_plus=model(x0+t_plus*z,t_plus)        
        x_t=model_minus(x0+t*z,t) 
        loss=torch.mean(torch.linalg.norm(x_t_plus-x_t,dim=1))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mu = math.exp(2 * math.log(0.95) / N)
            # update \theta_{-}
            for p, p_minus in zip(model.parameters(), model_minus.parameters()):
                p_minus.mul_(mu).add_(p, alpha=1 - mu)


        if i % 10 == 0:
            print('Epoch: {}/{}, Iter: {}/{}, Loss: {:.3f}'.format(
            epoch+1, num_epochs, i+1, len(loader), loss.item()))
    if loss.item()<100:
        loss_list.append(loss.item())
plt.plot(np.arange(len(loss_list)),loss_list)
plt.show()
torch.save(model, "ct.pt")
