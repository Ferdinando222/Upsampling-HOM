#%%
import torch

# Crea una matrice M di forma [1202, 512] e un vettore v di forma [1202]

x = torch.tensor([1.0,2.0,3.0,4.0], requires_grad=True)
y = torch.tensor([1.0,2.0,3.0,4.0], requires_grad=True)
z = torch.tensor([1.0,2.0,3.0,4.0], requires_grad=True)

f = torch.tensor([1.0,2.0,3.0], requires_grad=True)

x_v = x.view(-1, 1) * torch.ones(1,3)
y_v = y.view(-1, 1) * torch.ones(1, 3)
z_v = z.view(-1, 1) * torch.ones(1, 3)
f_r = f.repeat(4,1)

u = 3*x_v**3*f_r +2*y_v**3*f_r+4*z_v**3*f_r

#%%
# Definisci una funzione che utilizza M e v. Ad esempio, potrebbe essere il prodotto punto tra M e v.

# Utilizza torch.autograd.grad per calcolare il gradiente di output rispetto a M

grad_x,grad_y,grad_z = [],[],[]
grad_xx,grad_yy,grad_zz = [],[],[]
laplace = []

pde_res = torch.empty(u.shape[0],u.shape[1],dtype=torch.cfloat)

# Calcolo dei gradienti
for i in  range(u.shape[0]):
    x_i,y_i,z_i = x[i],y[i],z[i]
    u_i = 3*x_i**3*f +2*y_i**3*f+4*z_i**3*f
    grads = torch.autograd.grad(u_i, (x_i,y_i,z_i), create_graph=True,grad_outputs=torch.ones_like(u_i))


    # Estrazione dei gradienti
    d_x, d_y, d_z = grads

    d_xx = torch.autograd.grad(d_x,x_i,retain_graph=True,grad_outputs=torch.ones_like(d_x))[0]
    d_yy = torch.autograd.grad(d_y,y_i,retain_graph=True,grad_outputs=torch.ones_like(d_y))[0]
    d_zz = torch.autograd.grad(d_z,z_i,retain_graph=True,grad_outputs=torch.ones_like(d_z))[0]
    grad_x.append(d_x)
    grad_y.append(d_y)
    grad_z.append(d_z)
    grad_xx.append(d_xx)
    grad_yy.append(d_yy)
    grad_zz.append(d_zz)
    laplace.append(d_xx+d_yy+d_zz)
    k = torch.tensor((2*torch.pi*f/342),dtype=torch.cfloat)
    #pde_res[i,:] = torch.abs((d_xx+d_yy+d_zz)+ k**2 *u_i)**2

#%%
grad_x,grad_y,grad_z= torch.autograd.grad(u, (x, y, z), create_graph=True,grad_outputs=torch.ones_like(u))
# %%

grad_x,grad_y,grad_z= torch.autograd.grad(u, (x_v, y_v, z_v), create_graph=True,grad_outputs=torch.ones_like(u))
norm_pde = torch.mean(pde_res,dim=1)
loss = torch.mean(norm_pde)

# %% caso 2

d_x = torch.autograd.grad(u,x,create_graph=True,grad_outputs=torch.ones_like(u))[0]
d_y = torch.autograd.grad(u,y,create_graph=True,grad_outputs=torch.ones_like(u))[0]
d_z = torch.autograd.grad(u,z,create_graph=True,grad_outputs=torch.ones_like(u))[0]

# %%

d_xx = torch.autograd.grad(d_x,x,retain_graph=True,grad_outputs=torch.ones_like(d_x))[0]
d_yy = torch.autograd.grad(d_y,y,retain_graph=True,grad_outputs=torch.ones_like(d_y))[0]
d_zz = torch.autograd.grad(d_z,z,retain_graph=True,grad_outputs=torch.ones_like(d_z))[0]

laplace_2 = d_xx+d_yy+d_zz
pde_res  = laplace_2+((f/342)**2).t()@u.t()
loss_2 = torch.mean(torch.abs(pde_res)**2)
# %%
import numpy as np
k = torch.randn(4, 1)  # Assicurati che k sia [513, 1]
u = torch.randn(5, 4)

# Calcola il quadrato di k
k_squared = k ** 2

# Moltiplica k_squared (utilizzando il broadcasting) per u
result = np.dot(k_squared.t(),u.t())
# %%
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

# Calcola il prodotto riga-per-colonna tra A e B
product = np.dot(A, B.T)

# Stampa il risultato
print(product)
# %%

a = torch.tensor(3.0)
b = torch.tensor(4.0)
z = torch.complex(a, b)

# Calcola la norma L2 dello scalare complesso
norm_z = torch.abs(z)

# %%
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [7, 8, 9]])

B = np.array([[9, 8, 7]])

C  = B*A

# %%

#%%
import torch

# Crea una matrice M di forma [1202, 512] e un vettore v di forma [1202]

x = torch.tensor([1.0,2.0,3.0,4.0], requires_grad=True)
y = torch.tensor([1.0,2.0,3.0,4.0], requires_grad=True)
z = torch.tensor([1.0,2.0,3.0,4.0], requires_grad=True)

f = torch.tensor([1.0,2.0,3.0], requires_grad=True)

x_v = x.view(-1, 1) * torch.ones(1,3)
y_v = y.view(-1, 1) * torch.ones(1, 3)
z_v = z.view(-1, 1) * torch.ones(1, 3)
f_r = f.repeat(4,1)


l = x.repeat(3,1)

# %%
