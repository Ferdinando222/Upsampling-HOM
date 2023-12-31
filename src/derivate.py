# %%

import torch
from torch.autograd import grad

# Definisci la tua funzione
def f(x,y,z,f):
    return (x**2 +y**2+z**2)*f



# Calcola la prima derivata
x = torch.rand(1202, requires_grad=True)
y = torch.rand(1202, requires_grad=True)
z = torch.rand(1202, requires_grad=True)
fr = torch.rand(512, requires_grad=True)

x_extended = x.view(-1, 1)  # Trasforma x in un tensore colonna
y_extended = y.view(-1, 1)
z_extended = z.view(-1, 1)
y = f(x_extended,y_extended,z_extended,fr)
du_dx = grad(y.sum(), x, create_graph=True)[0]
du_dy= grad(y.sum(), x, create_graph=True)[0]
du_dz = grad(y.sum(), x, create_graph=True)[0]

# Calcola la seconda derivata
du_dxx= grad(du_dx.sum(), x)[0]
du_dyy= grad(du_dy.sum(), x)[0]
du_dzz= grad(du_dz.sum(), x)[0]

laplace = du_dxx+du_dyy+du_dzz

#helmholtz
pde_residual = laplace +fr@y.t()

# Stampa il risultato
print("Prima derivata:", du_dx,du_dy,du_dz)
print("Laplace:", laplace)
print("Pde residual", pde_residual)
# %%

# %%

import torch
from torch.autograd import grad

# Definisci la tua funzione
def f(x,y,z,f):
    return (x**2 +y**2+z**2)*f



# Calcola la prima derivata
x = torch.rand(1202, requires_grad=True)
y = torch.rand(1202, requires_grad=True)
z = torch.rand(1202, requires_grad=True)
fr = torch.rand(512, requires_grad=True)

x_extended = x.view(-1, 1)  # Trasforma x in un tensore colonna
y_extended = y.view(-1, 1)
z_extended = z.view(-1, 1)
y = f(x_extended,y_extended,z_extended,fr)
du_dx = grad(y.sum(), x, create_graph=True)[0]
du_dy= grad(y.sum(), x, create_graph=True)[0]
du_dz = grad(y.sum(), x, create_graph=True)[0]

# Calcola la seconda derivata
du_dxx= grad(du_dx.sum(), x)[0]
du_dyy= grad(du_dy.sum(), x)[0]
du_dzz= grad(du_dz.sum(), x)[0]

laplace = du_dxx+du_dyy+du_dzz

#helmholtz
pde_residual = laplace +fr@y.t()

laplace = torch.zeros_like(y)
for i in range (y.shape[1]):
    y_i = y[:,i].sum()
    du_dx = grad(y_i, x, create_graph=True)[0]
    du_dy= grad(y_i, x, create_graph=True)[0]
    du_dz = grad(y_i, x, create_graph=True)[0]

    # Calcola la seconda derivata
    du_dxx= grad(du_dx.sum(), x)[0]
    du_dyy= grad(du_dy.sum(), x)[0]
    du_dzz= grad(du_dz.sum(), x)[0]

    laplace[:,i] = du_dxx+du_dyy+du_dzz


pde_residual = laplace+fr*y
# Stampa il risultato
print("Prima derivata:", du_dx,du_dy,du_dz)
print("Laplace:", laplace)
print("Pde residual", pde_residual)
# %%

