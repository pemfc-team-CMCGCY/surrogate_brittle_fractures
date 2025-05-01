# Updated by Luis Blanco-Cocom (2024)
########## matern simulation

from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import sys, os, sympy, shutil, math
import gaussian_random_fields as gr 
from math import hypot
# Define 2D geometry
# Define 2D geometry

import timeit
start_time = timeit.default_timer()

xmin = 0.0
ymin = 0.0

xmax = 65
ymax = 120

xcenter = 36.5
ycenter = 51.0

xcenter1 = 20.0
ycenter1 = 20.0

xcenter2 = 20.0
ycenter2 = 100.0

radius = 10
radius1 = 5

domain = Rectangle(Point(xmin,ymin),Point(xmax,ymax)) - Circle(Point(xcenter,ycenter),radius,25) - \
Circle(Point(xcenter1,ycenter1),radius1,25) - Circle(Point(xcenter2,ycenter2),radius1,25)
domain.set_subdomain(1, Rectangle(Point(0., 0.), Point(10., 65.)))
domain.set_subdomain(2, Rectangle(Point(0., 65.), Point(10., 120.)))

mesh = generate_mesh(domain,45) #50 funciona l = 2.5, 2.20 en 50 Du
plt.figure(figsize=(8, 8))
plot(mesh)
plt.show()

mesh2 = UnitSquareMesh(500, 500) # Se escala al rectangulo
 
## mover al centro (0,0) 
 
mesh2.coordinates()[:, 0] = mesh2.coordinates()[:, 0]*120# - 0.5*np.ones((len(mesh2.coordinates()[:, 0]))) 
mesh2.coordinates()[:, 1] = mesh2.coordinates()[:, 1]*120# - 0.5*np.ones((len(mesh2.coordinates()[:, 1]))) 
 
V1 = FunctionSpace(mesh2, 'CG', 1) 
u1 = interpolate(Expression("2.7", degree = 2), V1) 
coordinates = mesh2.coordinates() 

################################### 
 
################################### 

nu1 = 1.5
length = 0.03 
Gc_base = 2.7
sigma_iter = 1  
print(sigma_iter)

nu = 1.5
length = 0.03
example = gr.matern_gaussian_random_field(band = 2, length = length, nu = nu1, \
                                              size = int(mesh2.num_vertices()**0.5), flag_normalize = True)

im0 = plt.imshow(example) 
plt.colorbar(im0) 

plt.show() 

ma = np.max(example)
mi = np.min(example)
max1 = np.max([ma, np.abs(mi)])
example = example/max1
    
n, m = example.shape 
 
random_field_vector = np.zeros(n*m) 
 
k=0 
for kkk in range(m): 
    for iii in range(kkk+1): 
        random_field_vector[k] = Gc_base  + (Gc_base/sigma_iter)*example[kkk -iii, iii] 
        k = k + 1 
          
for kkk in range(m-1): 
    for iii in range((m-1)- (kkk)): 
        random_field_vector[k] = Gc_base  + (Gc_base/sigma_iter)*example[(m-1) -  iii, kkk + iii+1] 
        k = k + 1 
        
u1.vector()[:] = random_field_vector 
 
####### Interpolate 
    
V = FunctionSpace(mesh2, 'CG', 1) 
Gc_1 = interpolate(u1, V) 
 
    #print(len(Gc.vector()), len(u1.vector())) 
    #plot(mesh) 
    #plt.show()  
 
    #im=plot(u1) 
    #plt.colorbar(im) 
    #plt.show() 
    
im=plot(Gc_1) 
plt.colorbar(im) 
plt.show() 

# interpolar al rectangulo

VVV = FunctionSpace(mesh, 'CG', 1) 
Gc = interpolate(Gc_1, VVV) 

plt.figure(figsize=(3, 4.5))
im=plot(Gc) 
plt.colorbar(im) 
plt.savefig('comparative_concrete_M.eps', format='eps')
plt.show()
