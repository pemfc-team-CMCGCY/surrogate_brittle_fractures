from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import numpy
import sys, os, sympy, shutil, math
import gaussian_random_fields as gr 


#set_log_active(False) 
#================================== 
# mesh (concident nodes) for the crack 
#=================================== 


nx = 1 
ny = 1 
num = 1 
mesh = Mesh() 
editor = MeshEditor() 
editor.open(mesh, "triangle", 2, 2) 
editor.init_vertices(10) 
editor.init_cells(8) 
editor.add_vertex(0, Point(0.0, 0.0)) 
editor.add_vertex(1,  Point(0.5, 0.0)) 
editor.add_vertex(2,  Point(0.5, 0.5)) 
editor.add_vertex(3,  Point(0.0, 0.5)) 
editor.add_vertex(4,  Point(-0.5, 0.5)) 
editor.add_vertex(5,  Point(-0.5, 0.0)) 
editor.add_vertex(6,  Point(-0.5, -0.5)) 
editor.add_vertex(7,  Point(0.0, -0.5)) 
editor.add_vertex(8,  Point(0.5, -0.5)) 
editor.add_vertex(9,  Point(-0.5, 0.0)) 
editor.add_cell(0, np.array([0, 1, 3],dtype=np.uintp)) 
editor.add_cell(1, np.array([1, 2, 3],dtype=np.uintp)) 
editor.add_cell(2, np.array([0, 3, 4],dtype=np.uintp)) 
editor.add_cell(3, np.array([0, 4, 5],dtype=np.uintp)) 
editor.add_cell(4, np.array([0, 9, 7],dtype=np.uintp)) 
editor.add_cell(5, np.array([6, 7, 9],dtype=np.uintp))      
editor.add_cell(6, np.array([0, 7, 8],dtype=np.uintp)) 
editor.add_cell(7, np.array([0, 8, 1],dtype=np.uintp)) 
editor.close() 

#-------------------------------------- 
# change num_refine for mesh refinement 
#-------------------------------------- 


num_refine = 6 
h=0.5**num_refine 
for i in range(num_refine): 
    mesh=refine(mesh) 
print("number of unknown",mesh.num_vertices()) 
print("number of elements",mesh.num_cells()) 

#plot(mesh) 
#plt.show() #interactive()  

#------------------------------------------------ 
#           Parameters 
#------------------------------------------------ 

l_fac = 1.5 # original 2 
#Gc =  2.7 
hm = mesh.hmin() 
l_o = l_fac*hm # 0.015
print(l_o, hm, l_fac) 

ndim = mesh.topology().dim() 

L = 1.0; T = 0.5; H = 1.0; L1 = -0.5; 
left = CompiledSubDomain("near(x[1], %s, 1e-4)"%L1)
right = CompiledSubDomain("near(x[1], %s, 1e-4)"%T)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0) #FacetFunction("size_t", mesh)
boundaries.set_all(0)
left.mark(boundaries, 1) # mark left abajo as 1
right.mark(boundaries, 2) # mark right arriba as 2
ds = Measure("ds",subdomain_data=boundaries) # left: ds(1), right: ds(2)


E, nu = Constant(210.0e3), Constant(0.3)

#Gc = Constant(2.7)

ell = l_o #Constant(0.005)

def w(alpha):
    """Dissipated energy function as a function of the damage """
    return alpha

def a(alpha):
    """Stiffness modulation as a function of the damage """
    k_ell = Constant(1.e-6) # residual stiffness
    return (1-alpha)**2+k_ell

def eps(u):
    """Strain tensor as a function of the displacement"""
    return sym(grad(u))

def sigma_0(u):
    """Stress tensor of the undamaged material as a function of the displacement"""   
    
    lmbda = 121.15e3 
    mu = 80.77e3 
    
    return 2.0*mu*(eps(u)) + lmbda*tr(eps(u))*Identity(ndim)

def sigma(u,alpha):
    """Stress tensor of the damaged material as a function of the displacement and the damage"""
    return (a(alpha))*sigma_0(u)



# Create function space for 2D elasticity + Damage

V_u = VectorFunctionSpace(mesh, "P", 1)
V_alpha = FunctionSpace(mesh, "P", 1)

# Define the function, test and trial fields

u, du, v = Function(V_u), TrialFunction(V_u), TestFunction(V_u)
alpha, dalpha, beta = Function(V_alpha), TrialFunction(V_alpha), TestFunction(V_alpha)
u.rename('displacement','displacement')
u.rename('damage','damage')

##### Random field G_c 

############################################################################# 
 
####### Interpolate 
 
mesh2 = UnitSquareMesh(200, 200) 
 
## mover al centro (0,0) 
 
mesh2.coordinates()[:, 0] = mesh2.coordinates()[:, 0] - 0.5*np.ones((len(mesh2.coordinates()[:, 0]))) 
mesh2.coordinates()[:, 1] = mesh2.coordinates()[:, 1] - 0.5*np.ones((len(mesh2.coordinates()[:, 1]))) 
 
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
    
     #Normalizado

ma = numpy.max(example)
mi = numpy.min(example)
max1 = numpy.max([ma, numpy.abs(mi)])
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
    
V = FunctionSpace(mesh, 'CG', 1) 
Gc = interpolate(u1, V) 
    
im=plot(Gc) 
plt.colorbar(im) 
plt.show() 
 

######


z = sympy.Symbol("z")
a0 = sympy.symbols('a')
b0 = sympy.symbols('b')
a0 = 0
b0 = 1
c_w0 = 4*sympy.integrate(sympy.sqrt(w(z)),(z,a0,b0))
c_w = float(c_w0)
print("c_w = ",c_w)
c_1w0 = sympy.integrate(sympy.sqrt(1/w(z)),(z,a0,b0))
c_1w = float(c_1w0)
print("c_1/w = ",c_1w)
tmp = 2*(sympy.diff(w(z),z)/sympy.diff(1/a(z),z)).subs({"z":0})
sigma_c = sympy.sqrt(tmp*E/(c_w*ell))
#print("sigma_c =", sigma_c*math.sqrt(Gc))
eps_c = float(sigma_c*math.sqrt(np.mean(Gc.vector()))/E)
#print("eps_c = %2.3f"%eps_c)


elastic_energy = 0.5*inner(sigma(u,alpha), eps(u))*dx
dissipated_energy = Gc/float(c_w)*(w(alpha)/ell + ell*dot(grad(alpha), grad(alpha)))*dx
total_energy = elastic_energy + dissipated_energy 

# First directional derivative wrt u
E_u = derivative(total_energy,u,v)

# First and second directional derivative wrt alpha
E_alpha = derivative(total_energy,alpha,beta)
E_alpha_alpha = derivative(E_alpha,alpha,dalpha)

# Displacement

u_R = Expression(("t","0"),t = 0.,degree=0)
bcu_0 = DirichletBC(V_u, u_R, boundaries, 2)
bcu_1 = DirichletBC(V_u, Constant((0.,0.)), boundaries, 1)
bc_u = [bcu_1, bcu_0]

# Damage

class Middle(SubDomain):  
    def inside(self,x,on_boundary): 
        tol = 1e-3 
        return abs(x[1]) < tol and x[0] <= 0.0 

middle = Middle()
bc_alpha = [DirichletBC(V_alpha,Constant(1.0),middle)]

from ufl import replace

E_du = replace(E_u,{u:du})
problem_u = LinearVariationalProblem(lhs(E_du), rhs(E_du), u, bc_u)
solver_u = LinearVariationalSolver(problem_u)
solver_u.parameters.update({"linear_solver" : "umfpack"})

class DamageProblem(OptimisationProblem):

    def f(self, x):
        """Function to be minimised"""
        alpha.vector()[:] = x
        return assemble(total_energy)

    def F(self, b, x):
        """Gradient (first derivative)"""
        alpha.vector()[:] = x
        assemble(E_alpha, b)

    def J(self, A, x):
        """Hessian (second derivative)"""
        alpha.vector()[:] = x
        assemble(E_alpha_alpha, A)

solver_alpha_tao = PETScTAOSolver()
solver_alpha_tao.parameters.update({"method": "tron","linear_solver" : "umfpack", 
                                    "line_search": "gpcg", "report": True})

lb = interpolate(Constant("0."), V_alpha) # lower bound, initialize to 0
ub = interpolate(Constant("1."), V_alpha) # upper bound, set to 1

for bc in bc_alpha:
    bc.apply(lb.vector())
    bc.apply(ub.vector())
    
def alternate_minimization(u,alpha,tol=1.e-5,maxiter=100,alpha_0=interpolate(Constant("0.0"), V_alpha)):
    
    # initialization
    iter = 1; err_alpha = 1
    alpha_error = Function(V_alpha)
    
    # iteration loop
    while err_alpha>tol and iter<maxiter:
        
        # solve elastic problem
        solver_u.solve()
        
        # solve damage problem
        solver_alpha_tao.solve(DamageProblem(), alpha.vector(), lb.vector(), ub.vector())# test error
        alpha_error.vector()[:] = alpha.vector() - alpha_0.vector()
        err_alpha = np.linalg.norm(alpha_error.vector(), ord = np.Inf)

        alpha_0.assign(alpha)
        
        iter=iter+1
        
    return (err_alpha, iter)

def postprocessing():

    # Save number of iterations for the time step
    
    iterations[i_t] = np.array([t,i_t])
    # Calculate the energies
    
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)
    energies[i_t] = np.array([t,elastic_energy_value,surface_energy_value,elastic_energy_value+surface_energy_value])
    
    # Calculate the axial force resultant
    forces[i_t] = np.array([t,assemble(sigma(u,alpha)[1,0]*ds(2))]) ### arriba
    
    # Dump solution to file
    print(t, i_t+1)
    file_alpha << (alpha,t)
    file_u << (u,t)
    
    # Save some global quantities as a function of the time
    np.savetxt(savedir+'/energies.txt', energies)
    np.savetxt(savedir+'/forces.txt', forces)
    np.savetxt(savedir+'/iterations.txt', iterations)
    
def critical_stress():
    xs = sympy.Symbol('x')
    wx = w(xs); sx = 1/(E*H*a(xs));
    res = sympy.sqrt(2*wx.diff(xs)/(sx.diff(xs)*ell))*((np.mean(Gc.vector())*H/c_w)**0.5)
    return res.evalf(subs={xs:0})

def plot_stress():
    plt.plot(forces[:,0], forces[:,1], 'b*', linewidth = 2)
    plt.xlabel('Displacement')
    plt.ylabel('Force')
    force_cr = critical_stress()

def plot_energy():
    p1, = plt.plot(energies[:,0], energies[:,1],'b*',linewidth=2)
    p2, = plt.plot(energies[:,0], energies[:,2],'r^',linewidth=2)
    p3, = plt.plot(energies[:,0], energies[:,3],'ko',linewidth=2)
    plt.legend([p1, p2, p3], ["Elastic","Dissipated","Total"])
    plt.xlabel('Displacement')
    plt.ylabel('Energies')
    force_cr = critical_stress()
    plt.axvline(x = force_cr/(E*H)*L, color = 'grey',linestyle = '--', linewidth = 2)
    plt.axhline(y = H,color = 'grey', linestyle = '--', linewidth = 2)

def plot_energy_stress():
    plt.subplot(211)
    plot_stress()
    plt.subplot(212)
    plot_energy()
    plt.savefig(savedir+'/energies_force.png')
    plt.show()
    
import timeit

start_time = timeit.default_timer()


loads = 1e-3*np.linspace(0,20,20)

ii = 4

savedir = "Files/results_cr/Iter_" + str(ii) + "/" 

if os.path.isdir(savedir):
    shutil.rmtree(savedir)    
file_alpha = File(savedir+"/alpha.pvd") 
file_u = File(savedir+"/u.pvd") 

#  loading and initialization of vectors to store time datas

energies = np.zeros((len(loads),4))
iterations = np.zeros((len(loads),2))
forces = np.zeros((len(loads),2))

lb.interpolate(Constant(0.))

for (i_t, t) in enumerate(loads):
    u_R.t = t
    
    # solve alternate minimization
    alternate_minimization(u,alpha,maxiter=30)
    
    # updating the lower bound to account for the irreversibility
    lb.vector()[:] = alpha.vector()
    postprocessing()


print('Simulation Done with no error :)')

elapsed = timeit.default_timer() - start_time

print('el tiempo de ejecución es:', elapsed, 'segundos')
    
plt.figure(0)
plot(u,mode='displacement')
plt.figure(1)

plot(alpha)
plt.show()

plot(u)
plt.show()
                           
                        
plot_energy_stress()
