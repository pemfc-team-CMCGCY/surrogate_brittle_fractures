# %load Simulacion_rama_matern.py
### Programa 5 repeticiones con matern rama

from dolfin import *
import numpy as np
from mshr import *
import matplotlib.pyplot as plt
from math import hypot
import gaussian_random_fields as gr 
import numpy
import timeit

start_time_T = timeit.default_timer()

#------------------------------------------------
#           Parameters
#------------------------------------------------

#mesh = Mesh('mesh.xml')

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

num_refine = 2
#h=0.5**num_refine
#for i in range(num_refine):
#    mesh=refine(mesh)
print("number of unknown",mesh.num_vertices())
print("number of elements",mesh.num_cells())



l_fac = 2 #1.5 # original 2

hm = mesh.hmin()
l_o = l_fac*hm
print(l_o, hm, l_fac)

#parametros = np.array([[121.15e3, 80.77e3]])

l = l_o
#lmbda = 1.94e3 #121.15e3
#mu = 2.45e3 #80.77e3              
    
def fractura(parametros, sigma_iter, n_par, n_rep): 
     
    ############################################################################# 
 
    ###                      segunda malla 
 
         
    ############################################################################# 
 
 
    ####### se interpola la primera malla con la segunda 
 
    start_time = timeit.default_timer()
    
 
    mesh2 = UnitSquareMesh(200, 200) 
 
    ## mover al centro (0,0) 
 
    mesh2.coordinates()[:, 0] = mesh2.coordinates()[:, 0] - 0.5*np.ones((len(mesh2.coordinates()[:, 0]))) 
    mesh2.coordinates()[:, 1] = mesh2.coordinates()[:, 1] - 0.5*np.ones((len(mesh2.coordinates()[:, 1]))) 
 
    V1 = FunctionSpace(mesh2, 'CG', 1) 
    u1 = interpolate(Expression("2.7", degree = 2), V1) 
    coordinates = mesh2.coordinates() 
                                               
    #plt.show() 
 
    ################################### 
 
    Gc_base = 2.7
    
    print(sigma_iter)

    nu = 1.5
    length = 0.03
    example = gr.matern_gaussian_random_field(band = 2, length = length, nu = nu, \
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
    
    ############################################## 
     
    num_par = int(n_par) 
    num_sim = int(n_rep)
     
    lmbda = parametros[0] 
    mu = parametros[1] 
     

    #################################



    # Define Space
    V = FunctionSpace(mesh, 'CG', 1)
    W = VectorFunctionSpace(mesh, 'CG', 1)
    WW = FunctionSpace(mesh, 'DG', 0)
    p, q = TrialFunction(V), TestFunction(V)
    u, v = TrialFunction(W), TestFunction(W)


    # Constituive functions
    
    def epsilon(u):
        return sym(grad(u))
    def sigma(u):
        return 2.0*mu*epsilon(u)+lmbda*tr(epsilon(u))*Identity(len(u))
    def psi(u):
        return 0.5*(lmbda+mu)*(0.5*(tr(epsilon(u))+abs(tr(epsilon(u)))))**2+\
               mu*inner(dev(epsilon(u)),dev(epsilon(u)))		
    def H(uold,unew,Hold):
        return conditional(lt(psi(uold),psi(unew)),psi(unew),Hold)
		

    class top(SubDomain):
        def inside(self,x,on_boundary):
            tol = 1e-10
            return abs(x[1]-0.5) < tol and on_boundary

    class bottom(SubDomain):
        def inside(self,x,on_boundary):
            tol = 1e-10
            return abs(x[1]+0.5) < tol and on_boundary
        
    class Middle(SubDomain): 
        def inside(self,x,on_boundary):
            tol = 1e-3
            return abs(x[1]) < tol and x[0] <= 0.0

    middle = Middle()  
    top = top()
    bot = bottom()

    load = Expression("t", t = 0.0, degree=1)

    bcbot= DirichletBC(W, Constant((0.0,0.0)), bot)
    bctopx = DirichletBC(W.sub(1), Constant(0.0), top)
    bctopy = DirichletBC(W.sub(0), load, top)

    bc_u = [bcbot, bctopx, bctopy]
    bc_phi = [DirichletBC(V, Constant(1.0), middle)]

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    top.mark(boundaries,1)
    ds = Measure("ds")(subdomain_data=boundaries)
    n = FacetNormal(mesh)

    # Variational form
    unew, uold = Function(W), Function(W)
    pnew, pold, Hold = Function(V), Function(V), Function(V)

    E_du = ((1.0-pold)**2 + 0.0*1e-6)*inner(grad(v),sigma(u))*dx
    E_phi = (Gc*l*inner(grad(p),grad(q))+((Gc/l)+2.0*H(uold,unew,Hold))\
            *inner(p,q)-2.0*H(uold,unew,Hold)*q)*dx

    p_disp = LinearVariationalProblem(lhs(E_du), rhs(E_du), unew, bc_u)
    p_phi = LinearVariationalProblem(lhs(E_phi), rhs(E_phi), pnew, bc_phi)
    solver_disp = LinearVariationalSolver(p_disp)
    solver_phi = LinearVariationalSolver(p_phi)


    # Initialization of the iterative procedure and output requests 
    
    K= 5
    t = 0 
    u_r = 0.022 
    deltaT  = 1e-4 #1e-4 #5e-4 #1e-4
    tol = 1e-3 

    conc_Gc = File ('Files/HPFM_cr/Gc_' + str(num_par)  + '_' + str(num_sim)  +'.pvd')  
    conc_Gc << Gc 
    conc_f = File ('Files/HPFM_cr/phi_'  + str(num_par)  + '_' + str(num_sim)  + '.pvd') 
    conc_f1 = File ('Files/HPFM_cr/u_'  + str(num_par)  + '_' + str(num_sim)  + '.pvd') 
    fname = open('Files/HPFM_cr/Tension_y_'  + str(num_par)  + '_' + str(num_sim)  + '.txt', 'w') 
    fname2 = open('Files/HPFM_cr/Tension_x_'  + str(num_par)  + '_' + str(num_sim)  + '.txt', 'w')
                

    # Staggered scheme 
    while t<=u_r: 
        t += deltaT 
       # if t >= 0.013: 
       #      deltaT = 5e-4 
                
        load.t=t 
     
        iter = 0 
        err = 1 
 
        while err > tol: 
            iter += 1 
            solver_disp.solve() 
            solver_phi.solve() 
            err_u = errornorm(unew,uold,norm_type = 'l2',mesh = None) 
            err_phi = errornorm(pnew,pold,norm_type = 'l2',mesh = None) 
            err = max(err_u,err_phi) 
         
            uold.assign(unew) 
            pold.assign(pnew) 
            Hold.assign(project(psi(unew), WW)) 
 
            if err < tol: 
 
                print ('Iterations:', iter, ', Total time', t) 
 
                if round(t*1e5) % 50 == 0: 
                    conc_f << pnew 
                    conc_f1 << unew
 
                Traction = dot(sigma(unew),n) 
                fy = Traction[1]*ds(1) 
                fname.write(str(t) + "\t") 
                fname.write(str(assemble(fy)) + "\n") 
            
                fx = Traction[0]*ds(1) 
                fname2.write(str(t) + "\t") 
                fname2.write(str(assemble(fx)) + "\n") 
            
            
      #t +=deltaT      
    
    fname.close()
    fname2.close()

    print('Simulation Done with no error :)')

    elapsed = timeit.default_timer() - start_time

    print('el tiempo de ejecución es:', elapsed, 'segundos')
    
    return elapsed
                    
def execution(N): 
    
    # N repetitions
    # M index to increase s = G_c/sigma
     
    vals = [] 
    vals = np.array([[121.15e3, 80.77e3]]) # 
    S_ = np.array([100])
    N_s = len(S_) 
     
    N_iter = int(N)  
    
    for cont in range(N_s):
        
        fname3 = open('Files/HPFM_cr/Time_'  + str(cont)  + '.txt', 'w')   
                    
        for cont2 in range(N_iter):
                      
            tiempo = fractura(vals[0], S_[cont], cont, cont2) 
                      
            fname3.write(str(tiempo) + "\n") 
                      
        fname3.close()
        
         
    print("Fin") 
     
############################################################################ 
 
# Executions

execution(1) 
 
elapsed_T = timeit.default_timer() - start_time_T 
 
print('el tiempo de ejecución es:', elapsed_T, 'segundos') 
 
