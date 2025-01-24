"""
Simulates two-group isotropically scattering 
neutron transport eigenvalue problem in
a 2-D heterogeneous reactor geometry
using diffusion approximation

NOTE: Origin locationed at Bottom Left of Geometry
"""

#imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D


#fxns
def get_sl():
    """Returns sidelength of one reactor unit"""
    sl = 23.1226 # [cm]
    return sl

def get_L():
    """Returns sidelength of the entire geometry"""
    sl = get_sl()
    L = 17.0 * sl
    return L
    
def get_h():
    """Returns width of spatial cell (NOTE: width = height)"""
    sl = get_sl()
    h = sl # [cm] h = sl
    #h = sl / 2.0 # [cm] h = sl / 2.0
    #h = sl / 4.0 # [cm] h = sl / 4.0
    return h

def get_J():
    """Returns number of spatial cells in x and y directions"""
    h = get_h()
    L = get_L()
    J = L  / h
    return int(J)

def get_GS_tol():
    """Returns tolerance for Gauss-Seidel iterative method"""
    GS_tol = 1E-6
    return GS_tol

def get_k_tol():
    """Returns convergence criterion for k value"""
    k_tol = 1E-6
    return k_tol

def get_flux_tol():
    """Returns convergence criterion for flux value"""
    flux_tol = 1E-5
    return flux_tol

def grid():
    """Returns x and y coordinates for each flux"""
    J = get_J()+1
    h = get_h()
    x = np.zeros((J,J))
    y = np.zeros((J,J))
    for j in range(J):
        for i in range(J):
            x[j][i] = i * h
            y[j][i] = j * h
    return x, y

def geometry_builder():
    """Returns matrix containing reactor materials and geometry"""
    geom = np.zeros((17,17))
    geom[0] = [0,0,0,0,3,3,3,3,3,3,3,3,3,0,0,0,0]
    geom[1] = [0,0,3,3,3,4,4,4,4,4,4,4,3,3,3,0,0]
    geom[2] = [0,3,3,4,4,8,1,1,1,1,1,8,4,4,3,3,0]
    geom[3] = [0,3,4,4,5,1,7,1,7,1,7,1,5,4,4,3,0]
    geom[4] = [3,3,4,5,2,8,2,8,1,8,2,8,2,5,4,3,3]
    geom[5] = [3,4,8,1,8,2,8,2,6,2,8,2,8,1,8,4,3]
    geom[6] = [3,4,1,7,2,8,1,8,2,8,1,8,2,7,1,4,3]
    geom[7] = [3,4,1,1,8,2,8,1,8,1,8,2,8,1,1,4,3]
    geom[8] = [3,4,1,7,1,6,2,8,1,8,2,6,1,7,1,4,3]
    geom[9] = [3,4,1,1,8,2,8,1,8,1,8,2,8,1,1,4,3]
    geom[10] = [3,4,1,7,2,8,1,8,2,8,1,8,2,7,1,4,3]
    geom[11] = [3,4,8,1,8,2,8,2,6,2,8,2,8,1,8,4,3]
    geom[12] = [3,3,4,5,2,8,2,8,1,8,2,8,2,5,4,3,3]
    geom[13] = [0,3,4,4,5,1,7,1,7,1,7,1,5,4,4,3,0]
    geom[14] = [0,3,3,4,4,8,1,1,1,1,1,8,4,4,3,3,0]
    geom[15] = [0,0,3,3,3,4,4,4,4,4,4,4,3,3,3,0,0]
    geom[16] = [0,0,0,0,3,3,3,3,3,3,3,3,3,0,0,0,0]
    return geom

def geom_fm(scale):
    geom = geometry_builder()
    original_size = geom.shape
    refined_size = tuple(s * scale for s in original_size)
    refined_geom = np.zeros(refined_size, dtype=geom.dtype)
    for j in range(original_size[0]):
        for i in range(original_size[1]):
            # Define the range in the refined matrix that corresponds to the current cell
            refined_row_start = j * scale
            refined_row_end = (j + 1) * scale
            refined_col_start = i * scale
            refined_col_end = (i + 1) * scale

            # Fill in the cells in the refined matrix
            refined_geom[refined_row_start:refined_row_end, refined_col_start:refined_col_end] = geom[j, i]
    return refined_geom

def material(x, y, geom):
    """Returns material # at given (x,y) coordinates in geometry"""
    sl = get_sl()
    x_unit = int(x / sl)
    y_unit = len(geom) - 1 - int(y / sl)
    if x_unit == len(geom):
        x_unit = len(geom) - 1
    if y_unit == len(geom):
        y_unit = len(geom) - 1
    return(int(geom[y_unit][x_unit]))

def material_mod(i, j, geom):
    """Returns material # at given (x,y) coordinates for modified geometry"""
    h = get_h()
    x = i * h
    y = j * h
    return material(x, y, geom)

def get_D():
    """Returns diffusion coefficient matrix"""
    D = [[0.0, 1.436, 1.4366, 1.32, 1.4389, 1.4381, 1.4385, 1.4389, 1.4393],
        [0.0, 0.3635, 0.3636, 0.2772, 0.3638, 0.3665, 0.3665, 0.3679, 0.3680]] # [cm^-1]
    return D

def get_sigma_r():
    """Returns macroscopic removal cross-section matrix"""
    sigma_r = [[0.0, 0.0272582, 0.0272995, 0.0257622, 0.0274640, 0.0272930, 0.0273240, 0.0272900, 0.0273210],
        [0.0, 0.075058, 0.078436, 0.071596, 0.091408, 0.084828, 0.087314, 0.088024, 0.090510]] # [cm^-1]
    return sigma_r

def get_sigma_Ds():
    """Returns macroscopic downscattering (group 1 to group 2) cross section array"""
    sigma_Ds = [0.0, 0.017754, 0.017621, 0.023106, 0.017101, 0.017290, 0.017192, 0.017125, 0.017027] # [cm^-1]
    return sigma_Ds

def get_sigma_Us():
    """Returns macroscopic upscattering (group 2 to group 1) cross section"""
    sigma_Us = 0.0 # [cm^-1]
    return sigma_Us

def get_nu_sigf():
    """Returns matrix containing product of nu and macroscopic fission cross section"""
    nu_sigf = [[0.0, 0.0058708, 0.0061908, 0.0, 0.0074527, 0.0061908, 0.0064285, 0.0061908, 0.0064285],
        [0.0, 0.096067, 0.103580, 0.0, 0.132360, 0.103580, 0.109110, 0.103580, 0.109110]] # [cm^-1]
    return nu_sigf
    
def fission_builder_G1(group):
    """Returns fission source matrix for group 1 (group option == 2 returns init. fission source for G2"""
    J = get_J()+1
    M_f_arr = []
    h = get_h()
    geom = geometry_builder()
    nu_sigf = get_nu_sigf()[group-1]
    for j in range(J):
        for i in range(J):
            mat1 = material((i-1.0) * h, (j-1.0) * h, geom)   #i-1, j-1
            sig1 = nu_sigf[mat1]
            mat2 = material((i-1.0) * h, j * h, geom)   #i-1, j
            sig2 = nu_sigf[mat2]
            mat3 = material(i * h, (j-1.0) * h, geom)   #i, j-1
            sig3 = nu_sigf[mat3]
            mat4 = material(i * h, j * h, geom)   #i, j
            sig4 = nu_sigf[mat4]
            if mat1==0 or mat2==0 or mat3==0 or mat4==0:
                M_f_arr.append(0.0)
            else:
                M_f_arr.append( ( (h**2) / (4.0) ) * (sig1 + sig2 + sig3 + sig4))
    arr = np.array(M_f_arr)
    M_f_diag = np.diag(arr) # converts to diagonal matrix
    return M_f_diag

def fission_builder_G2(phi_g1):
    """Returns fission source matrix for group 2, given flux from group 1"""
    J = get_J()+1
    M_f_arr = []
    M_s_arr = []
    h = get_h()
    geom = geometry_builder()
    nu_sigf = get_nu_sigf()
    sigma_Ds = get_sigma_Ds()
    for j in range(J):
        for i in range(J):
            mat1 = material((i-1.0) * h, (j-1.0) * h, geom)   #i-1, j-1
            nu_sigf1 = nu_sigf[1][mat1]
            sigma_Ds1 = sigma_Ds[mat1]
            mat2 = material((i-1.0) * h, j * h, geom)   #i-1, j
            nu_sigf2 = nu_sigf[1][mat2]
            sigma_Ds2 = sigma_Ds[mat2]
            mat3 = material(i * h, (j-1.0) * h, geom)   #i, j-1
            nu_sigf3 = nu_sigf[1][mat3]
            sigma_Ds3 = sigma_Ds[mat3]
            mat4 = material(i * h, j * h, geom)   #i, j
            nu_sigf4 = nu_sigf[1][mat4]
            sigma_Ds4 = sigma_Ds[mat4]
            if mat1==0 or mat2==0 or mat3==0 or mat4==0:
                M_f_arr.append(0.0)
                M_s_arr.append(0.0)
            else:
                M_f_arr.append( ( (h**2) / (4.0) ) * (nu_sigf1 + nu_sigf2 + nu_sigf3 + nu_sigf4))
                M_s_arr.append( ( (h**2) / (4.0) ) * (sigma_Ds1 + sigma_Ds2 + sigma_Ds3 + sigma_Ds4))
    arr = (np.array(M_f_arr) + (np.array(M_s_arr)*phi_g1))
    M_f_diag = np.diag(arr) # converts to diagonal matrix
    return M_f_diag

def diffusion_builder(group):
    """Returns diffusion matrix given energy group"""
    J = get_J()+1
    M_d = np.zeros((J*J,J*J))
    h = get_h()
    geom = geometry_builder()
    D = get_D()[group-1]
    sigma_r = get_sigma_r()[group-1]
    for j in range(J):
        for i in range(J):
            n = (j * J) + i
            x = i * h
            y = j * h
            mat = material(x, y, geom)
            if mat == 0:
                M_d[n][n] = 1.0
            elif i==0 and j==0:
                mat4 = material(i * h, j * h, geom)   #i, j
                sigma_r_ij = sigma_r[mat4]
                M_d[n][n] = ((h**2) * sigma_r_ij) + D[mat4]
            elif i==J-1 and j==0:
                mat2 = material((i-1.0) * h, j * h, geom)   #i-1, j
                sigma_r_ij = sigma_r[mat2]
                M_d[n][n] = ((h**2) * sigma_r_ij) + D[mat2]
            elif i==J-1 and j==J-1:
                mat1 = material((i-1.0) * h, (j-1.0) * h, geom)   #i-1, j-1
                sigma_r_ij = sigma_r[mat1]
                M_d[n][n] = ((h**2) * sigma_r_ij) + D[mat1]
            elif i==0 and j==J-1:
                mat3 = material(i * h, (j-1.0) * h, geom)   #i, j-1
                sigma_r_ij = sigma_r[mat3]
                M_d[n][n] = ((h**2) * sigma_r_ij) + D[mat3]
            elif i==0:
                mat3 = material(i * h, (j-1.0) * h, geom)   #i, j-1
                mat4 = material(i * h, j * h, geom)   #i, j
                sigma_r_ij = (0.5) * (sigma_r[mat3] + sigma_r[mat4])
                M_d[n][n+1] = (-0.5) * (D[mat3] + D[mat4])
                M_d[n][n] = ((h**2) * sigma_r_ij) + (0.5 * (D[mat3] + D[mat4]))
            elif i==J-1:
                mat1 = material((i-1.0) * h, (j-1.0) * h, geom)   #i-1, j-1
                mat2 = material((i-1.0) * h, j * h, geom)   #i-1, j
                sigma_r_ij = (0.5) * (sigma_r[mat1] + sigma_r[mat2])
                M_d[n][n-1] = (-0.5) * (D[mat1] + D[mat2])
                M_d[n][n] = ((h**2) * sigma_r_ij) + (0.5 * (D[mat1] + D[mat2]))
            elif j==0:
                mat2 = material((i-1.0) * h, j * h, geom)   #i-1, j
                mat4 = material(i * h, j * h, geom)   #i, j
                sigma_r_ij = (0.5) * (sigma_r[mat2] +sigma_r[mat4])
                M_d[n][n+J] = (-0.5) * (D[mat2] + D[mat4])
                M_d[n][n] = ((h**2) * sigma_r_ij) + (0.5 * (D[mat2] + D[mat4]))
            elif j==J-1:
                mat1 = material((i-1.0) * h, (j-1.0) * h, geom)   #i-1, j-1
                mat3 = material(i * h, (j-1.0) * h, geom)   #i, j-1
                sigma_r_ij = (0.5) * (sigma_r[mat1] +sigma_r[mat3])
                M_d[n][n-J] = (-0.5) * (D[mat1] + D[mat3])
                M_d[n][n] = ((h**2) * sigma_r_ij) + (0.5 * (D[mat1] + D[mat3]))
            else:
                mat1 = material((i-1.0) * h, (j-1.0) * h, geom)   #i-1, j-1
                mat2 = material((i-1.0) * h, j * h, geom)   #i-1, j
                mat3 = material(i * h, (j-1.0) * h, geom)   #i, j-1
                mat4 = material(i * h, j * h, geom)   #i, j
                sigma_r_ij = (0.25) * (sigma_r[mat1] + sigma_r[mat2] + sigma_r[mat3] + sigma_r[mat4])
                M_d[n][n-J] = (-0.5) * (D[mat1] + D[mat3])
                M_d[n][n-1] = (-0.5) * (D[mat1] + D[mat2])
                M_d[n][n+1] = (-0.5) * (D[mat3] + D[mat4])
                M_d[n][n+J] = (-0.5) * (D[mat2] + D[mat4])
                M_d[n][n] = ((h**2) * sigma_r_ij) + ((0.5 * (D[mat1] + D[mat3])) + (0.5 * (D[mat1] + D[mat2])) + (0.5 * (D[mat3] + D[mat4])) + (0.5 * (D[mat2] + D[mat4])))
    return M_d

def corner_check(j, i, phi_m, group):
    """Determines if a location is on a corner and returns leakage"""
    h = get_h()
    geom = geometry_builder()
    """
    J = get_J()
    zeros_col = np.zeros((J,1))
    geom = np.append(zeros_col, geom, axis=1)
    geom = np.append(geom, zeros_col, axis=1)
    zeros_row = np.zeros((1,J+2))
    geom = np.append(zeros_row, geom, axis=0)
    geom = np.append(geom, zeros_row, axis=0)
    """
    D = get_D()[group-1]
    d = 2.0 * np.array(D)
    d[d == 0.0] = 1.0E-54
    ret = False, 0.0
    if (phi_m[j][i-1]==0 and phi_m[j-1][i]==0): # top left corner
        mat = material_mod(i, (j-1), geom)
        ret = True, 2.0 * (phi_m[j][i] * (-1.0/d[mat]) * h * (D[mat]))
    elif (phi_m[j][i+1]==0 and phi_m[j-1][i]==0): # top right corner
        mat = material_mod((i-1), (j-1), geom)
        ret = True, 2.0 * (phi_m[j][i] * (-1.0/d[mat]) * h * (D[mat]))
    elif (phi_m[j][i-1]==0 and phi_m[j+1][i]==0): # bottom left corner
        mat = material_mod((i), (j), geom)
        ret = True, 2.0 * (phi_m[j][i] * (-1.0/d[mat]) * h * (D[mat]))
    elif (phi_m[j][i+1]==0 and phi_m[j+1][i]==0): # bottom right corner
        mat = material_mod((i-1), (j), geom)
        ret = True, 2.0 * (phi_m[j][i] * (-1.0/d[mat]) * h * (D[mat]))
    return ret
    
def side_check(j, i, phi_m, group):
    """Determines if a location is on a side and returns leakage"""
    h = get_h()
    geom = geometry_builder()
    """
    J = get_J()
    zeros_col = np.zeros((J,1))
    geom = np.append(zeros_col, geom, axis=1)
    geom = np.append(geom, zeros_col, axis=1)
    zeros_row = np.zeros((1,J+2))
    geom = np.append(zeros_row, geom, axis=0)
    geom = np.append(geom, zeros_row, axis=0)
    """
    D = get_D()[group-1]
    d = 2.0 * np.array(D) # extrapolation length
    d[d == 0.0] = 1.0E-54
    ret = False, 0.0
    if (phi_m[j-1][i]==0): # top side
        mat1 = material_mod(i, (j-1), geom)
        mat2 = material_mod((i-1), (j-1), geom)
        ret = True, phi_m[j][i] * h * -1.0 * ((D[mat1] + D[mat2]) / d[mat1] + d[mat2]) 
    elif (phi_m[j+1][i]==0): # bottom side
        mat1 = material_mod(i, j, geom)
        mat2 = material_mod((i-1), j, geom)
        ret = True, phi_m[j][i] * h * -1.0 * ((D[mat1] + D[mat2]) / d[mat1] + d[mat2]) 
    elif (phi_m[j][i-1]==0): # left side
        mat1 = material_mod(i, (j-1), geom)
        mat2 = material_mod(i, j, geom)
        ret = True, phi_m[j][i] * h * -1.0 * ((D[mat1] + D[mat2]) / d[mat1] + d[mat2]) 
    elif (phi_m[j][i+1]==0): # right side
        mat1 = material_mod((i-1), (j-1), geom)
        mat2 = material_mod((i-1), j, geom)
        ret = True, phi_m[j][i] * h * -1.0 * ((D[mat1] + D[mat2]) / d[mat1] + d[mat2]) 
    return ret

def leakage_solver(phi, group):
    """Returns leakage distribution for given flux array"""
    J = get_J()+1
    phi_m = np.flipud(phi).reshape((J,J)) # matrix of fluxes with (i,j)=(0,0) at top left of geometry
    zeros_col = np.zeros((J,1))
    phi_m = np.append(zeros_col, phi_m, axis=1)
    phi_m = np.append(phi_m, zeros_col, axis=1)
    zeros_row = np.zeros((1,J+2))
    phi_m = np.append(zeros_row, phi_m, axis=0)
    phi_m = np.append(phi_m, zeros_row, axis=0)
    leakage = np.zeros_like(phi.reshape(J,J))
    for j in range(1, J):
        for i in range(1, J):
            if phi_m[j][i] > 0.0:
                corner, c_leak = corner_check(j, i, phi_m, group)
                if corner == True:
                    leakage[j][i] = c_leak
                else:
                    side, s_leak = side_check(j, i, phi_m, group)
                    if side == True:
                        leakage[j][i] = s_leak
    #leakage = np.delete(leakage, 0, 0)
    #leakage = np.delete(leakage, -1, 0)
    #leakage = np.delete(leakage, 0, 1)
    #leakage = np.delete(leakage, -1, 1)
    return leakage

def gauss_seidel(A, b):
    """Solves linear system of equations Ax=b via Gauss-Seidel method"""
    J = len(A[0])
    D = np.zeros((J,J))
    U = np.zeros((J,J))
    L = np.zeros((J,J))
    x_new = np.ones(J)
    x_old = np.zeros(J)
    temp = max(abs(x_new - x_old))
    tol = get_GS_tol()
    for row in range(J):
        for col in range(J):
            if row==col:
                D[row][col] = A[row][col]
            if row == (col+1):
                L[row][col] = A[row][col]
            if row == (col-1):
                U[row][col] = A[row][col]
    while (temp >= tol):
        x_new = np.linalg.inv(D + L) @ (b - U @ x_old)
        temp = max(abs(x_new - x_old))
        x_old = x_new.copy()
    return x_new 

def diffusion_solver():
    """Solves eigenvalue fission problem via diffusion theory"""
    k = [1.0] # list used for variable length, 1 is initial guess
    k_tol = get_k_tol()
    flux_tol = get_flux_tol()
    J = get_J()+1
    M_d_G1 = diffusion_builder(1)
    M_d_G2 = diffusion_builder(2)
    M_f_G1 = fission_builder_G1(1)
    G2_fis = fission_builder_G1(2) # initial distribution of fission neutrons for G2
    phi_last_G1 = np.zeros(J*J)
    phi_last_G1.fill(1.0/sum(np.diag(M_f_G1))) # initial (n=0) guess for flux G1
    phi_last_G2 = np.zeros(J*J)
    phi_last_G2.fill(1.0/sum(np.diag(G2_fis))) # initial (n=0) guess for flux G2
    phi_int_G1 = np.zeros(J*J) # current flux, before normalization G1
    phi_curr_G1 = np.zeros(J*J) # current flux, after normalization G1
    phi_int_G2 = np.zeros(J*J) # current flux, before normalization G2
    phi_curr_G2 = np.zeros(J*J) # current flux, after normalization G2
    converged = False
    i = 0 
    while(converged == False):
        print("i:",i,"k:",k[-1])
        b_G1 = (1.0 / k[-1]) * (M_f_G1 @ phi_last_G1)
        phi_int_G1 = gauss_seidel(M_d_G1, b_G1)
        M_f_G2 = fission_builder_G2(phi_int_G1)
        b_G2 = (1.0 / k[-1]) * (M_f_G2 @ phi_last_G2)
        phi_int_G2 = gauss_seidel(M_d_G2, b_G2)
        k.append(k[-1] * ((np.diag(M_f_G1) @ phi_int_G1) + (np.diag(M_f_G2) @ phi_int_G2)))
        phi_curr_G1 = (k[-2] / k[-1]) * phi_int_G1
        phi_curr_G2 = (k[-2] / k[-1]) * phi_int_G2
        phi_curr_G1[phi_curr_G1 == float('inf')] = 0.0
        phi_curr_G2[phi_curr_G2 == float('inf')] = 0.0
        if ((k[-1] - k[-2]) < k_tol) and (max(abs(phi_curr_G2 - phi_last_G2)) < flux_tol) and (max(abs(phi_curr_G1 - phi_last_G1)) < flux_tol):
            converged = True
        phi_last_G1 = phi_curr_G1
        phi_last_G2 = phi_curr_G2
        i += 1
    k.pop(0)
    return max(k), len(k), phi_curr_G1, phi_curr_G2

def plotter(dist, title, leak):
    """Plots given distribution"""
    J = get_J()
    X, Y = grid()
    dist_matrix = dist.reshape(J+1, J+1)
    # color mapping for different materials
    #colors = np.flipud(geometry_builder()).astype('object')
    scale = J // 17
    colors = geom_fm(scale).astype('object')    
    color_map = {0: 'w', 1: 'y', 2: 'm', 3: 'r', 4: 'c', 5: 'k', 6: 'b', 7: 'mediumseagreen', 8: 'g'}
    for key, color in color_map.items():
        colors[colors == key] = color

    # plot for group 1
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf1 = ax.plot_surface(X, Y, dist_matrix, facecolors=colors, linewidth=0)
    ax.set_zlim(0.0, max(max(dist),0.0005))
    if leak:
        ax.set_zlim(min(min(dist),-0.0005), 0.0)
    ax.zaxis.set_major_locator(LinearLocator(6))
    ax.set_title(title)
    title = title.replace(" ", "_")
    plt.savefig(title+'_plot.png')    
    plt.show()
    
def legend():
    """Plots the legend"""
    # color mapping for different materials
    # colors = np.flipud(geometry_builder()).astype('object')
    J = get_J()
    scale = J // 17
    colors = geom_fm(scale).astype('object')
    color_map = {0: 'w', 1: 'y', 2: 'm', 3: 'r', 4: 'c', 5: 'k', 6: 'b', 7: 'mediumseagreen', 8: 'g'}
    legend_elements = [Patch(facecolor=color, label=str(key)) for key, color in color_map.items() if key != 0]
    for key, color in color_map.items():
        colors[colors == key] = color
        
    # plotting
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.legend(handles=legend_elements, title="Materials", loc="center")
    plt.show()

def writer(dist, file_name):
    """Outputs given distribution matrix to text file"""
    J = get_J()+1
    f = open(file_name, "w")
    for j in range(J):
        f.write('[ ')
        for i in range(J):
            f.write('{:.4e}, '.format(dist[j][i]))
        f.write(' ]\n')
    f.close()

def output():
    """Runs solver and plots distribution"""
    k, i, phi_g1, phi_g2 = diffusion_solver()
    print("k_eff:",k, "iterations:",i)
    f = open('output.txt','w')
    f.write("k_eff:"+str(k)+"iterations:"+str(i))
    f.close()
    leak_g1 = np.array(leakage_solver(phi_g1, 1))
    leak_g2 = np.array(leakage_solver(phi_g2, 2))
    plotter(phi_g1, 'Group 1 Neutron Flux', False)
    plotter(phi_g2, 'Group 2 Neutron Flux', False)
    plotter(leak_g1.flatten(), 'Group 1 Leakage', True)
    plotter(leak_g2.flatten(), 'Group 2 Leakage', True)
    legend()
    
#main
M_f = fission_builder_G1(1)
M_d = diffusion_builder(1)
plt.spy(M_d)
plt.title("Diffusion Coefficient Matrix")
plt.show()
