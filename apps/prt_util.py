import os
import trimesh
import numpy as np
import math
from scipy.special import sph_harm
import argparse
from tqdm import tqdm
import gc

def factratio(N, D):
    if N >= D:
        prod = 1.0
        for i in range(D+1, N+1):
            prod *= i
        return prod
    else:
        prod = 1.0
        for i in range(N+1, D+1):
            prod *= i
        return 1.0 / prod

def KVal(M, L):
    return math.sqrt(((2 * L + 1) / (4 * math.pi)) * (factratio(L - M, L + M)))

def AssociatedLegendre(M, L, x):
    if M < 0 or M > L or np.max(np.abs(x)) > 1.0:
        return np.zeros_like(x)
    
    pmm = np.ones_like(x)
    if M > 0:
        somx2 = np.sqrt((1.0 + x) * (1.0 - x))
        fact = 1.0
        for i in range(1, M+1):
            pmm = -pmm * fact * somx2
            fact = fact + 2
    
    if L == M:
        return pmm
    else:
        pmmp1 = x * (2 * M + 1) * pmm
        if L == M+1:
            return pmmp1
        else:
            pll = np.zeros_like(x)
            for i in range(M+2, L+1):
                pll = (x * (2 * i - 1) * pmmp1 - (i + M - 1) * pmm) / (i - M)
                pmm = pmmp1
                pmmp1 = pll
            return pll

def SphericalHarmonic(M, L, theta, phi):
    if M > 0:
        return math.sqrt(2.0) * KVal(M, L) * np.cos(M * phi) * AssociatedLegendre(M, L, np.cos(theta))
    elif M < 0:
        return math.sqrt(2.0) * KVal(-M, L) * np.sin(-M * phi) * AssociatedLegendre(-M, L, np.cos(theta))
    else:
        return KVal(0, L) * AssociatedLegendre(0, L, np.cos(theta))

def save_obj(mesh_path, verts):
    file = open(mesh_path, 'w')    
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    file.close()

def sampleSphericalDirections(n):
    xv = np.random.rand(n,n)
    yv = np.random.rand(n,n)
    theta = np.arccos(1-2 * xv)
    phi = 2.0 * math.pi * yv

    phi = phi.reshape(-1)
    theta = theta.reshape(-1)

    vx = -np.sin(theta) * np.cos(phi)
    vy = -np.sin(theta) * np.sin(phi)
    vz = np.cos(theta)
    return np.stack([vx, vy, vz], 1), phi, theta

def getSHCoeffs(order, phi, theta):
    shs = []
    for n in range(0, order+1):
        for m in range(-n,n+1):
            s = SphericalHarmonic(m, n, theta, phi)
            shs.append(s)
    
    return np.stack(shs, 1)

def computePRT(mesh_path, n, order):
    mesh = trimesh.load(mesh_path, process=False)
    vectors_orig, phi, theta = sampleSphericalDirections(n)
    SH_orig = getSHCoeffs(order, phi, theta)

    w = 4.0 * math.pi / (n*n)

    origins = mesh.vertices
    normals = mesh.vertex_normals
    n_v = origins.shape[0]

    # Giảm kích thước batch xuống 100
    batch_size = 100
    n_batches = (n_v + batch_size - 1) // batch_size
    
    # Khởi tạo PRT_all với float32 để tiết kiệm bộ nhớ
    PRT_all = np.zeros((n_v, SH_orig.shape[1]), dtype=np.float32)
    
    for i in tqdm(range(n)):
        SH = SH_orig[(i*n):((i+1)*n)]
        vectors = vectors_orig[(i*n):((i+1)*n)]
        
        for b in range(n_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, n_v)
            
            batch_origins = origins[start_idx:end_idx]
            batch_normals = normals[start_idx:end_idx]
            
            # Tính toán cho batch hiện tại
            batch_origins = np.repeat(batch_origins[:,None], n, axis=1).reshape(-1,3)
            batch_normals = np.repeat(batch_normals[:,None], n, axis=1).reshape(-1,3)
            batch_vectors = np.repeat(vectors[None,:], end_idx-start_idx, axis=0).reshape(-1,3)
            batch_SH = np.repeat(SH[None,:], end_idx-start_idx, axis=0).reshape(-1,SH.shape[1])

            dots = (batch_vectors * batch_normals).sum(1)
            front = (dots > 0.0)

            delta = 1e-3*min(mesh.bounding_box.extents)
            hits = mesh.ray.intersects_any(batch_origins + delta * batch_normals, batch_vectors)
            nohits = np.logical_and(front, np.logical_not(hits))

            PRT = (nohits.astype(np.float32) * dots)[:,None] * batch_SH
            PRT_all[start_idx:end_idx] += PRT.reshape(-1, n, SH.shape[1]).sum(1)
            
            # Giải phóng bộ nhớ
            del batch_origins, batch_normals, batch_vectors, batch_SH, dots, front, hits, nohits, PRT
            gc.collect()

    PRT = w * PRT_all
    return PRT, mesh.faces

def testPRT(dir_path, n=40):
    if dir_path[-1] == '/':
        dir_path = dir_path[:-1]
    sub_name = dir_path.split('/')[-1][:-4]
    obj_path = os.path.join(dir_path, sub_name + '_100k.obj')
    os.makedirs(os.path.join(dir_path, 'bounce'), exist_ok=True)

    PRT, F = computePRT(obj_path, n, 2)
    np.savetxt(os.path.join(dir_path, 'bounce', 'bounce0.txt'), PRT, fmt='%.8f')
    np.save(os.path.join(dir_path, 'bounce', 'face.npy'), F)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/home/shunsuke/Downloads/rp_dennis_posed_004_OBJ')
    parser.add_argument('-n', '--n_sample', type=int, default=40, help='squared root of number of sampling. the higher, the more accurate, but slower')
    args = parser.parse_args()

    testPRT(args.input)
