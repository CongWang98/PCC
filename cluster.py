# -*- coding: utf-8 -*-
# @Time    : 2021/01/24
# @Author  : Cong Wang
# @Github ：https://github.com/CongWang98
import os
import math
import torch
import rmsd
import argparse
import numpy as np
from tqdm import tqdm
from model import FCAE, AEparameter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.preprocessing as preprocessing
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from preprocessing import LoadLmpsFile, LoadAngDihFile, LoadLatentFile, CalAngle, CalDihedral, get_dis, get_abs, get_ang, SampleData, findCA


def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('-ds', '--dataset', default='example')
    ap.add_argument('-mt', '--modelname_time', default=None)
    ap.add_argument('-cc', '--clusters_count', default=10, type=int)
    ap.add_argument('-sr', '--samplerate', default=0.1, type=float)
    #ap.add_argument('-ld', '--latentdim', default=10, type=int)
    #ap.add_argument('-ids', '--inter_dims', nargs='+', type=int, default=[1000, 1000, 1000])    
    return ap.parse_args()


def GetCoor(a, b, c, angle, dihedral, bondlength):
    """
    The sequance is a-b-c-d. Give a angle(0-1, unit:pi), a dihedral(0-1, unit:2*pi) and the coordinate(unit: A) of a, b, c,
    return the coordinate of d.
    """
    ang = angle * math.pi
    dih = dihedral * 2 * math.pi
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    n1 = np.cross(ba, bc)
    cb_yan = bc / get_abs(bc) * (get_abs(ba) * math.cos(get_ang(a, b, c) * math.pi))
    ha = ba - cb_yan
    bc_yan = - bc / get_abs(bc) * (bondlength * math.cos(ang))
    hd = n1 / get_abs(n1) * (bondlength * math.sin(ang) * math.sin(dih))
    hh = ha / get_abs(ha) * (bondlength * math.sin(ang) * math.cos(dih))
    return c + bc_yan + hh + hd


def CalBondLength(coor_frame):
    """
    Give a frame of CGed trajectory, return the 'bond' length.
    """
    atom_lis = coor_frame.reshape(-1, 3)
    length_lis = []
    for i in range(len(atom_lis) - 1):
        atom1 = atom_lis[i]
        atom2 = atom_lis[i + 1]
        bond_length = (((atom2 - atom1) ** 2).sum()) ** (1 / 2)
        length_lis.append(bond_length)
    return np.array(length_lis)


def CalAveBondLength(coorlis):
    """
    Give a coorlis, return the average bond length list.
    """
    bond_lis_lis = np.array([CalBondLength(i) for i in coorlis])
    return bond_lis_lis.mean(0)


def AngdihToCoor(angdihlis, bondlength_lis):
    """
    Convert to a angdihlis to coordination list.
    """
    atom_num = int((len(angdihlis) + 5) / 2)
    if atom_num != len(bondlength_lis) + 1:
        raise ValueError('Inconsistant atom number')
    coor_lis = []
    angle = angdihlis[0]
    a = [0, 0, 0]
    b = [0, 0, bondlength_lis[0]]
    c = [0, bondlength_lis[1] * math.sin(angle * math.pi), bondlength_lis[0] - bondlength_lis[1] * math.cos(angle * math.pi)]
    coor_lis = np.array(coor_lis + a + b + c)
    for i in range(atom_num - 3):
        dihedral, angle = angdihlis[2 * i + 1], angdihlis[2 * i + 2]
        a, b, c = coor_lis[3 * i: 3 * i + 3], coor_lis[3 * i + 3: 3 * i + 6], coor_lis[3 * i + 6: 3 * i + 9]
        bondlength = bondlength_lis[i + 2]
        d = GetCoor(a, b, c, angle, dihedral, bondlength)
        coor_lis = np.concatenate((coor_lis, d), axis=0)
    return np.array(coor_lis)


def GenePdbString(coor_frame, CAlis=None):
    """
    Give a coor_frame , return pdb file string.
    """
    pdbs = 'CRYST1   60.000   60.000   60.000  90.00  90.00  90.00 P 1           1\n'
    atomlis = coor_frame.reshape(-1, 3)
    if not CAlis:
        CAlis = ['ALA'] * len(atomlis)
    elif len(CAlis) != len(atomlis):
        raise IndexError('Length of CA list not equal to length of CA index lis')
    for i in range(len(atomlis)):
        pdbs += 'ATOM    {:>3d}  CA  {} A  {:>2d}     {:7.3f} {:7.3f} {:7.3f}  1.00  0.00           C\n'.format(i + 1, CAlis[i], i + 1, atomlis[i][0], atomlis[i][1], atomlis[i][2])
    pdbs += 'END\n'
    return pdbs


def CoorlisToLmpstrj(coorlis, outfilepath, boxlength=60):
    """
    Generate lammps trajectory based on the given coorlis.
    """
    frame = len(coorlis)
    if len(coorlis[0]) % 3 != 0:
        raise ValueError('length of coorlis % 3 != 0')
    atom = int(len(coorlis[0]) / 3)
    f = open(outfilepath)
    for i in range(frame):
        coor_frame = coorlis[i]
        minx = min([coor_frame[j] for j in range(0, len(coor_frame), 3)])
        miny = min([coor_frame[j] for j in range(1, len(coor_frame), 3)])
        minz = min([coor_frame[j] for j in range(0, len(coor_frame), 3)])
        f.write('ITEM: TIMESTEP\n{}\n'.format(i))
        f.write('ITEM: NUMBER OF ATOMS\n{}\n'.format(atom))
        f.write('ITEM: BOX BOUNDS pp pp pp\n{} {}\n{} {}\n{} {}'.format(minx,
                minx + boxlength, miny, miny + boxlength,
                minz, minz + boxlength))
        f.write('ITEM: ATOMS id type xu yu zu\n')
        for j in range(atom):
            x_, y_, z_ = coor_frame[3 * j], coor_frame[3 * j + 1], coor_frame[3 * j + 2]
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(j + 1, 1, x_, y_, z_))
    f.close()
    print('[INFO] Re-build a trj file {} from a coordinate list'.format(outfilepath))


def ConformationPlot(coor_frame):
    """
    Give a frame of trajectory, plot it.
    """
    x_lis = [coor_frame[j] for j in range(0, len(coor_frame), 3)]
    y_lis = [coor_frame[j] for j in range(1, len(coor_frame), 3)]
    z_lis = [coor_frame[j] for j in range(2, len(coor_frame), 3)]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_lis, y_lis, z_lis, c='r', s=30)
    ax.set_zlabel('z')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    plt.show()


def CalRmsdFrame(coor_frame, ref_coor_frame):
    """
    Calculate the rmsd between two frames.
    """
    coor_tmp = coor_frame.reshape(-1, 3)
    ref_tmp = ref_coor_frame.reshape(-1, 3)
    coor_tmp -= rmsd.centroid(coor_tmp)
    ref_tmp -= rmsd.centroid(ref_tmp)
    return rmsd.kabsch_rmsd(coor_tmp, ref_tmp)


def GetRmsd(outangdihlis, coorlis):
    """
    Reconstruct the coordination list from a angdih list, then calculate rmsd.
    """
    ave_bond = CalAveBondLength(coorlis)
    print('average bond length:\n', ave_bond)
    out_coorlis = []
    for frame in tqdm(outangdihlis, desc='[INFO] reconstructing'):
        out_coorlis.append(AngdihToCoor(frame, ave_bond))
    out_coorlis = np.array(out_coorlis)
    rmsd_lis = []
    for i in tqdm(range(len(coorlis))):
        rmsd_lis.append(CalRmsdFrame(coorlis[i], out_coorlis[i]))
    return np.array(rmsd_lis)


def LoadAll(dataset, modelname_time):
    """
    Load coordination list, angdih list and latent list.
    """
    coor_path = 'trajectory/{}/ca_lmptrj'.format(dataset)
    angdih_path = 'dataset/{}'.format(dataset)
    latent_path = 'training_result/{}/{}/latent'.format(dataset, modelname_time)
    latentfiles = [i for i in os.listdir(latent_path) if i.split('.')[-1] == 'latent']
    angfiles = [i for i in os.listdir(angdih_path) if i.split('.')[-1] == 'angdih']
    lmpsfiles = [i for i in os.listdir(coor_path) if i.split('.')[-1] == 'lammpstrj']
    angfiles.sort()
    lmpsfiles.sort()
    latentfiles.sort()
    lmpslis, angdihlis, latentlis= [], [], []
    for i in tqdm(range(len(lmpsfiles)), desc='[INFO] Loading trj, angdih and latent file...'):
        if not angfiles[i].split('.')[0] == lmpsfiles[i].split('.')[0] == latentfiles[i].split('.')[0]:
            print(lmpsfiles, angfiles, latentfiles)
            raise TypeError('Wrong file name')
        lmpslis.append(LoadLmpsFile(coor_path + '/' + lmpsfiles[i], verbose=0))
        angdihlis.append(LoadAngDihFile(angdih_path + '/' + angfiles[i], verbose=0)[-1])
        latentlis.append(LoadLatentFile(latent_path + '/' + latentfiles[i], verbose=0)[1])
    lmpslis_t = lmpslis[0]
    for i in range(len(lmpslis) - 1):
        lmpslis_t = np.vstack((lmpslis_t, lmpslis[i + 1]))
    angdihlis_t = angdihlis[0]
    for i in range(len(angdihlis) - 1):
        angdihlis_t = np.vstack((angdihlis_t, angdihlis[i + 1]))
    latentlis_t = latentlis[0]
    for i in range(len(latentlis) - 1):
        latentlis_t = np.vstack((latentlis_t, latentlis[i + 1]))  
    return lmpslis_t, angdihlis_t, latentlis_t


if __name__ == "__main__":
    # Load all the parameters
    args = args_parse()
    dataset = args.dataset
    model_time = args.modelname_time
    #latent_dim = args.latentdim
    #inter_dims = args.inter_dims
    samplerate = args.samplerate
    n_clusters = args.clusters_count

    # Extract all alpha-carbon
    _, CAlis = findCA(dataset)

    # Load all the files
    coorlis, angdihlis, latentlis = LoadAll(dataset, model_time)

    # Calculate the mean and scale of the angle-dihedral data
    #stand_scaler = preprocessing.StandardScaler()
    #xtotal_stand = stand_scaler.fit_transform(angdihlis)
    #xmean = stand_scaler.mean_
    #xscale = stand_scaler.scale_
    #print('xmean:', xmean)
    #print('xscale:', xscale)

    # Load the trained model
    #checkpoint_folder = 'training_result/{}/{}/checkpoint'.format(dataset, model_time)
    #for file in os.listdir(checkpoint_folder):
    #    if file.split('_')[0] == 'final':
    #        checkpoint_path = checkpoint_folder + '/' + file
    #para = AEparameter(len(angdihlis[0]),inter_dims, latent_dim)
    #myae = FCAE(para)
    #myae.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    # Sample data to decrease calculation complexity
    latentlis_s = SampleData(latentlis, samplerate)
    coorlis_s = SampleData(coorlis, samplerate)

    # Tsne
    myTsne = TSNE(n_components=2, verbose=0)
    embedlis_s = myTsne.fit_transform(latentlis_s)
    print('[INFO] t-SNE embedding completed.')

    # K-means clustering
    km = KMeans(n_clusters=n_clusters, verbose=0)
    km.fit(latentlis_s)
    labellis_s = km.labels_
    center = km.cluster_centers_
    print('[INFO] K-means clustering completed. {} clusters divided.'.format(n_clusters))

    # Divide sample data to different clusters 
    clusters_coor = [[] for i in range(n_clusters)]
    clusters_latent = [[] for i in range(n_clusters)]
    clusters_embed = [[] for i in range(n_clusters)]
    for i in range(len(labellis_s)):
        clusters_coor[labellis_s[i]].append(coorlis_s[i])
        clusters_latent[labellis_s[i]].append(latentlis_s[i])
        clusters_embed[labellis_s[i]].append(embedlis_s[i])
    print('[INFO] The number of frames of each cluster:')
    for i in range(len(clusters_coor)):
        print('cluster {}: {} frames'.format(i + 1, len(clusters_coor[i])))

    # Find the center of each cluster and write pdb files
    center_latent, center_coor, center_embed = [], [], []
    rmsdlis = []
    for i in tqdm(range(n_clusters), desc='[INFO] Generating pdb files of each cluster'):
        centerpath = 'clustering_result/{}/{}/{}_clusters'.format(dataset, model_time, n_clusters)
        if not os.path.exists(centerpath):
            os.makedirs(centerpath)
        clus = clusters_coor[i]
        cluslatent = clusters_latent[i]
        clusembed = clusters_embed[i]
        for j in range(len(clus)):
            if not os.path.exists(centerpath + '/cluster_{}'.format(i + 1)):
                os.makedirs(centerpath + '/cluster_{}'.format(i + 1))
            with open(centerpath + '/cluster_{}/{}.pdb'.format(i + 1, j + 1), 'w') as f:
                f.write(GenePdbString(clus[j], CAlis))

        center_latent_i = cluslatent[0]
        center_coor_i = clus[0]
        center_embed_i = clusembed[0]
        indexi = 0
        for j in range(len(cluslatent)):
            if np.linalg.norm(center_latent_i - center[i]) > np.linalg.norm(center[i] - cluslatent[j]):
                center_latent_i = cluslatent[j]
                center_coor_i = clus[j]
                center_embed_i = clusembed[j]
                indexi = j
        #print(center[i], center_latent_i)
        center_latent.append(center_latent_i)
        center_coor.append(center_coor_i)
        center_embed.append(center_embed_i)
        #rmsd_i = CalRmsdLis(np.array(groupcoor[i]), np.array(center_coor_i)).mean()
        
        with open(centerpath + '/cluster_{}_center.pdb'.format(i + 1), 'w') as f:
            f.write(GenePdbString(center_coor_i, CAlis))
    print('[INFO] Center structure of each cluster saved.')
    center_latent = np.array(center_latent)
    center_coor = np.array(center_coor)
    center_embed = np.array(center_embed)

    #print(len(center_embed))
    #print(center_embed)

    # Plot tsne figure
    plt.scatter(embedlis_s[:,0], embedlis_s[:,1],c=labellis_s, cmap=plt.cm.get_cmap('rainbow', n_clusters), alpha=0.8, s=2)
    #plt.scatter(center_embed[:,0], center_embed[:,1],c=[i for i in range(n_clusters)], cmap=plt.cm.get_cmap('rainbow', n_clusters), marker='o', s=100)
    cb=plt.colorbar()
    cb.set_ticks([])
    cb.set_label('clusters')
    plt.xticks([])
    plt.yticks([])
    plt.title('Visualization of latent space using t-SNE')
    plt.savefig(centerpath + '/tsne.png', dpi=150)
    print('[INFO] t-SNE figure saved.')
