# -*- coding: utf-8 -*-
# @Time    : 2021/01/24
# @Author  : Cong Wang
# @Github ：https://github.com/CongWang98
import os
import re
import math
import argparse
import mdtraj as md
import numpy as np
from tqdm import tqdm


def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('-trjfolder', '--trajectory_folder', default='example')
    # ap.add_argument('-toptype', '--topology_file_type', default='pdb')
    ap.add_argument('-trjtype', '--trajectory_file_type', default='xtc')
    # ap.add_argument('-calist', '--alpha_carbon_index_list', nargs='+', type=int, default=None)
    return ap.parse_args()


def convert2lmptrj(trjfolder, trjtype='xtc', toptype='pdb'):
    '''
    Convert trajectory to lammps trajectory
    '''
    folder = 'trajectory/' + trjfolder
    outfolder = 'trajectory/' + trjfolder + '/lmptrj'
    files = os.listdir(folder)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    print('[INFO] Trying to convert trj files in {}'.format(folder))
    for file in files:
        if file.split('.')[-1] == toptype:
            topfile = file
            break
    trjfilecount = 0
    trjframe = 0
    for file in tqdm(files):
        if file.split('.')[-1] == trjtype:
            trjfilecount += 1
            # print(folder + '/' + file)
            # print(folder + '/' + topfile)
            trj_tmp = md.load(folder + '/' + file, top=folder + '/' + topfile)
            trj_tmp.unitcell_lengths = np.array([[60.0, 60.0, 60.0] for i in range(len(trj_tmp.xyz))])
            # print(file)
            # print(len(trj_tmp.xyz))
            # print(trj_tmp.unitcell_lengths)
            # print(trj_tmp.unitcell_angles)
            trjframe += len(trj_tmp)
            out_file = file.split('.')[0] + '.lammpstrj'
            trj_tmp.save_lammpstrj(outfolder + '/' + out_file)
            print('[INFO] {} was loaded. It contains {} frames.'.format(file, len(trj_tmp)))
    print('[INFO] {} trajectory files are loaded. There are {} frames in total.'.format(trjfilecount, trjframe))


def findCA(trjfolder):
    """
    Return a list of the index and a list the abbreviate of all CA in a pdb file.
    """
    files = os.listdir('trajectory/' + trjfolder)
    topfile = None
    for file in files:
        if file.split('.')[-1] == 'pdb':
            topfile = 'trajectory/' + trjfolder + '/' + file
            break
    if not topfile:
        raise TypeError('Only pdb file is accepted.')
    else:
        CA_index_lis = []
        CA_lis = []
        with open(topfile, 'r') as f:
            line = f.readline()
            while line:
                if line.split()[0] == 'ATOM':
                    if line.split()[2] == 'CA':
                        CA_index_lis.append(int(line.split()[1]))
                        CA_lis.append(line.split()[3])
                line = f.readline()
    return CA_index_lis, CA_lis


def camap(trjfolder, calist):
    ca_amount = len(calist)
    calist.sort()
    lmp_folder = 'trajectory/' + trjfolder + '/lmptrj'
    ca_lmp_folder = 'trajectory/' + trjfolder + '/ca_lmptrj'
    if not os.path.exists(ca_lmp_folder):
        os.makedirs(ca_lmp_folder)
    files = os.listdir(lmp_folder)
    print('[INFO] Trying to map trajectory files in {}'.format(lmp_folder))
    trjfilecount = 0
    for file in files:
        if file.split('.')[-1] == 'lammpstrj':
            fin = open(lmp_folder + '/' + file, 'r')
            output_file = ca_lmp_folder + '/' + file
            fout = open(output_file, 'w')
            trjfilecount += 1
            frame = 0
            line = fin.readline()
            while True:
                if not line:
                    break
                if (line + ' placeholder').split()[1] == 'ATOMS':
                    fout.write(line)
                    frame += 1
                    cacount = 0
                    atomcount = 0
                    line = fin.readline()
                    while line and (line + ' placeholder').split()[0] != 'ITEM:':
                        atomcount += 1
                        if cacount >= len(calist):
                            pass
                        elif atomcount == calist[cacount]:
                            cacount += 1
                            coor_lis = line.split()[2:]
                            fout.write('{} {} '.format(cacount, 1))
                            fout.write(' '.join(coor_lis))
                            fout.write('\n')
                        else:
                            pass
                        line = fin.readline()
                    if frame % 10000 == 0:
                        print('[INFO] {} frames mapped.'.format(frame))
                elif (line + ' placeholder').split()[1] == 'NUMBER':
                    fout.write(line)
                    fin.readline()
                    fout.write('{}\n'.format(ca_amount))
                    line = fin.readline()
                else:
                    fout.write(line)
                    line = fin.readline()
            print('[INFO] {} is mapped. It contains {} frames.'.format(file, frame))
            fin.close()
    print('[INFO] {} trajetory files are loaded in total.'.format(trjfilecount))


def get_ang(a, b, c):
    '''
    The angle is a-b-c. Return a scale value ranging from 0-1(unit: pi).
    '''
    # print('a=', a)
    # print('b=', b)
    # print('c=', c)
    ba = np.array(a) - np.array(b)
    # print('ba=', ba)
    bc = np.array(c) - np.array(b)
    # print('bc=', bc)
    cos = np.dot(ba, bc).sum() / (get_abs(ba) * get_abs(bc))
    # print('ba*bc/|ba||bc|=', cos)
    # print('theta=', math.acos(cos) * 180 / math.pi)
    return math.acos(cos) / math.pi


def get_dih(a, b, c, d):
    '''
    The dihedral is a-b-c-d.Return a scale value ranging from 0-1(unit: 2pi).
    '''
    # print('a=', a)
    # print('b=', b)
    # print('c=', c)
    # print('d=', d)
    n1 = np.cross(np.array(a) - np.array(b), np.array(c) - np.array(b))
    # print('n1=ba cross bc=', n1)
    n2 = np.cross(np.array(b) - np.array(c), np.array(d) - np.array(c))
    # print('n2=cb cross cd=', n2)
    cos = np.dot(n1, n2).sum() / (get_abs(n1) * get_abs(n2))
    # print('cos=n1*n2/|n1||n2|=', cos)
    # print('theta=', math.acos(cos) * 180 / math.pi)
    theta = math.acos(cos)
    if np.dot(n1, np.array(d) - np.array(c)) < 0:
        theta = 2 * math.pi - theta
    return theta / (2 * math.pi)


def get_dis(a, b):
    '''
    Return the distance between two atom.
    '''
    tmp = np.array(a) - np.array(b)
    return math.sqrt(np.dot(tmp, tmp).sum())


def get_abs(a):
    '''
    Return the absolute value of a verctor.
    '''
    return get_dis(a, [0, 0, 0])


def LoadLmpsFile(trj_file, verbose=1):
    '''
    Extract the coordinate list from a lammpstrj file.
    '''
    if verbose:
        print('[INFO] Loading data from {}.'.format(trj_file))
    coor_lis = []
    if not trj_file.split('.')[-1] == 'lammpstrj':
        raise TypeError('Wrong file type')
    with open(trj_file, 'r') as f:
        count = 0
        if verbose:
            print('[INFO] Trajtory file is loading......')
        while True:
            line = f.readline()
            if not line:
                if verbose:
                    print('\n[INFO] Loading ended! {} frames loaded in total.'.format(count))
                break
            if (line+' 1').split()[1] == 'ATOMS':
                count += 1
                coor_i = []
                '''
                for i in range(13):
                    line = f.readline()
                    line_sp = line.split()
                    coor_i += [float(coor) for coor in line_sp[2:5]]
                '''
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line_sp = line.split()
                    if line_sp[0] == 'ITEM:':
                        break
                    coor_i += [float(coor) for coor in line_sp[2:5]]
                coor_lis.append(coor_i)
                if (not count % 10000) and verbose:
                    print('\r[INFO] {} frames loaded.'.format(count), end='')
    return np.array(coor_lis)


def CalAngle(frame):
    '''
    Give the coordinate of a single frame, return angles of the frame.
    '''
    assert(len(frame) % 3 == 0)
    atomnumber = int(len(frame) / 3)
    angle_lis = []
    for i in range(atomnumber - 2):
        start = i * 3
        a = frame[start: start + 3]
        b = frame[start + 3: start + 6]
        c = frame[start + 6: start + 9]
        angle = get_ang(a, b, c)
        angle_lis.append(angle)
    return np.array(angle_lis, dtype=np.float32)


def CalDihedral(frame):
    '''
    Give the coordinate of a single frame, return dihedrals of the frame.
    '''
    atomnumber = int(len(frame) / 3)
    dihedral_lis = []
    for i in range(atomnumber - 3):
        start = i * 3
        a = frame[start: start + 3]
        b = frame[start + 3: start + 6]
        c = frame[start + 6: start + 9]
        d = frame[start + 9: start + 12]
        dihedral = get_dih(a, b, c, d)
        dihedral_lis.append(dihedral)
    return np.array(dihedral_lis, dtype=np.float32)


def GenAngDihFile(filepath, folderpath=None, verbose=1):
    '''
    Load a lammpstrj file, then generate a file contain angles and dihedrals.
    '''
    coor_lis = LoadLmpsFile(filepath)
    # print(coor_lis)
    frame = len(coor_lis)
    # print(frame)
    # print(len(coor_lis[0]))
    atom = int(len(coor_lis[0]) / 3)
    assert (len(coor_lis[0]) % 3 == 0)
    ifilename = re.split(r'[/\\]', filepath)[-1]
    if folderpath is None:
        folderpath = '/'.join([i for i in re.split(r'[/\\]', filepath)[0:-1] if i != ''])
    ofilename = re.split(r'[.]', ifilename)[0] + '.angdih'
    # print(ifilename)
    # print(folderpath)
    # print(ofilename)
    angle_lis = np.array([CalAngle(i) for i in tqdm(coor_lis)], dtype=np.float32)
    dihedral_lis = np.array([CalDihedral(i) for i in tqdm(coor_lis)], dtype=np.float32)
    with open(folderpath + '/' + ofilename, 'w') as f:
        f.write('{} {}\n'.format(frame, atom))
        for i in range(frame):
            angle = angle_lis[i]
            dihedral = dihedral_lis[i]
            combine_lis = []
            combine_lis.append(str(angle[0]))
            for j in range(len(dihedral)):
                combine_lis.append(str(dihedral[j]))
                combine_lis.append(str(angle[j + 1]))
            f.write(' '.join(combine_lis) + '\n')
        if verbose:
            print('[INFO] {}/{} has been generated.'.format(folderpath, ofilename))


def GenAllAngDihFile(path, outfolder=None, verbose=1):
    '''
    Generate ang-dih file for all lammpstrj files in a folder.
    '''
    files = os.listdir(path)
    count = 0
    for file in files:
        if file.split('.')[-1] == 'lammpstrj' or file.split('.')[-1] == 'ca':
            GenAngDihFile(path + '/' + file, outfolder)
            count += 1
    if verbose:
        print('[INFO] {} angdih files from {} have been generated'.format(count, path))


def LoadAngDihFile(filepath, verbose=1):
    '''
    Load angle and dihedrals from a file, return the number of frames,
    atoms and a np array.
    '''
    lis = []
    if verbose:
        print('[INFO] Loading angdih file {}.'.format(filepath))
    if filepath.split('.')[-1] != 'angdih':
        raise TypeError('Not a angdih file')
    with open(filepath, 'r') as f:
        line = f.readline()
        frame, atom = [int(i) for i in line.split()]
        line = f.readline()
        while line:
            sp = [float(i) for i in line.split()]
            if sp != []:
                lis.append(sp)
            line = f.readline()
    if verbose:
        print('[INFO] Load angdih file {}. Frame: {} Atom: {}'.format(filepath, frame, atom))
    if len(lis) != frame:
        raise ValueError('Wrong frame count')
    return frame, atom, np.array(lis, dtype=np.float32)


def LoadLatentFile(filepath, verbose=1):
    '''
    Load latent variables from a file, return latent dim, a latent array.
    '''
    lis = []
    if verbose:
        print('[INFO] Loading latent file {}.'.format(filepath))
    if filepath.split('.')[-1] != 'latent':
        raise TypeError('Not a latent file')
    with open(filepath, 'r') as f:
        line = f.readline()
        while line:
            sp = [float(i) for i in line.split()]
            lis.append(sp)
            line = f.readline()
    if verbose:
        print('[INFO] Load latent file {}. Frame: {} '.format(filepath, len(lis)))
    return len(lis), np.array(lis, dtype=np.float32)


def SampleData(lis, samplerate):
    '''
    Sample data from a given list.
    '''
    return np.array([lis[i] for i in range(len(lis)) if i % int(1/samplerate) == 0])


def DivideAdlis(lis):
    """Give an adlis, divide it into lis of each angle and dihedral"""
    anslis = []
    for i in range(len(lis[0])):
        tmp = np.array([lis[j][i] for j in range(len(lis))])
        anslis.append(tmp)
    return anslis


if __name__ == "__main__":
    args = args_parse()
    trjfolder = args.trajectory_folder
    toptype = 'pdb'
    trjtype = args.trajectory_file_type
    # calist = args.alpha_carbon_index_list

    convert2lmptrj(trjfolder, trjtype, toptype)
    calis, _ = findCA(trjfolder)
    camap(trjfolder, calis)
    ca_lmp_folder = 'trajectory/' + trjfolder + '/ca_lmptrj'
    angdih_folder = 'dataset/' + trjfolder
    if not os.path.exists(angdih_folder):
        os.makedirs(angdih_folder)
    GenAllAngDihFile(ca_lmp_folder, outfolder=angdih_folder, verbose=1)
