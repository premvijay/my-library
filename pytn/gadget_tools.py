import numpy as np
import os
import gc

class Snapshot():
    def __init__(self, snapfile=None, hdf5_support='True'):
        if hdf5_support:
            import h5py
            self.h5py = h5py
        if snapfile is None:
            print("Instantiated a snapshot object, use 'from_binary' method to read from binary.")
        else:
            self.from_binary(snapfile)


    def from_binary(self, filename = None, header=True):
        assert type(filename) is str, "This class requires the gadget filename as input"
        self.filename = filename
        if not os.path.exists(self.filename): self.filename += '.hdf5' 
        self.filetype = 'gadget_binary' if self.filename[-5:]!='.hdf5' else 'gadget_hdf5'
        if header==True:
            self.read_header()
        else:
            pass

    def read_header(self):
        if self.filetype == 'gadget_binary':
            file = open(self.filename,'rb')
            header_size = np.fromfile(file, dtype=np.uint32, count=1)
            print ("reading the first block (header) which contains ", header_size, " bytes")
            self.N_prtcl_thisfile = np.fromfile(file, dtype=np.uint32, count=6)    ## The number of particles of each type present in the file
            self.mass_table       = np.fromfile(file, dtype=np.float64, count=6)     ## Gives the mass of different particles
            self.scale_factor     = np.fromfile(file, dtype=np.float64, count=1)[0]   ##Time of output,  or expansion factor for cosmological simulations
            self.redshift         = np.fromfile(file, dtype=np.float64, count=1)[0]   ## Redshift of the snapshot
            self.flag_sfr         = np.fromfile(file, dtype=np.int32, count=1)[0]     ##Flag for star 
            self.flag_feedback    = np.fromfile(file, dtype=np.int32, count  = 1)[0]  ##Flag for feedback
            self.N_prtcl_total    = np.fromfile(file, dtype=np.uint32, count = 6)  ## Total number of each particle present in the simulation
            self.flag_cooling     = np.fromfile(file, dtype=np.int32, count =1)[0]     ## Flag used for cooling
            self.num_files        = np.fromfile(file, dtype=np.int32, count = 1)[0] ## Number of files in each snapshot
            self.box_size         = np.fromfile(file, dtype = np.float64, count = 1)[0]  ## Gives the box size if periodic boundary conditions are used
            self.Omega_m_0        = np.fromfile(file, dtype = np.float64, count=1)[0]     ## Matter density at z = 0 in the units of critical density
            self.Omega_Lam_0      = np.fromfile(file, dtype = np.float64, count=1)[0]## Vacuum Energy Density at z=0 in the units of critical density
            self.Hubble_param     = np.fromfile(file, dtype = np.float64, count =1 )[0] ## gives the hubble constant in units of 100 kms^-1Mpc^-1  
            self.flag_stellar_age = np.fromfile(file, dtype = np.int32 , count =1)[0]  ##Creation time of stars
            self.flag_metals      = np.fromfile(file, dtype = np.int32 , count =1)[0] ##Flag for metallicity values
            self.N_prtcl_total_HW = np.fromfile(file, dtype = np.int32, count = 6) ## For simulations more that 2^32 particles this field holds the most significant word of the 64 bit total particle number,  otherwise 0
            self.flag_entropy_ICs = np.fromfile(file, dtype = np.int32, count = 1)[0] ## Flag that initial conditions contain entropy instead of thermal energy in the u block
            file.seek(256 +4 , 0)
            header_size_end = np.fromfile(file, dtype = np.int32, count =1)[0]
            print ('Header block is read and it contains ', header_size_end, 'bytes.')
            
        elif self.filetype == 'gadget_hdf5':
            h5file = self.h5py.File(self.filename, 'r')
            self.N_prtcl_thisfile = h5file['Header'].attrs['NumPart_ThisFile']    ## The number of particles of each type present in the file
            self.mass_table       = h5file['Header'].attrs['MassTable']     ## Gives the mass of different particles
            self.scale_factor     = h5file['Header'].attrs['Time']   ##Time of output,  or expansion factor for cosmological simulations
            self.redshift         = h5file['Header'].attrs['Redshift']   ## Redshift of the snapshot
            self.N_prtcl_total    = h5file['Header'].attrs['NumPart_Total']   ## Total number of each particle present in the simulation
            self.num_files        = h5file['Header'].attrs['NumFilesPerSnapshot'] ## Number of files in each snapshot
            self.box_size         = h5file['Header'].attrs['BoxSize']  ## Gives the box size if periodic boundary conditions are used
            # self.Omega_m_0        = h5file['Parameters'].attrs['Omega0']     ## Matter density at z = 0 in the units of critical density
            # self.Omega_Lam_0      = h5file['Parameters'].attrs['OmegaLambda'] ## Vacuum Energy Density at z=0 in the units of critical density
            # self.Hubble_param     = h5file['Parameters'].attrs['HubbleParam'] ## gives the hubble constant in units of 100 kms^-1Mpc^-1  
            # # self.num_part_types   = h5file['Config'].attrs['NTYPES']
            # self.params           = h5file['Parameters'].attrs

            self.h5file = h5file

        self.prtcl_types = ["Gas","Halo","Disk",  "Bulge", "Stars", "Bndry"]
        
    def positions(self, prtcl_type="Halo",max_prtcl=None):
        if self.filetype == 'gadget_binary':
            file = open(self.filename,'rb')
            file.seek(256+8, 0)
            position_block_size = np.fromfile(file, dtype = np.int32, count =1)[0]
            print ("reading the second block (position) which contains ", position_block_size, " bytes")
            i = 0
            while self.prtcl_types[i] != prtcl_type:
                file.seek(self.N_prtcl_thisfile[i]*3*4, 1)
                i += 1
            N_prtcl = self.N_prtcl_thisfile[i] if max_prtcl is None else max_prtcl
            posd = np.fromfile(file, dtype = np.float32, count = N_prtcl*3)  ### The positions are arranged in the binary file as follow: x1,y1,z1,x2,y2,z2,x3,y3,z3 and so on till xn,yn,zn
            posd = posd.reshape((N_prtcl, 3))   ## reshape keeps the fastest changing axis in the end, since x,y,z dimensions are the ones changing the fastest they are given the last axis.
            if max_prtcl is not None:
                print('Positions of {} particles is read'.format(N_prtcl))
            else:
                end  = np.fromfile(file, dtype = np.int32, count =1)[0]
                print ('Position block is read and it contains ', end, 'bytes')
            return posd

        elif self.filetype == 'gadget_hdf5':
            # h5file = self.h5py.File(self.filename, 'r')
            # if prtcl_type=="Halo": 
            type_num = self.prtcl_types.index(prtcl_type)
            return self.h5file[f'PartType{type_num:d}']['Coordinates'][:]

    def velocities(self, prtcl_type="Halo",max_prtcl=None):
        if self.filetype == 'gadget_binary':
            file = open(self.filename,'rb')
            file.seek(256+8+8 + int(self.N_prtcl_thisfile.sum())*3*4, 0)
            velocity_block_size = np.fromfile(file, dtype = np.int32, count =1)[0]
            print ("reading the third block (position) which contains ", velocity_block_size, " bytes")
            i = 0
            while self.prtcl_types[i] != prtcl_type:
                file.seek(self.N_prtcl_thisfile[i]*3*4, 1)
                i += 1
            N_prtcl = self.N_prtcl_thisfile[i] if max_prtcl is None else max_prtcl
            veld = np.fromfile(file, dtype = np.float32, count = N_prtcl*3)  ### The velocities are arranged in the binary file as follow: vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3 and so on till vxn,vyn,vzn
            veld = veld.reshape((N_prtcl, 3))   ## reshape keeps the fastest changing axis in the end, since vx,vy,vz dimensions are the ones changing the fastest they are given the last axis.
            if max_prtcl is not None:
                print('velocities of {} particles is read'.format(N_prtcl))
            else:
                end  = np.fromfile(file, dtype = np.int32, count =1)[0]
                print ('velocity block is read and it contains ', end, 'bytes')
            return veld

        elif self.filetype == 'gadget_hdf5':
            # h5file = self.h5py.File(self.filename, 'r')
            # if prtcl_type=="Halo": 
            type_num = self.prtcl_types.index(prtcl_type)
            return self.h5file[f'PartType{type_num:d}']['Velocities'][:]

    def IDs(self, prtcl_type="Halo",max_prtcl=None):
        if self.filetype == 'gadget_binary':
            raise NotImplementedError

        elif self.filetype == 'gadget_hdf5':
            # h5file = self.h5py.File(self.filename, 'r')
            # if prtcl_type=="Halo": 
            type_num = self.prtcl_types.index(prtcl_type)
            return self.h5file[f'PartType{type_num:d}']['ParticleIDs'][:]




def read_positions_all_files(snapshot_filepath_prefix,downsample=1, rand_seed=10, prtcl_type='Halo'):
    pos_list = []
    np.random.seed(rand_seed)

    file_number = 0
    while True:
        filepath = snapshot_filepath_prefix + ''
        if not os.path.exists(filepath): filepath = snapshot_filepath_prefix + '.hdf5'

        if not os.path.exists(filepath): filepath = snapshot_filepath_prefix + '.{0:d}'.format(file_number)
        if not os.path.exists(filepath): filepath = snapshot_filepath_prefix + '.{0:d}'.format(file_number) + '.hdf5'
    
        # print(filepath, os.path.exists(filepath))

        snap = Snapshot(filepath)
        # snap.from_binary()

        posd_thisfile = snap.positions(prtcl_type=prtcl_type, max_prtcl=None)
        if downsample != 1:
            rand_ind = np.random.choice(posd_thisfile.shape[0], size=posd_thisfile.shape[0]//downsample, replace=False)
            # print(snap.N_prtcl_thisfile, downsample, 'n', snap.N_prtcl_thisfile//downsample, rand_ind)
            posd_thisfile = posd_thisfile[rand_ind]


        pos_list.append(posd_thisfile)
        if file_number == snap.num_files-1:
            break
        else:
            file_number += 1

    posd = np.vstack(pos_list)
    del pos_list[:]
    del pos_list
    gc.collect()

    return posd


def read_velocities_all_files(snapshot_filepath_prefix,downsample=1, rand_seed=10):
    vel_list = []
    np.random.seed(rand_seed)

    file_number = 0
    while True:
        filepath = snapshot_filepath_prefix + ''
        if not os.path.exists(filepath): filepath = snapshot_filepath_prefix + '.hdf5'

        if not os.path.exists(filepath): filepath = snapshot_filepath_prefix + '.{0:d}'.format(file_number)
        if not os.path.exists(filepath): filepath = snapshot_filepath_prefix + '.{0:d}'.format(file_number) + '.hdf5'

        # filename_suffix = '.{0:d}'.format(file_number) if not os.path.exists(snapshot_filepath_prefix) else ''
        # filepath = os.path.join(binary_files_dir, filename)
        # filepath = snapshot_filepath_prefix + filename_suffix
        print(filepath)

        snap = Snapshot()
        snap.from_binary(filepath)

        veld_thisfile = snap.velocities(prtcl_type="Halo", max_prtcl=None)
        if downsample != 1:
            rand_ind = np.random.choice(veld_thisfile.shape[0], size=veld_thisfile.shape[0]//downsample, replace=False)
            # print(snap.N_prtcl_thisfile, downsample, 'n', snap.N_prtcl_thisfile//downsample, rand_ind)
            veld_thisfile = veld_thisfile[rand_ind]


        vel_list.append(veld_thisfile)
        if file_number == snap.num_files-1:
            break
        else:
            file_number += 1

    veld = np.vstack(vel_list)
    del vel_list[:]

    return veld

def read_partIDs_all_files(snapshot_filepath_prefix,downsample=1, rand_seed=10, prtcl_type='Halo'):
    pID_list = []
    np.random.seed(rand_seed)

    file_number = 0
    while True:
        filepath = snapshot_filepath_prefix + ''
        if not os.path.exists(filepath): filepath = snapshot_filepath_prefix + '.hdf5'

        if not os.path.exists(filepath): filepath = snapshot_filepath_prefix + '.{0:d}'.format(file_number)
        if not os.path.exists(filepath): filepath = snapshot_filepath_prefix + '.{0:d}'.format(file_number) + '.hdf5'
    
        print(filepath, os.path.exists(filepath))

        snap = Snapshot()
        snap.from_binary(filepath)

        pID_thisfile = snap.IDs(prtcl_type=prtcl_type, max_prtcl=None)
        if downsample != 1:
            rand_ind = np.random.choice(pID_thisfile.shape[0], size=pID_thisfile.shape[0]//downsample, replace=False)
            # print(snap.N_prtcl_thisfile, downsample, 'n', snap.N_prtcl_thisfile//downsample, rand_ind)
            pID_thisfile = pID_thisfile[rand_ind]


        pID_list.append(pID_thisfile)
        if file_number == snap.num_files-1:
            break
        else:
            file_number += 1

    pID = np.concatenate(pID_list)
    del pID_list[:]
    del pID_list
    gc.collect()

    return pID


def read_all_hdf5(quantity, prtcl_type_num, filename_pref):
    import h5py
    prtcl_type_str = 'PartType'+str(int(prtcl_type_num))
    i=0
    filename = filename_pref + f'.{i:d}.hdf5'
    h5file = h5py.File(filename, 'r')
    N = h5file['Header'].attrs['NumFilesPerSnapshot']
    data_list = []
    while i<N:
        filename = filename_pref + f'.{i:d}.hdf5'
        h5file = h5py.File(filename, 'r')
        data_list.append( h5file[f'PartType{prtcl_type_num:d}'][quantity] )
        i+=1
    return np.concatenate(data_list, axis=0)
        




## To do for RAM limited situations
# def read_snapshot_all_files(snapshot_filepath_prefix, get_pos=True, get_vel=True, downsample=1):
#     file_number = 0
#     filename_suffix = '.{0:d}'.format(file_number)
#     filepath = snapshot_filepath_prefix + filename_suffix
#     print(filepath)

#     snap = Snapshot()
#     snap.from_binary(filepath)

    
#     if get_pos:
#         posd = np.zeros(shape=(self.N_prtcl_total,3))
#     if get_vel:
#         veld = np.zeros(shape=(self.N_prtcl_total,3))
    
#     for file_number in range(snap.num_files):
#         filename_suffix = '.{0:d}'.format(file_number)
#         filepath = snapshot_filepath_prefix + filename_suffix
#         print(filepath)

#         snap = Snapshot()
#         snap.from_binary(filepath)

#         posd = snap.velocities(prtcl_type="Halo", max_prtcl=None)

#     while True:
#         filename_suffix = '.{0:d}'.format(file_number)
#         filepath = snapshot_filepath_prefix + filename_suffix
#         print(filepath)

#         snap = Snapshot()
#         snap.from_binary(filepath)

#         posd_thisfile = snap.velocities(prtcl_type="Halo", max_prtcl=None)
#         if downsample != 1:
#             rand_ind = np.random.choice(posd_thisfile.shape[0], size=posd_thisfile.shape[0]//downsample, replace=False)
#             # print(snap.N_prtcl_thisfile, downsample, 'n', snap.N_prtcl_thisfile//downsample, rand_ind)
#             posd_thisfile = posd_thisfile[rand_ind]


#         pos_list.append(posd_thisfile)
#         if file_number == snap.num_files-1:
#             break
#         else:
#             file_number += 1

#     posd = np.vstack(pos_list)
#     # del pos_list[:]

#     return posd


# def downsample()



























