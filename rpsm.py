import pickle
import numpy as np
import torch
from tabulate import tabulate
from body import HumanBody

def infer(unary, pairwise, body):
    """
    Args:
        unary:

        pairwise:

        body:
    
    Return:
        pose3d_as_cube_idx, states_of_all_joints
    """
    skeleton = body.skeleton
    skeleton_sorted_by_level = body.skeleton_sorted_by_level

    root_idx = 6

    states_of_all_joints = {}
    for node in skeleton_sorted_by_level:
        children_state = []
        u = unary[node['idx']].clone()
        if len(node['children']) == 0:
            energy = u
            children_state = [[-1]] * energy.numel()
        else:
            for child in node['children']:
                pw = pairwise[(node['idx'], child)]
                ce = states_of_all_joints[child]['Energy']
                ce = ce.expand_as(pw)
                pwce = torch.mul(pw, ce)
                max_v, max_i = torch.max(pwce, dim=1)
                u = torch.mul(u, max_v)
                children_state.append(max_i.detach().cpu().numpy())

            children_state = np.array(children_state).T

        res = {'Energy': u, 'State': children_state}
        states_of_all_joints[node['idx']] = res

    pose3d_as_cube_idx = []
    energy = states_of_all_joints[root_idx]['Energy'].detach().cpu().numpy()
    cube_idx = np.argmax(energy)
    pose3d_as_cube_idx.append([root_idx, cube_idx])

    queue = pose3d_as_cube_idx.copy()
    while queue:
        joint_idx, cube_idx = queue.pop(0)
        children_state = states_of_all_joints[joint_idx]['State']
        state = children_state[cube_idx]

        children_index = skeleton[joint_idx]['children']
        if -1 not in state:
            for joint_idx, cube_idx in zip(children_index, state):
                pose3d_as_cube_idx.append([joint_idx, cube_idx])
                queue.append([joint_idx, cube_idx])

    pose3d_as_cube_idx.sort()
    return pose3d_as_cube_idx, states_of_all_joints

def get_loc_from_states_of_joints(states_of_all_joints, grids):
    njoints = len(states_of_all_joints)
    pose3d = torch.zeros((njoints, 3))
    for joint_idx in range(njoints):
        energy = torch.nn.functional.softmax(states_of_all_joints[joint_idx]['Energy'].view(-1,1), dim=0)
        # energy = states_of_all_joints[joint_idx]['Energy']
        # energy = (energy/(torch.sum(energy)+1e-8)).view(-1,1)
        energy = energy.expand_as(grids[joint_idx])
        pose3d[joint_idx] = torch.sum(energy*grids[joint_idx], dim=0)
   
    return pose3d

def get_loc_from_cube_idx(grid, pose3d_as_cube_idx):
    """
    Estimate 3d joint locations from cube index.

    Args:
        grid: a list of grids
        pose3d_as_cube_idx: a list of tuples (joint_idx, cube_idx)
    Returns:
        pose3d: 3d pose
    """
    njoints = len(pose3d_as_cube_idx)
    pose3d = torch.zeros(njoints, 3, device=grid[0].device)
    single_grid = len(grid) == 1
    for joint_idx, cube_idx in pose3d_as_cube_idx:
        gridid = 0 if single_grid else joint_idx
        pose3d[joint_idx] = grid[gridid][cube_idx]
    return pose3d

def compute_limb_length(keypoints_3d_gt, skeleton):
    limb_length_gt = {}
    for node in skeleton:
        current = node['idx']
        children = node['children']
        for child in children:
            limb_length_gt[(current, child)] = np.linalg.norm(keypoints_3d_gt[current] - keypoints_3d_gt[child])
    return limb_length_gt

def pdist2(x, y):
    """
    Compute distance between each pair of row vectors in x and y

    Args:
        x: tensor of shape n*p
        y: tensor of shape m*p
    Returns:
        dist: tensor of shape n*m
    """
    p = x.shape[1]
    n = x.shape[0]
    m = y.shape[0]
    xtile = torch.cat([x] * m, dim=1).view(-1, p)
    ytile = torch.cat([y] * n, dim=0)
    dist = torch.pairwise_distance(xtile, ytile)
    return dist.view(n, m)

def probabilty_function(x, sigma, mu):
    return np.exp((-1/2)*(((x-mu)/sigma)**2))/(sigma*np.sqrt(2*np.pi))

def compute_pairwise(skeleton, limb_length_gt, grids, tolerance):
    """
    Args:
        skeleton: n lists of dicts of {'childern:'[], 'idx':[], 'level':[], 'name':[], 'parent':[]}
            -n: number of joints
        
        limb_length_gt: a dict of k keys
            -k: number of child-parent connections (16 connections - (lelbow, lwrist), (pelvis, lhip), ...)

        grids: n lists of tensors of size (nbins * 3)
            -n: number of joints
            -nbins: number of bins in the grid
        
        tolerance: a single value
    Return:
        pairwise: a dict of k keys of n * n values
            -k: number of child-parent connections (16 connections - (lelbow, lwrist), (pelvis, lhip), ...)
            -n: number of bins in grid
    """
    pairwise = {}
    for node in skeleton:
        current = node['idx']
        children = node['children']
        for child in children:
            # length of the segments based on the ground truth data
            expect_length = limb_length_gt[(current, child)]

            # compute the distance between the child grid and the parent grid
            distance = pdist2(grids[current], grids[child])
            
            # compute the pairwise probability term using a gaussian distribution
            probability = probabilty_function(distance - expect_length, tolerance, 0)

            # normalize the probabilities
            probability = (probability/torch.max(probability + 1e-9))

            pairwise[(current, child)] = probability

            # pairwise[(current, child)] = (torch.abs(distance - expect_length) < tolerance).float()
    return pairwise

def compute_grid(boxSize, boxCenter, nBins, device=None):
    
    grid1D = torch.linspace(-boxSize / 2, boxSize / 2, nBins, device=device)

    gridx, gridy, gridz = torch.meshgrid(
        grid1D + boxCenter[0],
        grid1D + boxCenter[1],
        grid1D + boxCenter[2],)
    
    gridx = gridx.contiguous().view(-1, 1)
    gridy = gridy.contiguous().view(-1, 1)
    gridz = gridz.contiguous().view(-1, 1)
    grid = torch.cat([gridx, gridy, gridz], dim=1)

    return grid

def compute_unary_term(volumes_pred, coord_volumes_pred, grids):
    """
    compute the probability of the joints existence in the bins of grid

    Args:
        volumes_pred: array of size (n * i * j * k)
            -n: number of joints
            -i: length
            -j: width
            -k: height
        
        coord_volumes_pred: array of size (i * j * k * 3)
            -i: length
            -j: width
            -k: height

        grids: n lists of tensors of size (nbins * 3)
            -n: number of joints
            -nbins: number of bins in the grid
    
    Returns:
        all_unary_list: a list of tensors of size nbins

    """
    # initialization of parameters
    njoints = volumes_pred.shape[0]
    nbins = grids[0].shape[0]

    xxx = coord_volumes_pred[:,0,0,0]
    yyy = coord_volumes_pred[0,:,0,1]
    zzz = coord_volumes_pred[0,0,:,2]

    idxx,idxy, idxz = np.meshgrid(np.arange(-1,2), np.arange(-1,2), np.arange(-1,2))
    idxs = np.stack([idxx, idxy, idxz], axis=-1)
    idxs = idxs.reshape((-1, 3))

    index = np.ones((njoints, nbins, 27, 3), dtype=np.int16)
    dist  = np.zeros((njoints, nbins, 27), dtype=np.float64)
    ndist = np.zeros_like(dist)
    all_unary_list = []
    all_unary = np.zeros((njoints,nbins), dtype=np.float64)
    unary_temp = np.zeros((njoints,nbins,27), dtype=np.float64)

    for i in range(njoints):
        for j, grid  in enumerate(grids[i]):
            # find the index of closest coordinate in the coord_volumes_pred to the bins of grid
            index[i][j][0][0] = np.argmin(np.abs(xxx - grid[0].numpy()))
            index[i][j][0][1] = np.argmin(np.abs(yyy - grid[1].numpy()))
            index[i][j][0][2] = np.argmin(np.abs(zzz - grid[2].numpy()))

            # determine the indices of neighboring bins of the located bin in coord_volumes_pred
            index[i][j][1:] = index[i][j][0]*index[i][j][1:]
            index[i][j] = index[i][j] + idxs

            # compute the distance of the bin of grid to the determined bins of coord_volumes_pred
            for k in range(index.shape[2]):
                dist[i][j][k] = np.linalg.norm(coord_volumes_pred[index[i][j][k][0], index[i][j][k][1], index[i][j][k][2]] - grid.numpy())
            
            # normalize the distances
            ndist[i][j] = (1/dist[i][j])/np.sum(1/dist[i][j])

            # compute the unary term, based on the normalized distances of the bin of grid from the bins of the coord_volumes_pred
            for k in range(index.shape[2]):
                unary_temp[i][j][k] = ndist[i][j][k]*volumes_pred[i, index[i][j][k][0], index[i][j][k][1], index[i][j][k][2]]
            all_unary[i,j] = np.sum(unary_temp[i][j])

        all_unary_list.append(torch.from_numpy(all_unary[i]).float())
   
    return all_unary_list

def rpsm(volumes_pred, keypoints_3d_pred, keypoints_3d_gt, coord_volumes_pred):
    # initialization
    rec_depth = 8
    njoints = keypoints_3d_gt.shape[0]
    nbins = 2
    device = torch.device('cpu')

    # computation of the limb lengths ground truth
    body = HumanBody()
    skeleton = body.skeleton
    limb_length_gt = compute_limb_length(keypoints_3d_gt, skeleton)

    grid_size = np.linalg.norm(coord_volumes_pred[0][0][0] - coord_volumes_pred[0][0][1]).round()

    pose3d = keypoints_3d_pred
    # pose3d = np.zeros((njoints,3), dtype=np.float64)
    # for i in range(njoints):
    #     ind = np.unravel_index(np.argmax(volumes_pred[i], axis=None), volumes_pred[i].shape)
    #     pose3d[i] = coord_volumes_pred[ind[0], ind[1], ind[2]]
    
    for k in range(rec_depth):
        grids = []
        tolerance = 0.1*np.multiply(np.sqrt(2), grid_size)

        for i in range(njoints):
            grids.append(compute_grid(grid_size, pose3d[i], nbins, device=device))

        pairwise = compute_pairwise(skeleton, limb_length_gt, grids, tolerance)
        unary = compute_unary_term(volumes_pred, coord_volumes_pred, grids)
        pose3d_cube, states_of_all_joints = infer(unary, pairwise, body)
        pose3d = get_loc_from_states_of_joints(states_of_all_joints, grids)
        # pose3d = get_loc_from_cube_idx(grids, pose3d_cube)

        grid_size = grid_size / 2
    
    return pose3d

if __name__=='__main__':
    with open('rpsm.pkl','rb') as  dict_data:
        data = pickle.load(dict_data)


    mpjpe_null = []
    batch_size = 10
    for i in range(batch_size):
        coord_volumes_pred = data['coord_volumes_pred'][i]
        keypoints_3d_gt = data['keypoints_3d_gt'][i]
        keypoints_3d_pred = data['keypoints_3d_pred'][i]
        volumes_pred = data['volumes_pred'][i]

        # Scale the volumes_pred (between 0 - 10) which will be used for computation of unary potential
        volumes_pred = (volumes_pred/np.max(volumes_pred))*5

        pose3d = rpsm(volumes_pred, keypoints_3d_pred, keypoints_3d_gt, coord_volumes_pred).numpy()

        mpjpe_pred = (np.sqrt(np.sum(((keypoints_3d_gt) - (pose3d))**2, axis=1)))
        mpjpe_pose = (np.sqrt(np.sum(((keypoints_3d_gt) - (keypoints_3d_pred))**2, axis=1)))
        mpjpe_null.append(np.sqrt(np.sum(((pose3d) - (keypoints_3d_pred))**2, axis=1)))

        mpjpe1 = np.mean(np.sqrt(np.sum(((keypoints_3d_gt - keypoints_3d_gt[6]) - (pose3d-pose3d[6]))**2, axis=1)))
        mpjpe2 = np.mean(np.sqrt(np.sum(((keypoints_3d_gt - keypoints_3d_gt[6]) - (keypoints_3d_pred-keypoints_3d_pred[6]))**2, axis=1)))

        mpjpe3 = (np.sqrt(np.sum(((keypoints_3d_gt[6]) - (pose3d[6]))**2)))
        mpjpe4 = (np.sqrt(np.sum(((keypoints_3d_gt[6]) - (keypoints_3d_pred[6]))**2)))

        mpjpe5 = np.mean(np.sqrt(np.sum((keypoints_3d_gt - pose3d)**2, axis=1)))
        mpjpe6 = np.mean(np.sqrt(np.sum((keypoints_3d_gt - keypoints_3d_pred)**2, axis=1)))
        
        print(mpjpe1, mpjpe2, '          ', mpjpe3, mpjpe4, '          ', mpjpe5, mpjpe6)
        
        # body = HumanBody()
        # skeleton = body.skeleton
        # header = []
        # for i in range(17):
        #     header.append(skeleton[i]['name'])
        # table = [[mpjpe_pose.reshape(1,-1)],[mpjpe_pred.reshape(1,-1)]]
        # print(tabulate(table, header, tablefmt='orgtbl'))