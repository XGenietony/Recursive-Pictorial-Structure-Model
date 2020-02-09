import pickle
import numpy as np
import torch
from tabulate import tabulate
from body import HumanBody

def infer(unary, body, limb_length_gt, grids, tolerance):
    skeleton_sorted_by_level = body.skeleton_sorted_by_level
    njoints = len(skeleton_sorted_by_level)

    pose3d = np.zeros((njoints, 3), dtype=np.float64)
    root_index = 6
    root_node = skeleton_sorted_by_level[0]
    u = unary[root_node['idx']].clone()
    pose3d[root_node['idx']] = get_loc_from_potential(u, grids[root_node['idx']])

    for node in skeleton_sorted_by_level:
        if node['idx'] != root_index:
            u = unary[node['idx']].clone()
            p = compute_pairwise(pose3d[node['parent']], grids[node['idx']], limb_length_gt[(node['parent'], node['idx'])], tolerance)
            potential = torch.mul(p,u)
            pose3d[node['idx']] = get_loc_from_potential(potential, grids[node['idx']])

    return pose3d

def  get_loc_from_potential(potential, grid):
    energy = torch.nn.functional.softmax(potential.view(-1, 1), dim=0)
    energy = energy.expand_as(grid)
    loc3d  = torch.sum(energy*grid, dim=0)
    return loc3d

def compute_limb_length(keypoints_3d_gt, skeleton):
    limb_length_gt = {}
    for node in skeleton:
        current = node['idx']
        children = node['children']
        for child in children:
            limb_length_gt[(current, child)] = np.linalg.norm(keypoints_3d_gt[current] - keypoints_3d_gt[child])
    return limb_length_gt

def probabilty_function(x, sigma, mu):
    return np.exp((-1/2)*(((x-mu)/sigma)**2))/(sigma*np.sqrt(2*np.pi))

def compute_pairwise(pose3d, grid, limb_length_gt, tolerance):
    distance = torch.sqrt(torch.sum((torch.from_numpy(pose3d).view(-1,3).expand_as(grid) - grid)**2, axis=1))
    expect_length = limb_length_gt
    probability = probabilty_function(distance - expect_length, tolerance, 0)
    pairwise = probability
    return pairwise*2000
    
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

    pose3d = keypoints_3d_pred.copy()
    # pose3d[6] = keypoints_3d_gt[6].copy()
    
    for k in range(rec_depth):
        grids = []
        tolerance = np.multiply(np.sqrt(2), grid_size)
        tolerance = grid_size

        for i in range(njoints):
            grids.append(compute_grid(grid_size, pose3d[i], nbins, device=device))

        unary = compute_unary_term(volumes_pred, coord_volumes_pred, grids)
        pose3d = infer(unary, body, limb_length_gt, grids, tolerance)

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

        # volumes_pred = (volumes_pred/np.max(volumes_pred))

        keypoints_3d_pose3d = rpsm(volumes_pred, keypoints_3d_pred, keypoints_3d_gt, coord_volumes_pred)

        mpjpe_pred = (np.sqrt(np.sum(((keypoints_3d_gt) - (keypoints_3d_pose3d))**2, axis=1)))
        mpjpe_pose = (np.sqrt(np.sum(((keypoints_3d_gt) - (keypoints_3d_pred))**2, axis=1)))
        mpjpe_null.append(np.sqrt(np.sum(((keypoints_3d_pose3d) - (keypoints_3d_pred))**2, axis=1)))

        body = HumanBody()
        skeleton = body.skeleton
        limb_length_pose = compute_limb_length(keypoints_3d_pose3d, skeleton)
        limb_length_pred = compute_limb_length(keypoints_3d_pred, skeleton)
        limb_length_gt = compute_limb_length(keypoints_3d_gt, skeleton)

        count = 0
        comp_length_pose = np.zeros((16,), dtype=np.float16)
        comp_length_pred = np.zeros((16,), dtype=np.float16)
        for node in skeleton:
            current = node['idx']
            children = node['children']
            for child in children:
                comp_length_pose[count] = np.abs(limb_length_pose[(current, child)] - limb_length_gt[(current, child)])
                comp_length_pred[count] = np.abs(limb_length_pred[(current, child)] - limb_length_gt[(current, child)])
                count = count + 1
        comp_length_pose = np.mean(comp_length_pose)
        comp_length_pred = np.mean(comp_length_pred)

        mpjpe1 = np.mean(np.sqrt(np.sum(((keypoints_3d_gt - keypoints_3d_gt[6]) - (keypoints_3d_pose3d-keypoints_3d_pose3d[6]))**2, axis=1)))
        mpjpe2 = np.mean(np.sqrt(np.sum(((keypoints_3d_gt - keypoints_3d_gt[6]) - (keypoints_3d_pred-keypoints_3d_pred[6]))**2, axis=1)))

        mpjpe3 = (np.sqrt(np.sum(((keypoints_3d_gt[6]) - (keypoints_3d_pose3d[6]))**2)))
        mpjpe4 = (np.sqrt(np.sum(((keypoints_3d_gt[6]) - (keypoints_3d_pred[6]))**2)))

        mpjpe5 = np.mean(np.sqrt(np.sum((keypoints_3d_gt - keypoints_3d_pose3d)**2, axis=1)))
        mpjpe6 = np.mean(np.sqrt(np.sum((keypoints_3d_gt - keypoints_3d_pred)**2, axis=1)))
        
        print(comp_length_pose, comp_length_pred, '              ', mpjpe1, mpjpe2, '          ', mpjpe3, mpjpe4, '          ', mpjpe5, mpjpe6)