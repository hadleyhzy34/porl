import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import einsum
# from einops import rearrange, repeat


# from pointnet2_ops import pointnet2_utils


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def fps_2d(xy, npoint):
    """
    Input:
        xyz: pointcloud data, [N, 2]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    device = xy.device
    N, C = xy.shape
    centroids = torch.zeros(npoint, dtype=torch.long).to(device)  #(n,)
    distance = torch.ones(N).to(device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xy[farthest, :].view(1, 2)
        dist = torch.sum((xy - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def batch_farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, 362]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N = xyz.shape

    xy_points = torch.zeros((B,npoint,3),device=device)
    idx_points = torch.zeros((B,npoint),dtype=torch.long,device=device)

    for i in range(B):
        points = xyz[i]  #(362,)
        rad_points = torch.zeros((361, 2))  #(361,2)
        rad_points[:360,0] = torch.cos((torch.arange(0,360)) * torch.pi / 180) * points[:360]
        rad_points[:360,1] = torch.sin((torch.arange(0,360)) * torch.pi / 180) * points[:360]
        rad_points[360,0] = points[-2]
        rad_points[360,1] = points[-1]
        rad_points[360,2] = 1.
        rad_points = rad_points[points<8.,:]  #(n,2)
        idx_points[i] = fps_2d(rad_points,npoint)  #(np,)
        xy_points[i] = rad_points
    return xy_points, idx_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(agent, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, agent).__init__()
        agent.groups = groups
        agent.kneighbors = kneighbors
        agent.use_xyz = use_xyz
        if normalize is not None:
            agent.normalize = normalize.lower()
        else:
            agent.normalize = None
        if agent.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (agent.normalize), set to None. Should be one of [center, anchor].")
            agent.normalize = None
        if agent.normalize is not None:
            add_channel=3 if agent.use_xyz else 0
            agent.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            agent.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(agent, xyz, points):
        B, N, C = xyz.shape
        S = agent.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=agent.groups, replacement=False).long()
        fps_idx = farthest_point_sample(xyz, agent.groups).long()
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz, agent.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(agent.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if agent.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if agent.normalize is not None:
            if agent.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if agent.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if agent.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = agent.affine_alpha*grouped_points + agent.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, agent.kneighbors, 1)], dim=-1)
        return new_xyz, new_points

    def first_forward(agent, xyz, points):
        B, N, C = xyz.shape
        S = agent.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=agent.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, agent.groups).long()
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz, agent.groups).long()  # [B, npoint]
        # new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        # new_points = index_points(points, fps_idx)  # [B, npoint, d]
        new_xyz, fps_idx = batch_farthest_point_sample(xyz, points)
        new_points = index_points(points, fps_idx)

        idx = knn_point(agent.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if agent.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if agent.normalize is not None:
            if agent.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if agent.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if agent.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = agent.affine_alpha*grouped_points + agent.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, agent.kneighbors, 1)], dim=-1)
        return new_xyz, new_points

class ConvBNReLU1D(nn.Module):
    def __init__(agent, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, agent).__init__()
        agent.act = get_activation(activation)
        agent.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            agent.act
        )

    def forward(agent, x):
        return agent.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(agent, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, agent).__init__()
        agent.act = get_activation(activation)
        agent.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            agent.act
        )
        if groups > 1:
            agent.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                agent.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            agent.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(agent, x):
        return agent.act(agent.net2(agent.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(agent, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, agent).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        agent.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        agent.operation = nn.Sequential(*operation)

    def forward(agent, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = agent.transfer(x)
        batch_size, _, _ = x.size()
        x = agent.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(agent, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, agent).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        agent.operation = nn.Sequential(*operation)

    def forward(agent, x):  # [b, d, g]
        return agent.operation(x)


class PointMLPModel(nn.Module):
    def __init__(agent, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        super(PointMLPModel, agent).__init__()
        agent.stages = len(pre_blocks)
        agent.class_num = class_num
        agent.points = points
        agent.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        agent.local_grouper_list = nn.ModuleList()
        agent.pre_blocks_list = nn.ModuleList()
        agent.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = agent.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            agent.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            agent.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            agent.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        agent.act = get_activation(activation)
        agent.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            agent.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            agent.act,
            nn.Dropout(0.5),
            nn.Linear(256, agent.class_num)
        )

    def forward(agent, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = agent.embedding(x)  # B,D,N
        for i in range(agent.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = agent.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = agent.pre_blocks_list[i](x)  # [b,d,g]
            x = agent.pos_blocks_list[i](x)  # [b,d,g]

        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = agent.classifier(x)
        return x

class Model(nn.Module):
    def __init__(agent, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        super(Model, agent).__init__()
        agent.stages = len(pre_blocks)
        agent.class_num = class_num
        agent.points = points
        agent.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        agent.local_grouper_list = nn.ModuleList()
        agent.pre_blocks_list = nn.ModuleList()
        agent.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = agent.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            agent.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            agent.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            agent.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        agent.act = get_activation(activation)
        agent.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            agent.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            agent.act,
            nn.Dropout(0.5),
            nn.Linear(256, agent.class_num)
        )

    # def transform(agent, state):
    #     """
    #     Description:
    #     args:
    #         state: (b,362)
    #     return:
    #         points: (b,3,n)
    #     """
    #     goal = 
    #
    def forward(agent, x):
        pdb.set_trace()
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = agent.embedding(x)  # B,D,N
        for i in range(agent.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = agent.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = agent.pre_blocks_list[i](x)  # [b,d,g]
            x = agent.pos_blocks_list[i](x)  # [b,d,g]

        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = agent.classifier(x)
        return x



def pointMLP(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                   k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs)


def pointMLPElite(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=32, groups=1, res_expansion=0.25,
                   activation="relu", bias=False, use_xyz=False, normalize="anchor",
                   dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                   k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2], **kwargs)

if __name__ == '__main__':
    data = torch.rand(2, 3, 1024)
    print("===> testing pointMLP ...")
    model = pointMLP()
    out = model(data)
    print(out.shape)

