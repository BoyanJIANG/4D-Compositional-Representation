from lib.encoder import conv, pointnet, mesh_encoder, pointnet_multiscale_v2 as pointnet2


encoder_dict = {
    'resnet18': conv.Resnet18,
    'pointnet_simple': pointnet.SimplePointnet,
    'pointnet_resnet': pointnet.ResnetPointnet,
    'pointnet_resnet2': pointnet2.ResnetPointnet,
    'mesh_encoder': mesh_encoder.SpiralEncoder,
    # 'LocalPoolPointnet': conv_onet.LocalPoolPointnet
}


encoder_temporal_dict = {
    'conv_3D': conv.ConvEncoder3D,
    'pointnet_2stream': pointnet.ResnetPointnet2Stream,
    'pointnet_resnet': pointnet.TemporalResnetPointnet,
    'pointnet_resnet2': pointnet2.TemporalResnetPointnet,
    'mesh_encoder': mesh_encoder.SpiralEncoder,
    # 'LocalPoolPointnet': conv_onet.LocalPoolPointnet
}