from typing import *
import numpy as np
import torch
import utils3d
import nvdiffrast.torch as dr
from tqdm import tqdm
import comfy.utils
import xatlas
import pyvista as pv
from pymeshfix import _meshfix
import igraph
import cv2
from .random_utils import sphere_hammersley_sequence
from .render_utils import render_multiview
from ..representations import Strivec, Gaussian, MeshExtractResult

def safe_call(func, **kwargs):
    """Attempts to call a function by matching provided kwargs to its expected names."""
    import inspect
    sig = inspect.signature(func)
    params = sig.parameters
    
    # Map common aliases
    aliases = {
        'at': ['target', 'center', 'lookat'],
        'eye': ['origin', 'camera_pos', 'pos'],
        'up': ['up_vector'],
        'extrinsics': ['matrix', 'm'],
        'vertices': ['verts', 'v'],
        'faces': ['indices', 'f'],
        'context': ['ctx']
    }
    
    final_kwargs = {}
    for key, value in kwargs.items():
        if key in params:
            final_kwargs[key] = value
        elif key in aliases:
            for alias in aliases[key]:
                if alias in params:
                    final_kwargs[alias] = value
                    break
    
    # If still missing some params, just try passing original kwargs as fallback
    try:
        return func(**final_kwargs)
    except TypeError:
        return func(**kwargs)

def safe_get_projection(fov=None, intrinsics=None, near=0.1, far=10.0):
    device = fov.device if fov is not None else (intrinsics.device if torch.is_tensor(intrinsics) else 'cuda')
    if fov is not None:
        for func_name in ['perspective_from_fov', 'fov_to_perspective']:
            func = getattr(utils3d.torch, func_name, None)
            if func:
                try: return func(fov=fov, aspect=1.0, near=near, far=far)
                except: continue
    if intrinsics is not None:
        intr_t = torch.tensor(intrinsics).to(device) if not torch.is_tensor(intrinsics) else intrinsics
        for func_name in ['perspective_from_intrinsics', 'intrinsics_to_perspective', 'perspective_from_intrinsic']:
            func = getattr(utils3d.torch, func_name, None)
            if func:
                try: return func(intrinsics=intr_t, near=near, far=far)
                except: continue
    return torch.eye(4, device=device)

def safe_rasterize(rastctx, verts, faces, w, h, uv=None, view=None, projection=None):
    for name in ['rasterize_triangles', 'rasterize_triangle_faces', 'rasterize']:
        func = getattr(utils3d.torch, name, None)
        if func is not None:
            kwargs = {'context': rastctx, 'vertices': verts, 'faces': faces, 'width': w, 'height': h, 'view': view, 'projection': projection}
            try:
                if uv is not None: return safe_call(func, **kwargs, uv=uv)
                return safe_call(func, **kwargs)
            except: continue
    raise AttributeError("Could not find a valid rasterization function.")

@torch.no_grad()
def _fill_holes(verts, faces, max_hole_size=0.04, max_hole_nbe=32, resolution=128, num_views=500, debug=False, verbose=False):
    yaws, pitchs = [], []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    yaws, pitchs = torch.tensor(yaws).cuda(), torch.tensor(pitchs).cuda()
    radius, fov = 2.0, torch.deg2rad(torch.tensor(40.0)).cuda()
    projection = safe_get_projection(fov=fov, near=1.0, far=3.0)
    
    views = []
    for (yaw, pitch) in zip(yaws, pitchs):
        orig = torch.tensor([torch.sin(yaw) * torch.cos(pitch), torch.cos(yaw) * torch.cos(pitch), torch.sin(pitch)]).cuda().float() * radius
        # 🛡️ Using safe_call for look_at
        view = safe_call(utils3d.torch.view_look_at, eye=orig, at=torch.tensor([0, 0, 0]).float().cuda(), up=torch.tensor([0, 0, 1]).float().cuda())
        views.append(view)
    views = torch.stack(views, dim=0)

    visblity = torch.zeros(faces.shape[0], dtype=torch.int32, device=verts.device)
    rastctx = utils3d.torch.RastContext(backend='cuda')
    for i in tqdm(range(views.shape[0]), total=views.shape[0], disable=not verbose, desc='Rasterizing'):
        buffers = safe_rasterize(rastctx, verts[None], faces, resolution, resolution, view=views[i], projection=projection)
        face_id = torch.unique(buffers['face_id'][0][buffers['mask'][0] > 0.95] - 1).long()
        visblity[face_id] += 1
    
    visblity = visblity.float() / num_views
    edges, face2edge, edge_degrees = utils3d.torch.compute_edges(faces)
    # 🛡️ Using safe_call for connected components
    connected_components = safe_call(utils3d.torch.compute_connected_components, faces=faces, edges=edges, face2edge=face2edge)
    outer_face_indices = torch.zeros(faces.shape[0], dtype=torch.bool, device=faces.device)
    for cc in connected_components:
        outer_face_indices[cc] = visblity[cc] > min(max(visblity[cc].quantile(0.75).item(), 0.25), 0.5)
    
    inner_face_indices = torch.nonzero(visblity == 0).reshape(-1)
    if inner_face_indices.shape[0] == 0: return verts, faces
    dual_edges, _ = safe_call(utils3d.torch.compute_dual_graph, face2edge=face2edge)
    g = igraph.Graph(); g.add_vertices(faces.shape[0]); g.add_edges(dual_edges.cpu().numpy())
    g.add_vertex('s'); g.add_vertex('t')
    g.add_edges([(f, 's') for f in inner_face_indices.cpu().numpy()])
    g.add_edges([(f, 't') for f in outer_face_indices.nonzero().reshape(-1).cpu().numpy()])
    cut = g.mincut('s', 't')
    remove_face_indices = torch.tensor([v for v in cut.partition[0] if v < faces.shape[0]], dtype=torch.long, device=faces.device)
    mask = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
    mask[remove_face_indices] = 0
    faces = faces[mask]
    faces, verts = safe_call(utils3d.torch.remove_unreferenced_vertices, faces=faces, vertices=verts)
            
    mesh = _meshfix.PyTMesh()
    mesh.load_array(verts.cpu().numpy(), faces.cpu().numpy())
    mesh.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    v_new, f_new = mesh.return_arrays()
    return torch.tensor(v_new, device='cuda', dtype=torch.float32), torch.tensor(f_new, device='cuda', dtype=torch.int32)

def postprocess_mesh(vertices, faces, simplify=True, simplify_ratio=0.9, fill_holes=True, **kwargs):
    if simplify and simplify_ratio > 0:
        cells = np.concatenate([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces.astype(np.int32)], axis=1)
        mesh = pv.PolyData(vertices.astype(np.float32), cells).decimate(simplify_ratio)
        vertices, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
    if fill_holes:
        v_t, f_t = torch.tensor(vertices).cuda(), torch.tensor(faces.astype(np.int32)).cuda()
        v_t, f_t = _fill_holes(v_t, f_t, verbose=kwargs.get('verbose', False))
        vertices, faces = v_t.cpu().numpy(), f_t.cpu().numpy()
    return vertices, faces

def bake_texture(vertices, faces, uvs, observations, masks, extrinsics, intrinsics, texture_size=2048, near=0.1, far=10.0, verbose=False):
    v, f, uv_t = torch.tensor(vertices).cuda(), torch.tensor(faces.astype(np.int32)).cuda(), torch.tensor(uvs).cuda()
    obs_t = [torch.tensor(o / 255.0).float().cuda() for o in observations]
    masks_t = [torch.tensor(m>0).bool().cuda() for m in masks]
    # 🛡️ Using safe_call for extrinsics
    views = [safe_call(utils3d.torch.extrinsics_to_view, extrinsics=torch.tensor(e).cuda()) for e in extrinsics]
    projs = [safe_get_projection(intrinsics=i, near=near, far=far) for i in intrinsics]
    rastctx = utils3d.torch.RastContext(backend='cuda')
    tex, wts = torch.zeros((texture_size**2, 3), device='cuda'), torch.zeros(texture_size**2, device='cuda')
    for i in range(len(views)):
        rast = safe_rasterize(rastctx, v[None], f, obs_t[i].shape[1], obs_t[i].shape[0], uv=uv_t[None], view=views[i], projection=projs[i])
        m = rast['mask'][0].bool() & masks_t[i]
        uv_map = (rast['uv'][0].flip(0) * texture_size).floor().long()[m]
        idx = uv_map[:, 0] + (texture_size - uv_map[:, 1] - 1) * texture_size
        tex.scatter_add_(0, idx.view(-1, 1).expand(-1, 3), obs_t[i][m])
        wts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    mask = wts > 0
    tex[mask] /= wts[mask][:, None]
    final = np.clip(tex.reshape(texture_size, texture_size, 3).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    return cv2.inpaint(final, (wts == 0).cpu().numpy().astype(np.uint8).reshape(texture_size, texture_size), 3, cv2.INPAINT_TELEA)

def finalize_mesh(app_rep, mesh, simplify=0.95, fill_holes=True, texture_size=1024, verbose=True):
    v, f = mesh.vertices.detach().cpu().numpy(), mesh.faces.detach().cpu().numpy()
    v, f = postprocess_mesh(v, f, simplify=simplify > 0, simplify_ratio=simplify, fill_holes=fill_holes, verbose=verbose)
    vmapping, indices, uvs = xatlas.parametrize(v, f)
    v, f, uvs = v[vmapping], indices, uvs
    obs, extr, intr = render_multiview(app_rep, resolution=1024, nviews=100)
    masks = [np.any(o > 0, axis=-1) for o in obs]
    texture = bake_texture(v, f, uvs, obs, masks, [e.cpu().numpy() for e in extr], [i.cpu().numpy() for i in intr], texture_size=texture_size, verbose=verbose)
    uvs[:, 1] = 1 - uvs[:, 1]
    return v @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), f, uvs, texture.astype(np.float32) / 255