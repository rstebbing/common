##########################################
# File: vtk_.py                          #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import numpy as np
from itertools_ import pairwise

import vtk

# _SetInput_or_SetInputData
def _SetInput_or_SetInputData(obj, poly_data):
    if hasattr(obj, 'SetInput'):
        return obj.SetInput(poly_data)
    else:
        return obj.SetInputData(poly_data)

# make_vtkPoints
def make_vtkPoints(P):
    P = np.atleast_2d(P)
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(P.shape[0])
    for i, p in enumerate(P):
        points.SetPoint(i, *p)
    return points

# make_vtkCellArray
def make_vtkCellArray(T):
    cell_array = vtk.vtkCellArray()
    for t in T:
        cell_array.InsertNextCell(len(t))
        for i in t:
            cell_array.InsertCellPoint(i)
    return cell_array

# _apply_methods
def _apply_methods(obj, *args):
    for k, a in args:
        try:
            iter(a)
        except TypeError:
            a = (a,)
        getattr(obj, k)(*a)

# tubes
def tubes(T, P, *args):
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(make_vtkPoints(P))
    poly_data.SetLines(make_vtkCellArray(T))

    tube = vtk.vtkTubeFilter()
    _SetInput_or_SetInputData(tube, poly_data)
    _apply_methods(tube, *args)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor._locals = locals()
    return actor

# mesh
def mesh(F, P, *args):
    mesh = []
    for f in F:
        for i, j in pairwise(f, repeat=True):
            if j > i:
                i, j = j, i
            mesh.append((i, j))
    mesh = set(mesh)
    return tubes(mesh, P, *args)

# points
def points(P, *args):
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(make_vtkPoints(P))

    sphere = vtk.vtkSphereSource()
    _apply_methods(sphere, *args)

    glyph = vtk.vtkGlyph3D()
    _SetInput_or_SetInputData(glyph, poly_data)
    glyph.SetSourceConnection(sphere.GetOutputPort())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor._locals = locals()
    return actor

# surface
def surface(T, P, use_normals=True,
                 camera=None,
                 face_colours=None,
                 face_label_function_and_lut=None):
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(make_vtkPoints(P))
    poly_data.SetPolys(make_vtkCellArray(T))

    mapper = vtk.vtkPolyDataMapper()

    # `colour_faces_filter` is empty by default.
    colour_faces_filter = None

    if face_colours is not None:
        assert len(face_colours) == poly_data.GetNumberOfPolys()

        # Build `lut`.
        npy_face_colours = np.require(np.atleast_2d(face_colours), np.float64)
        face_colours = map(tuple, face_colours)
        unique_colours = sorted(set(face_colours))
        num_unique_colours = len(unique_colours)

        lut = make_vtkLookupTable(unique_colours)
        mapper.SetLookupTable(lut)

        # Set `lookup_indices`.
        B = npy_face_colours[:, np.newaxis, :] == unique_colours
        i, j = np.nonzero(np.all(B, axis=2))
        lookup_indices = np.empty(len(face_colours), dtype=j.dtype)
        lookup_indices[i] = j

        # Set `lookup`.
        vtk_lookup = vtk.vtkFloatArray()
        vtk_lookup.SetNumberOfValues(len(face_colours))
        lookup = vtk_to_numpy(vtk_lookup)
        lookup[:] = lookup_indices.astype(np.float64) / num_unique_colours

        poly_data.GetCellData().SetScalars(vtk_lookup)
    elif face_label_function_and_lut is not None:
        label_function, lut = face_label_function_and_lut

        # Set `lut` for the mapper.
        lut = make_vtkLookupTable(lut)
        mapper.SetLookupTable(lut)

        # Initialise `colour_faces_filter`.
        colour_faces_filter = make_colour_faces_filter(label_function)

    pipeline = [poly_data]
    def add_to_pipeline(to):
        from_ = pipeline[-1]
        if hasattr(from_, 'GetOutputPort'):
            to.SetInputConnection(from_.GetOutputPort())
        else:
            _SetInput_or_SetInputData(to, from_)
        pipeline.append(to)

    if colour_faces_filter is not None:
        add_to_pipeline(colour_faces_filter)

    if camera is not None:
        poly_data_depth = vtk.vtkDepthSortPolyData()
        poly_data_depth.SetCamera(camera)
        poly_data_depth.SetDirectionToBackToFront()
        poly_data_depth.SetDepthSortModeToBoundsCenter()
        add_to_pipeline(poly_data_depth)

    if use_normals:
        normals = vtk.vtkPolyDataNormals()
        add_to_pipeline(normals)

    add_to_pipeline(mapper)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor._locals = locals()
    return actor

# renderer
def renderer(*actors):
    ren = vtk.vtkRenderer()
    ren.SetBackground(1.0, 1.0, 1.0)
    for actor in actors:
        ren.AddActor(actor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(ren)
    render_window.SetSize(540, 480)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(render_window)

    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetCurrentRenderer(ren)
    iren.SetInteractorStyle(style)
    iren._locals = locals()

    return ren, iren

# KeyPressCallbackStyle
class KeyPressCallbackStyle(object):
    def __init__(self):
        self.KeyPressActivationOff()
        self.AddObserver('KeyPressEvent', self.OnKeyPress)
        self._callbacks = {}

    def add_callback(self, key, function):
        self._callbacks.setdefault(key, []).append(function)

    def on_key_press(self, key):
        for f in self._callbacks.get(key, []):
            f()

        for f in self._callbacks.get(None, []):
            f(key)

        vtk.vtkInteractorStyleTrackballCamera.OnKeyPress(self)

    def OnKeyPress(self, obj, event):
        rwi = self.GetInteractor()
        key = rwi.GetKeySym()
        return self.on_key_press(key)

# _InteractorStyle_factory
def _InteractorStyle_factory(new_class):
    def make(base_class):
        return type(base_class)(new_class.__name__,
                                (base_class,),
                                dict(new_class.__dict__))
    make.func_name == 'make_%s' % new_class.__name__
    return make

# make_KeyPressCallbackStyle
make_KeyPressCallbackStyle = _InteractorStyle_factory(KeyPressCallbackStyle)
