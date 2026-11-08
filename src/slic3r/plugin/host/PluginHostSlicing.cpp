#include "PluginHostBindings.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>
#include "slic3r/plugin/PluginBindingUtils.hpp"

#include "libslic3r/BoundingBox.hpp"
#include "libslic3r/ExPolygon.hpp"
#include "libslic3r/Surface.hpp"
#include "libslic3r/SurfaceCollection.hpp"
#include "libslic3r/ExtrusionEntity.hpp"
#include "libslic3r/ExtrusionEntityCollection.hpp"
#include "libslic3r/Layer.hpp"      // LayerRegion, Layer, SupportLayer
#include "libslic3r/Print.hpp"      // PrintRegion, PrintObject, Print

#include "libslic3r/Point.hpp"
#include "libslic3r/libslic3r.h"
#include <pybind11/stl.h>
#include <algorithm> // std::all_of (append_path level check)
#include <memory>
#include <string>
#include <utility>   // std::move (set_points / append_path)
#include <vector>
#include "libslic3r/Config.hpp"
#include "libslic3r/ObjectID.hpp"
#include "libslic3r/Polyline.hpp"
#include "libslic3r/PrintConfig.hpp"
#include <pybind11/detail/common.h>
#include <pybind11/detail/descr.h>
#include <pybind11/numpy.h>

namespace Slic3r { class Model; }

namespace py = pybind11;

namespace Slic3r {
namespace {
// Flatten an extrusion graph into a list of leaf ExtrusionPath* while walking the
// ORIGINAL Print-owned tree (never a temporary copy): the returned pointers stay
// valid for the execute(ctx) lifetime pinned by `owner`, so points() can hand out
// zero-copy views into path->polyline.points.
//
// This is deliberately NOT ExtrusionEntityCollection::flatten(): flatten() only
// unwraps nested collections (is_collection() is true solely for collections) and
// returns them by value, so it would (a) dangle if we viewed into the copy and
// (b) leave ExtrusionLoop/ExtrusionMultiPath intact — dropping every perimeter
// loop, since dynamic_cast<ExtrusionPath*> fails on a loop. We descend into
// loops/multipaths here to reach their contained paths.
static void collect_extrusion_paths(const ExtrusionEntity* ee, std::vector<const ExtrusionPath*>& out)
{
    if (ee == nullptr)
        return;
    if (const auto* coll = dynamic_cast<const ExtrusionEntityCollection*>(ee)) {
        for (const ExtrusionEntity* child : coll->entities)
            collect_extrusion_paths(child, out);
    } else if (const auto* loop = dynamic_cast<const ExtrusionLoop*>(ee)) {
        for (const ExtrusionPath& p : loop->paths)
            out.push_back(&p);
    } else if (const auto* mp = dynamic_cast<const ExtrusionMultiPath*>(ee)) {
        for (const ExtrusionPath& p : mp->paths)
            out.push_back(&p);
    } else if (const auto* path = dynamic_cast<const ExtrusionPath*>(ee)) {
        // Catches ExtrusionPath and its subclasses (Sloped/Contoured/Oriented) last,
        // after the composite types above have been ruled out.
        out.push_back(path);
    }
}

// Rebuild a layer's per-island bbox cache from lslices — the same inline pattern
// every C++ call site uses (PrintObjectSlice.cpp, Print.cpp, TreeSupport.cpp); no
// libslic3r helper exists to reuse.
static void refresh_lslices_bboxes(Layer& l)
{
    l.lslices_bboxes.clear();
    l.lslices_bboxes.reserve(l.lslices.size());
    for (const ExPolygon& island : l.lslices)
        l.lslices_bboxes.emplace_back(get_extents(island));
}

// One (N,3) int64 ndarray -> Points3, the vertex buffer of an ExtrusionPath polyline.
// Same contract as parse_polygon() in PluginHostGeometry.cpp. That helper is file-local
// there, and moving it into a header would force a long rebuild, so the few lines are
// repeated here. The checks reject malformed input up front, before the toolpath changes:
//  - Only a signed 8-byte integer dtype (coord_t == int64) is accepted. A float array is
//    almost always a units slip (millimeters instead of scaled coords), and integer arrays
//    cannot hold NaN or inf, so neither can reach the G-code writer.
//  - The shape must be (N,3): the x, y, z columns that ExtrusionPath.points() hands out.
//  - N >= 2: a path is a chain of segments, and the G-code writer reads the first and the
//    last point of every path without checking that they exist.
static Points3 parse_points3(py::handle h, const char* who)
{
    if (!py::isinstance<py::array>(h))
        throw py::value_error(std::string(who) + ": points must be an (N,3) int64 ndarray");
    py::array a = py::reinterpret_borrow<py::array>(h);
    if (a.dtype().kind() != 'i' || a.itemsize() != (py::ssize_t) sizeof(coord_t))
        throw py::value_error(std::string(who) + ": points must be int64 (scaled coords)");
    if (a.ndim() != 2 || a.shape(1) != 3)
        throw py::value_error(std::string(who) + ": points must have shape (N,3)");
    if (a.shape(0) < 2)
        throw py::value_error(std::string(who) + ": a path needs at least 2 points");
    // The dtype is already checked as int64. forcecast here only guarantees a C-contiguous
    // buffer: a strided view such as pts[::2] is a valid input but not contiguous memory.
    auto arr = py::array_t<coord_t, py::array::c_style | py::array::forcecast>::ensure(a);
    if (!arr)
        throw py::value_error(std::string(who) + ": could not read points as a contiguous int64 array");
    auto r = arr.unchecked<2>();
    Points3 pts;
    pts.reserve((size_t) arr.shape(0));
    for (py::ssize_t i = 0; i < arr.shape(0); ++i)
        pts.emplace_back((coord_t) r(i, 0), (coord_t) r(i, 1), (coord_t) r(i, 2));
    return pts;
}
} // namespace

void host_bindings::register_slicing(py::module_& host)
{
    // ------------------------------------------------------------------
    // Slicing print-graph data model — raw bindings of the classes the C++
    // pipeline itself uses, same nodelete/reference style as the Model and
    // Preset graphs in PluginHostModel.cpp / PluginHostPresets.cpp.
    //
    // LIFETIME (C++ semantics, the one rule of this API): every object handed
    // out below is a non-owning reference into the live slicing graph owned by
    // the Print. References — and every numpy view they hand out — are valid
    // only while the plugin hook (execute(ctx)) runs, and a container-replacing
    // mutator (SurfaceCollection.set / append / clear, Polygon.set_points / append,
    // ExPolygon.set_holes, ExtrusionPath.set_points, ExtrusionEntityCollection.remove /
    // clear) invalidates previously obtained references into that container, exactly
    // as std::vector operations invalidate C++ iterators. Do not stash references or
    // arrays across execute() calls; copy what you need.
    // ------------------------------------------------------------------

    py::enum_<SurfaceType>(host, "SurfaceType")
        .value("stTop", stTop)
        .value("stBottom", stBottom)
        .value("stBottomBridge", stBottomBridge)
        .value("stInternalAfterExternalBridge", stInternalAfterExternalBridge)
        .value("stInternal", stInternal)
        .value("stInternalSolid", stInternalSolid)
        .value("stInternalBridge", stInternalBridge)
        .value("stSecondInternalBridge", stSecondInternalBridge)
        .value("stInternalVoid", stInternalVoid)
        .value("stPerimeter", stPerimeter)
        .value("stCount", stCount)
        .export_values();

    // Surface: default holder (Python-owned instances are freed), so plugins can construct
    // their own Surface(surface_type, expolygon) — not just navigate the live slicing graph.
    // expolygon is a reference_internal property, same idiom as the Polygon/ExPolygon
    // accessors in PluginHostGeometry.cpp.
    py::class_<Surface>(host, "Surface")
        .def(py::init([](SurfaceType t, const ExPolygon& e) { return Surface(t, e); }),
             py::arg("surface_type"), py::arg("expolygon"))
        .def(py::init([](SurfaceType t) { return Surface(t); }), py::arg("surface_type"))
        .def_readwrite("surface_type", &Surface::surface_type,
            "This surface's SurfaceType. Assigning reclassifies it in place (geometry unchanged).")
        .def_readwrite("thickness", &Surface::thickness)
        .def_readwrite("bridge_angle", &Surface::bridge_angle)
        .def_readwrite("extra_perimeters", &Surface::extra_perimeters)
        .def_property("expolygon",
            [](Surface& s) -> ExPolygon& { return s.expolygon; },
            [](Surface& s, const ExPolygon& e) { s.expolygon = e; },
            py::return_value_policy::reference_internal,
            "This surface's geometry. Read returns a live ExPolygon ref; assign to replace it.")
        .def("area", [](const Surface& s) { return s.area(); })
        .def("is_top", [](const Surface& s) { return s.is_top(); })
        .def("is_bottom", [](const Surface& s) { return s.is_bottom(); })
        .def("is_bridge", [](const Surface& s) { return s.is_bridge(); })
        .def("is_internal", [](const Surface& s) { return s.is_internal(); })
        .def("is_external", [](const Surface& s) { return s.is_external(); })
        .def("is_solid", [](const Surface& s) { return s.is_solid(); });

    // SurfaceCollection: kept on py::nodelete — it is only ever a reference into the live
    // slicing graph (LayerRegion::slices/fill_surfaces), never constructed by a plugin.
    py::class_<SurfaceCollection, std::unique_ptr<SurfaceCollection, py::nodelete>>(host, "SurfaceCollection")
        .def("size", [](const SurfaceCollection& c) { return c.surfaces.size(); })
        .def("empty", [](const SurfaceCollection& c) { return c.empty(); })
        .def("clear", [](SurfaceCollection& c) { c.clear(); })
        .def("has", [](const SurfaceCollection& c, SurfaceType t) { return c.has(t); }, py::arg("surface_type"))
        .def("set_type", [](SurfaceCollection& c, SurfaceType t) { c.set_type(t); }, py::arg("surface_type"))
        .def("set", [](SurfaceCollection& c, const std::vector<ExPolygon>& src, SurfaceType t) { c.set(src, t); },
             py::arg("expolygons"), py::arg("surface_type"),
             "Replace all surfaces from a list of ExPolygon, all tagged `surface_type`.")
        .def("set", [](SurfaceCollection& c, const std::vector<Surface>& src) { c.set(src); },
             py::arg("surfaces"), "Replace all surfaces from a list of Surface (types preserved per surface).")
        .def("append", [](SurfaceCollection& c, const std::vector<ExPolygon>& src, SurfaceType t) { c.append(src, t); },
             py::arg("expolygons"), py::arg("surface_type"))
        .def("filter_by_type", [](py::object self, SurfaceType t) {
            SurfaceCollection& c = self.cast<SurfaceCollection&>();
            py::list out;
            // SurfaceCollection::filter_by_type returns SurfacesPtr, which is
            // std::vector<const Surface*> (see Surface.hpp), so iterate by const
            // pointer (py::cast accepts `const itype*` directly, see cast.h cast(const itype*)).
            for (const Surface* s : c.filter_by_type(t))
                out.append(py::cast(s, py::return_value_policy::reference_internal, self));
            return out;
        }, py::arg("surface_type"), "Surfaces of a given type as [Surface] refs. Invalidated by "
           "set()/append()/clear() on this collection (C++ vector semantics), same as .surfaces.")
        .def_property_readonly("surfaces", [](py::object self) {
            SurfaceCollection& c = self.cast<SurfaceCollection&>();
            py::list out;
            for (Surface& s : c.surfaces)
                out.append(py::cast(&s, py::return_value_policy::reference_internal, self));
            return out;
        }, "Surfaces as [Surface] references into the live collection. Invalidated by "
           "set()/append()/clear() on this collection (C++ vector semantics).");

    // --- Extrusion tree. Reads work as upstream. The zs fork adds the writable toolpath:
    // ExtrusionPath setters (set_points, width, height, mm3_per_mm, z_contoured) and the
    // collection mutators (append_path, remove, clear). Registered polymorphically: when a returned
    // ExtrusionEntity*'s dynamic type IS one of the classes registered below, pybind
    // hands the plugin that concrete type, so plugins walk the same tree shape C++ does.
    // When the dynamic type is NOT registered (e.g. ExtrusionLoopSloped, produced with
    // scarf seams), pybind falls back to the STATIC type at the cast site -- so such a
    // `.entities` child surfaces as a bare ExtrusionEntity (only .role is available).
    // flatten_paths() (a dynamic_cast walk) still yields proper ExtrusionPath leaves and
    // is the robust way to extract toolpaths.
    py::class_<ExtrusionEntity, std::unique_ptr<ExtrusionEntity, py::nodelete>>(host, "ExtrusionEntity")
        .def_property_readonly("role", [](const ExtrusionEntity& e) {
            return ExtrusionEntity::role_to_string(e.role());
        }, "Extrusion role as a human-readable string (e.g. \"Outer wall\", \"Sparse infill\").");

    py::class_<ExtrusionPath, ExtrusionEntity, std::unique_ptr<ExtrusionPath, py::nodelete>>(host, "ExtrusionPath")
        .def("points", [](py::object self) {
            const ExtrusionPath& p = self.cast<const ExtrusionPath&>();
            const Points3& pts = p.polyline.points;
            return with_numpy([&] {
                return py::object(make_readonly_rows<coord_t, 3>(
                    self, pts.empty() ? nullptr : pts.front().data(), (py::ssize_t) pts.size()));
            });
        }, "Path vertices as a read-only int64 (N,3) numpy view in scaled coords "
           "(the polyline is natively 3D on this branch). Requires numpy.")
        // set_points() installs a NEW buffer instead of handing out a writable view. The
        // polyline and its arc-fitting cache must change together, and only a call can
        // clear the cache at the moment the points change.
        .def("set_points", [](ExtrusionPath& p, py::handle src) {
            // Parse first: a malformed array raises ValueError and leaves the path untouched.
            Points3 pts = parse_points3(src, "ExtrusionPath.set_points");
            p.polyline.points = std::move(pts);
            // fitting_result caches the G2/G3 arcs that arc fitting computed from the OLD
            // points at posSimplifyPath. For a flat path with arc fitting on, GCode.cpp
            // prints that cache instead of the points, so a stale cache would print the
            // old geometry. An empty cache makes the writer emit G1 segments through the
            // new points. (A z_contoured path skips arc fitting anyway, GCode.cpp
            // _extrude_piece, but loop splitting and clipping still read the cache.)
            p.polyline.fitting_result.clear();
        }, py::arg("points"),
           "Replace all vertices from an (N,3) int64 ndarray in scaled coords, the layout "
           "points() returns (x, y, z). N >= 2. Clears the cached arc fitting, so the path "
           "prints as G1 segments. z is an offset from the layer's print_z, used only when "
           "z_contoured is True. Invalidates arrays previously returned by points(). Inside a "
           "loop, keep the end points of neighbouring paths equal. Raises ValueError on "
           "malformed input.")
        // The four scalars below are plain data members that the G-code writer reads when
        // it emits the path, so direct read-write access is enough: no cache depends on them.
        .def_readwrite("width", &ExtrusionPath::width,
            "Extrusion width in mm. Read-write. The preview draws it. The extruded volume "
            "comes from mm3_per_mm, not from width.")
        .def_readwrite("height", &ExtrusionPath::height,
            "Extrusion height in mm (the layer thickness this path fills). Read-write. With "
            "z_contoured, the writer scales the E of each segment by (height + dz) / height.")
        .def_readwrite("mm3_per_mm", &ExtrusionPath::mm3_per_mm,
            "Extruded volume per mm of head motion, in mm^3/mm. Read-write. The writer "
            "derives E from it.")
        .def_readwrite("z_contoured", &ExtrusionPath::z_contoured,
            "False (default): the z column is ignored and the path prints flat at the layer Z. "
            "True: each point prints at Z = layer Z + unscale(z), the E of each segment is "
            "scaled by (height + dz) / height with dz the z offset of the segment's end point "
            "(Ironing excepted, and not when flow_follows_z is False), arc fitting is skipped, "
            "and the travel into the path ends at the Z of its first point.")
        // A plain data member that the G-code writer reads at its two E-scaling sites
        // (GCode.cpp, _extrude_piece and the segmented emission path), so direct access is enough.
        .def_readwrite("flow_follows_z", &ExtrusionPath::flow_follows_z,
            "Read only for a z_contoured path. True (default): the writer scales the E of each "
            "segment by (height + dz) / height, right for a bead over a flat layer below. False: "
            "no scaling, right when the layer below carries the same z offsets (the gap under "
            "the bead stays `height`). Copied when Orca splits or copies the path.");

    py::class_<ExtrusionLoop, ExtrusionEntity, std::unique_ptr<ExtrusionLoop, py::nodelete>>(host, "ExtrusionLoop")
        .def_property_readonly("paths", [](py::object self) {
            ExtrusionLoop& l = self.cast<ExtrusionLoop&>();
            py::list out;
            for (ExtrusionPath& p : l.paths)
                out.append(py::cast(&p, py::return_value_policy::reference_internal, self));
            return out;
        }, "The loop's constituent paths as [ExtrusionPath].");

    py::class_<ExtrusionMultiPath, ExtrusionEntity, std::unique_ptr<ExtrusionMultiPath, py::nodelete>>(host, "ExtrusionMultiPath")
        .def_property_readonly("paths", [](py::object self) {
            ExtrusionMultiPath& m = self.cast<ExtrusionMultiPath&>();
            py::list out;
            for (ExtrusionPath& p : m.paths)
                out.append(py::cast(&p, py::return_value_policy::reference_internal, self));
            return out;
        }, "The multipath's constituent paths as [ExtrusionPath].");

    py::class_<ExtrusionEntityCollection, ExtrusionEntity,
               std::unique_ptr<ExtrusionEntityCollection, py::nodelete>>(host, "ExtrusionEntityCollection")
        .def("size", [](const ExtrusionEntityCollection& c) { return c.entities.size(); })
        .def_property_readonly("entities", [](py::object self) {
            ExtrusionEntityCollection& c = self.cast<ExtrusionEntityCollection&>();
            py::list out;
            for (ExtrusionEntity* e : c.entities)
                out.append(py::cast(e, py::return_value_policy::reference_internal, self));
            return out;
        }, "Child entities. Each is handed to you as its concrete type only when that type "
           "is registered; a child whose concrete type is unregistered (e.g. a scarf-seam "
           "ExtrusionLoopSloped) surfaces as a bare ExtrusionEntity exposing only .role. Use "
           "flatten_paths() to robustly reach every ExtrusionPath leaf.")
        .def("flatten_paths", [](py::object self) {
            const ExtrusionEntityCollection& c = self.cast<const ExtrusionEntityCollection&>();
            std::vector<const ExtrusionPath*> paths;
            collect_extrusion_paths(&c, paths);
            py::list out;
            for (const ExtrusionPath* p : paths)
                out.append(py::cast(const_cast<ExtrusionPath*>(p),
                                    py::return_value_policy::reference_internal, self));
            return out;
        }, "Every leaf ExtrusionPath under this tree (collections recursed into, "
           "loops/multipaths decomposed).")
        .def("append_path", [](py::object self, const std::string& role, py::handle points, float width,
                               float height, double mm3_per_mm, bool z_contoured) {
            ExtrusionEntityCollection& c = self.cast<ExtrusionEntityCollection&>();
            // Level check. The top level of LayerRegion.perimeters / .fills holds one child
            // collection per island, and GCode.cpp (the island loop of process_layer)
            // static_casts every child at that level to ExtrusionEntityCollection*. Its
            // assert is compiled out of Release builds, so a bare path there would be read
            // as a collection: undefined behavior, not an error. A collection that is empty
            // (its level cannot be told) or holds only collections is therefore refused.
            // Islands hold loops and paths, so appending to an island always passes.
            const bool collections_only = std::all_of(c.entities.begin(), c.entities.end(),
                [](const ExtrusionEntity* e) { return e->is_collection(); });
            if (collections_only)
                throw py::value_error("ExtrusionEntityCollection.append_path: this collection is empty or "
                                      "holds only collections (e.g. the top level of LayerRegion.perimeters "
                                      "or .fills); append to one of its child collections instead");
            // string_to_role() maps the strings .role returns ("Inner wall", "Sparse infill",
            // ...) back to the enum. It returns erNone for an unknown string. erMixed
            // ("Multiple") only describes a collection. Neither can label one path.
            const ExtrusionRole er = ExtrusionEntity::string_to_role(role);
            if (er == erNone || er == erMixed)
                throw py::value_error("ExtrusionEntityCollection.append_path: unknown or non-path role '" + role + "'");
            // The writer multiplies by mm3_per_mm and divides by height, so 0, a negative
            // value or NaN would print nothing, pull filament back, or emit inf.
            // `!(x > 0)` is also true for NaN, which `x <= 0` would let through.
            if (!(width > 0.f) || !(height > 0.f) || !(mm3_per_mm > 0.))
                throw py::value_error("ExtrusionEntityCollection.append_path: width, height and mm3_per_mm must be > 0");
            // Build the path in C++. Python only ever holds non-owning (py::nodelete)
            // references into the Print's graph, so a path made in Python could never be
            // handed over. The collection owns its children as raw pointers and deletes
            // them in clear(), remove() and its destructor. append() moves this stack
            // object into a heap copy that the collection owns from then on.
            ExtrusionPath path(er, mm3_per_mm, width, height);
            path.polyline.points = parse_points3(points, "ExtrusionEntityCollection.append_path");
            path.z_contoured = z_contoured;
            c.append(std::move(path));
            // Hand the new child back as a live reference, same policy as .entities, so
            // the plugin can keep editing it. reference_internal keeps this collection's
            // Python object alive while the reference exists.
            return py::cast(c.entities.back(), py::return_value_policy::reference_internal, self);
        }, py::arg("role"), py::arg("points"), py::arg("width"), py::arg("height"), py::arg("mm3_per_mm"),
           py::arg("z_contoured") = false,
           "Append a new ExtrusionPath built in C++ and return it. role: a string as .role "
           "returns (\"Inner wall\", \"Sparse infill\", ...). points: (N,3) int64 scaled coords, "
           "as for set_points(). width and height in mm, mm3_per_mm in mm^3/mm, all > 0. A new "
           "path carries no overhang data. Refused (ValueError) on an empty collection or one "
           "that holds only collections, such as the top level of LayerRegion.perimeters or "
           ".fills: append to one of its island collections. Existing references stay valid.")
        .def("remove", [](ExtrusionEntityCollection& c, py::ssize_t index) {
            // C++ remove() indexes the vector without a check, so the bounds check is here.
            if (index < 0 || static_cast<size_t>(index) >= c.entities.size())
                throw py::index_error("ExtrusionEntityCollection.remove: index out of range");
            c.remove(static_cast<size_t>(index));
        }, py::arg("index"),
           "Delete the child at index (0 <= index < size()). The child and everything under "
           "it is freed: a reference to it (from .entities or flatten_paths()) must not be "
           "used again, and later children move down by one index.")
        .def("clear", [](ExtrusionEntityCollection& c) { c.clear(); },
             "Delete every child. Every reference into this collection becomes invalid. An "
             "empty island prints nothing.")
        // no_sort is a plain data member that the G-code writer reads when it emits the layer,
        // so direct read-write access is enough. What it does depends on the level:
        //  - On a fill group (a child of LayerRegion.fills or .perimeters), it is honoured.
        //    GCode::ObjectByExtruder::Island::Region::append adds a group with no_sort set as ONE
        //    unit instead of spilling its paths into the chain list, and GCode::extrude_infill
        //    then prints that unit's children in stored order (it skips the re-chaining when
        //    no_sort is set). The unit itself is still placed among the region's other units by
        //    travel distance, and it is never reversed (can_reverse() is false with no_sort).
        //  - On the top level (LayerRegion.fills itself), GCode.cpp never reads it.
        .def_readwrite("no_sort", &ExtrusionEntityCollection::no_sort,
            "Read-write. True: the G-code writer prints this collection's children in stored "
            "order (no travel re-chaining, no reversal). Honoured on a fill group, the child "
            "of LayerRegion.fills / .perimeters; the group itself is still ordered among the "
            "region's other groups by travel. Ignored on the top-level collection.");

    py::class_<PrintRegion, std::unique_ptr<PrintRegion, py::nodelete>>(host, "PrintRegion")
        .def("config_keys", [](const PrintRegion& r) { return r.config().keys(); })
        .def("config_value", [](const PrintRegion& r, const std::string& key) {
            return config_value_or_none(r.config(), key);
        }, py::arg("key"),
           "Serialized value of this region's resolved config option, or None if absent.");

    auto layer_region = py::class_<LayerRegion, std::unique_ptr<LayerRegion, py::nodelete>>(host, "LayerRegion");
    layer_region
        .def_readonly("slices", &LayerRegion::slices,
            "Sliced, typed surfaces (SurfaceCollection). Edit in place, or replace with "
            "slices.set(expolygons, surface_type). At Step.posSlice this is the primary mutation "
            "target; the split slice loop runs make_perimeters() afterward so edits cascade downstream.")
        .def_readonly("fill_surfaces", &LayerRegion::fill_surfaces,
            "Surfaces prepared for infill (SurfaceCollection). Edit in place or via fill_surfaces.set(...).")
        .def_readonly("perimeters", &LayerRegion::perimeters,
            "Perimeter toolpaths (ExtrusionEntityCollection): one child collection per island, "
            "each holding loops and paths. Edit paths through the ExtrusionPath setters and "
            "islands through append_path / remove / clear.")
        .def_readonly("fills", &LayerRegion::fills,
            "Infill toolpaths (ExtrusionEntityCollection): one child collection per fill "
            "group. Same editing rules as perimeters.")
        .def("layer", [](LayerRegion& r) -> py::object {
            Layer* l = r.layer();
            if (l == nullptr)
                return py::none();
            return py::cast(l, py::return_value_policy::reference);
        }, "Owning Layer, or None.")
        .def("region", [](LayerRegion& r) -> const PrintRegion& { return r.region(); },
             py::return_value_policy::reference,
             "This region's PrintRegion (resolved per-region settings).")
        .def("config_value", [](const LayerRegion& r, const std::string& key) {
            return config_value_or_none(r.region().config(), key);
        }, py::arg("key"),
           "Serialized value of this region's resolved config option, or None if absent.");

    auto layer = py::class_<Layer, std::unique_ptr<Layer, py::nodelete>>(host, "Layer");
    layer
        .def_readonly("print_z", &Layer::print_z)
        .def_readonly("slice_z", &Layer::slice_z)
        .def_readonly("height", &Layer::height)
        .def_property_readonly("upper_layer", [](Layer& l) -> py::object {
            if (l.upper_layer == nullptr) return py::none();
            return py::cast(l.upper_layer, py::return_value_policy::reference);
        }, "The layer above, or None (graph navigation, like C++).")
        .def_property_readonly("lower_layer", [](Layer& l) -> py::object {
            if (l.lower_layer == nullptr) return py::none();
            return py::cast(l.lower_layer, py::return_value_policy::reference);
        }, "The layer below, or None.")
        .def("regions", [](py::object self) {
            Layer& l = self.cast<Layer&>();
            py::list out;
            for (LayerRegion* r : l.regions())
                out.append(py::cast(r, py::return_value_policy::reference_internal, self));
            return out;
        }, "Per-region data as [LayerRegion].")
        .def("make_slices", [](Layer& l) {
            l.make_slices();
            refresh_lslices_bboxes(l);
        }, "Re-derive lslices (merged islands) from the region slices and refresh the bbox "
           "cache — the C++ invariant-maintenance call after in-place slice edits.")
        .def("lslices", [](py::object self) {
            Layer& l = self.cast<Layer&>();
            py::list out;
            for (ExPolygon& e : l.lslices)
                out.append(py::cast(&e, py::return_value_policy::reference_internal, self));
            return out;
        }, "Merged per-layer islands as [ExPolygon] refs (in-place editable). Derived from the "
           "region slices; call make_slices() to re-derive after edits. Invalidated by make_slices().");

    py::class_<PrintObject, std::unique_ptr<PrintObject, py::nodelete>>(host, "PrintObject")
        .def("id", [](const PrintObject& o) { return o.id().id; },
             "Stable numeric object id (ObjectBase::id()).")
        .def("layers", [](py::object self) {
            PrintObject& o = self.cast<PrintObject&>();
            py::list out;
            for (Layer* l : o.layers())
                out.append(py::cast(l, py::return_value_policy::reference_internal, self));
            return out;
        }, "Object layers, bottom-up, as [Layer].")
        .def("support_layers", [](py::object self) {
            PrintObject& o = self.cast<PrintObject&>();
            py::list out;
            for (SupportLayer* sl : o.support_layers())
                out.append(py::cast(static_cast<Layer*>(sl),
                                    py::return_value_policy::reference_internal, self));
            return out;
        }, "Support layers as [Layer] (support-specific fields are not exposed).")
        .def("model_object", [](PrintObject& o) -> py::object {
            // The Print's model SNAPSHOT (worker-thread stable), reusing the
            // orca.host.ModelObject bindings — mesh access for slicing plugins.
            // o is non-const here, so model_object() already returns a non-const ModelObject*.
            return py::cast(o.model_object(), py::return_value_policy::reference);
        }, "The source orca.host.ModelObject from the Print's own model snapshot.")
        .def("bounding_box", [](const PrintObject& o) {
            const BoundingBox bb = o.bounding_box();
            return py::make_tuple(bb.min.x(), bb.min.y(), bb.max.x(), bb.max.y());
        }, "Object XY bounding box in scaled coords as (min_x, min_y, max_x, max_y). The "
           "sliced polygons live in this same frame, so its midpoint is the footprint center.")
        .def("trafo", [](const PrintObject& o) { return mat4_to_numpy(o.trafo()); },
             "Object-to-print 4x4 float64 affine matrix (copy). Requires numpy.")
        .def("config_keys", [](const PrintObject& o) { return o.config().keys(); })
        .def("config_value", [](const PrintObject& o, const std::string& key) {
            return config_value_or_none(o.config(), key);
        }, py::arg("key"),
           "Serialized value of a resolved per-object config option, or None if absent.");

    py::class_<Print, std::unique_ptr<Print, py::nodelete>>(host, "Print")
        .def("objects", [](py::object self) {
            Print& p = self.cast<Print&>();
            py::list out;
            for (PrintObject* o : p.objects())
                out.append(py::cast(o, py::return_value_policy::reference_internal, self));
            return out;
        }, "The print's objects as [PrintObject].")
        .def("model", [](Print& p) -> Model& { return const_cast<Model&>(p.model()); },
             py::return_value_policy::reference_internal,
             "The Print's own Model snapshot (worker-thread stable). Inside slicing "
             "hooks use THIS — never orca.host.model(), which is the live GUI model "
             "owned by another thread.")
        .def("config_keys", [](const Print& p) { return p.full_print_config().keys(); })
        .def("config_value", [](const Print& p, const std::string& key) {
            return config_value_or_none(p.full_print_config(), key);
        }, py::arg("key"),
           "Serialized value of the resolved (full) print config for this slice, or None.")
        .def("canceled", [](const Print& p) { return p.canceled(); },
             "True once cancellation was requested (prefer ctx.cancelled()).");
}

} // namespace Slic3r
